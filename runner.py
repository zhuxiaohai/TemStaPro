import datetime
from time import time
import os
from tqdm import tqdm
from collections import defaultdict

import numpy as np

import torch
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Subset

import utils
import dist_utils
from logger import TensorboardLogger


class DefaultRunner(object):
    def __init__(self, train_set, val_set, model, optimizer, scheduler, tokenizer, gpus, config, distributed=False):
        self.train_set = train_set
        self.val_set = val_set
        self.gpus = gpus
        self.distributed = distributed
        self.device = gpus[0] if self.distributed else torch.device(gpus[0]) if len(gpus) > 0 else torch.device('cpu')
        self.config = config
        self._model = model
        self._optimizer = optimizer
        self._scheduler = scheduler
        self._tokenizer = tokenizer

        self.best_loss = np.inf
        self.step = 0
        self.epoch = 0

        if self.distributed:
            self._model = self._model.cuda(self.device)
            self._model = DDP(self._model, device_ids=[self.device], find_unused_parameters=True)
            self.batch_size = int(self.config.train.batch_size / dist.get_world_size())
        elif self.device.type == 'cuda':
            self._model = self._model.cuda(self.device)
            self.batch_size = int(self.config.train.batch_size)

        timestamp = torch.tensor(datetime.datetime.now().timestamp()).to(self.device)
        # create directories from master rank only
        dist_utils.broadcast(timestamp, 0)
        timestamp = datetime.datetime.fromtimestamp(timestamp.float().item()).strftime("%Y-%m-%d-%H-%M-%S")
        self.timestamp_id = timestamp
        self.logs_dir = os.path.join(self.config.train.log_dir, self.timestamp_id)
        self.results_dir = os.path.join(self.config.test.output_path, self.timestamp_id)
        self.checkpoints_dir = os.path.join(self.config.train.save_path, self.timestamp_id)

        if (not self.distributed) or (self.distributed and dist_utils.is_master()):
            os.makedirs(self.logs_dir, exist_ok=True)
            os.makedirs(self.results_dir, exist_ok=True)
            os.makedirs(self.checkpoints_dir, exist_ok=True)
            self.logger = TensorboardLogger(self.logs_dir)
        else:
            self.logger = None

    def collate_fn(self, batch):
        token_encoding = self._tokenizer.batch_encode_plus([i[-1] for i in batch],
                                                           add_special_tokens=True,
                                                           padding='longest',
                                                           return_tensors='pt')
        return torch.tensor([i[0] for i in batch]), [str(i[1]) for i in batch], token_encoding

    def save(self, filename, var_list={}):
        state = {
            **var_list,
            "model": self._model.state_dict(),
            "optimizer": self._optimizer.state_dict(),
            "scheduler": self._scheduler.state_dict(),
            "config": self.config
        }
        checkpoint = os.path.join(self.checkpoints_dir, filename)
        torch.save(state, checkpoint)

    def load(self, checkpoint, load_optimizer=False, load_scheduler=False, map_location=None):
        print("Load checkpoint from %s" % checkpoint)

        state = torch.load(checkpoint, map_location=self.device if map_location is None else map_location)
        self._model.load_state_dict(state["model"])
        # self._model.load_state_dict(state["model"], strict=False)
        try:
            self.best_loss = state['best_loss']
        except:
            self.best_loss = state['cur_loss']
        self.epoch = state['cur_epoch']
        self.step = state['cur_step']

        if load_optimizer:
            self._optimizer.load_state_dict(state["optimizer"])
        if self.distributed or (self.device.type == 'cuda'):
            for optimizer_state in self._optimizer.state.values():
                for k, v in optimizer_state.items():
                    if isinstance(v, torch.Tensor):
                        optimizer_state[k] = v.cuda(self.device)

        if load_scheduler:
            self._scheduler.load_state_dict(state["scheduler"])

    def save_results(
        self, predictions, results_file, keys
    ):
        if results_file is None:
            return

        results_file_path = os.path.join(
            self.results_dir,
            f"{results_file}_{dist_utils.get_rank()}.npz",
        )
        np.savez_compressed(
            results_file_path,
            ids=predictions["id"],
            **{key: predictions[key] for key in keys},
        )

        dist_utils.synchronize()
        if dist_utils.is_master():
            gather_results = defaultdict(list)
            full_path = os.path.join(
                self.results_dir,
                f"{results_file}.npz",
            )

            for i in range(dist_utils.get_world_size()):
                rank_path = os.path.join(
                    self.results_dir,
                    f"{results_file}_{i}.npz",
                )
                rank_results = np.load(rank_path, allow_pickle=True)
                gather_results["ids"].extend(rank_results["ids"])
                for key in keys:
                    if key.find("forces") >= 0:
                        gather_results[key].extend(np.array_split(rank_results[key], np.cumsum(rank_results['chunk_idx'])[: -1]))
                    else:
                        gather_results[key].extend(rank_results[key])
                os.remove(rank_path)

            # Because of how distributed sampler works, some system ids
            # might be repeated to make no. of samples even across GPUs.
            _, idx = np.unique(gather_results["ids"], return_index=True)
            gather_results["ids"] = np.array(gather_results["ids"])[idx]
            for k in keys:
                if k.find("forces") >= 0:
                    gather_results[k] = np.concatenate(
                        [gather_results[k][idx_i] for idx_i in idx]
                    )
                elif k == "chunk_idx":
                    gather_results[k] = np.cumsum(
                        np.array(gather_results[k])[idx]
                    )[:-1]
                else:
                    gather_results[k] = np.array(gather_results[k])[idx]

            print(f"Writing results to {full_path}")
            np.savez_compressed(full_path, **gather_results)

    @torch.no_grad()
    def predict(
        self,
        split,
        results_file=None,
        disable_tqdm: bool = False,
    ):
        if split not in ['train', 'val', 'test']:
            raise ValueError('split should be either train, val, or test.')

        test_set = getattr(self, "%s_set" % split)

        if self.distributed:
            test_sampler = DistributedSampler(test_set, shuffle=False)
            pin_memory = True
        else:
            test_sampler = None
            pin_memory = False
        dataloader = DataLoader(test_set, batch_size=self.batch_size,
                                shuffle=False, num_workers=self.config.train.num_workers,
                                pin_memory=pin_memory, sampler=test_sampler, collate_fn=self.collate_fn)

        rank = dist_utils.get_rank()

        model = self._model
        model.eval()

        predictions = {"id": [], "res": []}

        for _, (_, pdb_ids, batch) in tqdm(
            enumerate(dataloader),
            total=len(dataloader),
            position=rank,
            desc="rank {}".format(rank),
            disable=disable_tqdm,
        ):
            if self.distributed or self.device.type == "cuda":
                if not self.distributed:
                    batch['input_ids'] = batch['input_ids'].to(self.device)
                    batch['attention_mask'] = batch['attention_mask'].to(self.device)
                else:
                    batch['input_ids'] = batch['input_ids'].cuda(self.device, non_blocking=True)
                    batch['attention_mask'] = batch['attention_mask'].cuda(self.device, non_blocking=True)

            out = model(batch)
            predictions["id"].extend(
                pdb_ids
            )
            predictions["res"].extend(
                out.cpu().detach().numpy()
            )

        self.save_results(predictions, results_file, keys=["res"])

        return predictions

    @torch.no_grad()
    def evaluate(self, split, verbose=0, epoch=None):
        def run_validate(loader):
            loss_fn = torch.nn.CrossEntropyLoss()
            batch_cnt = 0
            for labels, _, batch in loader:
                batch_cnt += 1
                if self.distributed or self.device.type == "cuda":
                    if not self.distributed:
                        batch['input_ids'] = batch['input_ids'].to(self.device)
                        batch['attention_mask'] = batch['attention_mask'].to(self.device)
                        labels = labels.to(self.device)
                    else:
                        batch['input_ids'] = batch['input_ids'].cuda(self.device, non_blocking=True)
                        batch['attention_mask'] = batch['attention_mask'].cuda(self.device, non_blocking=True)
                        labels = labels.cuda(self.device, non_blocking=True)

                outputs = model(batch)
                loss = loss_fn(outputs, labels)
                losses_meter.update(loss.item(), labels.size(0))
                acc = utils.accuracy(outputs, labels, (1,))
                top1_meter.update(acc[0], labels.size(0))

        if split not in ['train', 'val', 'test']:
            raise ValueError('split should be either train, val, or test.')

        test_set = getattr(self, "%s_set" % split)
        if self.distributed:
            test_sampler = DistributedSampler(test_set, shuffle=False, drop_last=True)
            pin_memory = True
        else:
            test_sampler = None
            pin_memory = False
        dataloader = DataLoader(test_set, batch_size=self.batch_size,
                                shuffle=False, num_workers=self.config.train.num_workers,
                                pin_memory=pin_memory, sampler=test_sampler, collate_fn=self.collate_fn)
        model = self._model
        model.eval()
        if test_sampler is not None:
            test_sampler.set_epoch(epoch)

        # code here
        eval_start = time()
        losses_meter = utils.AverageMeter('Loss', self.device, ':.2e')
        top1_meter = utils.AverageMeter('Acc@1', self.device, ':.2e')

        run_validate(dataloader)
        if self.distributed and (len(dataloader.sampler) * dist.get_world_size() < len(dataloader.dataset)) and \
                dist_utils.is_master():
            aux_val_dataset = Subset(dataloader.dataset,
                                     range(len(dataloader.sampler) * dist.get_world_size(),
                                           len(dataloader.dataset)))
            aux_val_loader = DataLoader(
                aux_val_dataset, batch_size=self.batch_size, shuffle=False,
                num_workers=self.config.train.num_workers, pin_memory=pin_memory, collate_fn=self.collate_fn)
            run_validate(aux_val_loader)

        if self.distributed:
            losses_meter.all_reduce()
            top1_meter.all_reduce()

        average_loss = losses_meter.avg
        average_top1 = top1_meter.avg

        if self.logger is not None:
            log_dict = {losses_meter.name: losses_meter.avg, top1_meter.name: top1_meter.avg}
            log_dict.update(
                {
                    "epoch": self.epoch,
                }
            )
            self.logger.log(
                log_dict,
                step=self.step,
                split=split,
            )

        if verbose:
            if (not self.distributed) or (self.distributed and dist_utils.is_master()):
                print('Evaluate %s Loss: %.5f, Acc@1 %.5f of %d samples | Time: %.5f' % (
                    split, average_loss, average_top1, losses_meter.count, time() - eval_start))

        return average_loss, average_top1

    def train(self, verbose=1):
        train_start = time()

        num_epochs = self.config.train.epochs
        if self.distributed:
            train_sampler = DistributedSampler(self.train_set)
            pin_memory = True
            shuffle = False
        else:
            train_sampler = None
            pin_memory = False
            shuffle = self.config.train.shuffle
        dataloader = DataLoader(self.train_set, batch_size=self.batch_size,
                                shuffle=shuffle, num_workers=self.config.train.num_workers,
                                pin_memory=pin_memory, sampler=train_sampler, collate_fn=self.collate_fn)
        size_of_loader = len(dataloader)

        model = self._model
        loss_fn = torch.nn.CrossEntropyLoss()
        start_epoch = self.step // size_of_loader
        print('start training...')

        for epoch in range(start_epoch, num_epochs):
            # train
            epoch_start = time()

            if train_sampler is not None:
                train_sampler.set_epoch(epoch)

            model.train()

            losses_meter = utils.AverageMeter('Loss', self.device, ':.2e')
            top1_meter = utils.AverageMeter('Acc@1', self.device, ':.2e')

            skip_steps = self.step % size_of_loader
            train_loader_iter = iter(dataloader)
            batch_cnt = 0
            for i in range(skip_steps, size_of_loader):
                (labels, _, batch) = next(train_loader_iter)

                batch_cnt += 1
                self.epoch = epoch + (i + 1) / size_of_loader
                self.step = start_epoch * len(dataloader) + i + 1

                if self.distributed or self.device.type == 'cuda':
                    if not self.distributed:
                        batch['input_ids'] = batch['input_ids'].to(self.device)
                        batch['attention_mask'] = batch['attention_mask'].to(self.device)
                        labels = labels.to(self.device)
                    else:
                        batch['input_ids'] = batch['input_ids'].cuda(self.device, non_blocking=True)
                        batch['attention_mask'] = batch['attention_mask'].cuda(self.device, non_blocking=True)
                        labels = labels.cuda(self.device, non_blocking=True)

                outputs = model(batch)
                loss = loss_fn(outputs, labels)
                losses_meter.update(loss.item(), labels.size(0))
                acc = utils.accuracy(outputs, labels, (1,))
                top1_meter.update(acc[0], labels.size(0))

                if not loss.requires_grad:
                    raise RuntimeError("loss doesn't require grad")
                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()

                if (batch_cnt % self.config.train.log_interval == 0) or ((epoch == 0) and (batch_cnt <= 10)):
                    if (not self.distributed) or (self.distributed and dist_utils.is_master()):
                        print('Epoch: %d | Step: %d | loss: %.5f | Acc@1: %.5f | Lr: %.5f | Time: %.5f' % \
                              (epoch, batch_cnt, losses_meter.avg, top1_meter.avg,
                               self._optimizer.param_groups[0]['lr'], time() - epoch_start))

                if batch_cnt % self.config.train.log_interval == 0:
                    if self.logger is not None:
                        log_dict = {losses_meter.name: losses_meter.avg, top1_meter.name: top1_meter.avg}
                        log_dict.update(
                            {
                                "lr": self._optimizer.param_groups[0]['lr'],
                                "epoch": self.epoch,
                                "step": self.step,
                            }
                        )
                        self.logger.log(
                            log_dict,
                            step=self.step,
                            split="train",
                        )

                    if self.config.train.save:
                        val_list = {
                            'cur_epoch': self.epoch,
                            'cur_step': self.step,
                            'cur_loss': losses_meter.avg,
                            'cur_acc': top1_meter.avg
                        }
                        self.save('checkpoint.pt', val_list)

                    if self.config.train.eval and self.config.train.eval_step:
                        average_eval_loss, average_eval_top1 = self.evaluate('val', verbose=1, epoch=epoch)
                        if average_eval_loss < self.best_loss:
                            self.best_loss = average_eval_loss
                            if self.config.train.save:
                                val_list = {
                                    'cur_epoch': self.epoch,
                                    'cur_step': self.step,
                                    'best_loss': self.best_loss,
                                    'best_acc': average_eval_top1
                                }
                                self.save('best_checkpoint.pt', val_list)

                if self.config.train.scheduler.type == "plateau":
                    if (batch_cnt % self.config.train.log_interval == 0) and self.config.train.eval and self.config.train.eval_step:
                        self._scheduler.step(average_eval_loss)
                elif self._scheduler is not None:
                    self._scheduler.step()

                losses_meter.reset()
                top1_meter.reset()

            if verbose:
                if (not self.distributed) or (self.distributed and (self.device == 0)):
                    print('Epoch: %d | Train time: %.5f' % (epoch, time() - epoch_start))

            if self.config.train.eval and self.config.train.eval_step:
                pass
            elif self.config.train.eval and (not self.config.train.eval_step):
                average_eval_loss, average_eval_top1 = self.evaluate('val', verbose=1, epoch=epoch)
                if average_eval_loss < self.best_loss:
                    self.best_loss = average_eval_loss
                    if self.config.train.save:
                        val_list = {
                            'cur_epoch': self.epoch,
                            'cur_step': self.step,
                            'best_loss': self.best_loss,
                            'best_acc': average_eval_top1
                        }
                        self.save('best_checkpoint.pt', val_list)

                if self.config.train.scheduler.type == "plateau":
                    self._scheduler.step(average_eval_loss)
                elif self._scheduler is not None:
                    self._scheduler.step()

        print('optimization finished.')
        print('Total time elapsed: %.5fs' % (time() - train_start))