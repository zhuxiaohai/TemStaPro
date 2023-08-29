import datetime
from time import time
import os

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
        self.logger = TensorboardLogger(os.path.join(self.config.train.log_dir, self.timestamp_id))

    def collate_fn(self, batch):
        token_encoding = self._tokenizer.batch_encode_plus([i[1] for i in batch],
                                                           add_special_tokens=True,
                                                           padding='longest',
                                                           return_tensors='pt')
        return torch.tensor([i[0] for i in batch]), token_encoding

    def save(self, checkpoint_dir, filename, var_list={}):
        state = {
                **var_list,
                "model": self._model.state_dict(),
                "optimizer": self._optimizer.state_dict(),
                "scheduler": self._scheduler.state_dict(),
                "config": self.config
        }
        checkpoint = os.path.join(checkpoint_dir, filename)
        torch.save(state, checkpoint)

    def load(self, checkpoint, load_optimizer=False, load_scheduler=False, map_location=None):
        print("Load checkpoint from %s" % checkpoint)

        state = torch.load(checkpoint, map_location=self.device if map_location is None else map_location)
        self._model.load_state_dict(state["model"])
        # self._model.load_state_dict(state["model"], strict=False)
        self.best_loss = state['best_loss']
        self.epoch = state['cur_epoch']
        self.step = state['cur_step']

        if load_optimizer:
            self._optimizer.load_state_dict(state["optimizer"])
        if self.distributed or (self.device.type == 'cuda'):
            for state in self._optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda(self.device)

        if load_scheduler:
            self._scheduler.load_state_dict(state["scheduler"])

    @torch.no_grad()
    def evaluate(self, split, verbose=0, epoch=None):
        def run_validate(loader):
            loss_fn = torch.nn.CrossEntropyLoss()
            for labels, batch in loader:
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
                (self.device == 0) and (dist.get_rank() == 0):
            aux_val_dataset = Subset(dataloader.dataset,
                                     range(len(dataloader.sampler) * dist.get_world_size(),
                                           len(dataloader.dataset)))
            aux_val_loader = DataLoader(
                    aux_val_dataset, batch_size=self.batch_size, shuffle=False,
                    num_workers=self.config.train.num_workers, pin_memory=pin_memory)
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
            if (not self.distributed) or (self.distributed and (self.device == 0)):
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
        best_loss = self.best_loss
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
                (labels, batch) = next(train_loader_iter)

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
                acc = utils.accuracy(outputs, labels, (1, ))
                top1_meter.update(acc[0], labels.size(0))

                if not loss.requires_grad:
                    raise RuntimeError("loss doesn't require grad")
                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()

                if batch_cnt % self.config.train.log_interval == 0 or (epoch == 0 and batch_cnt <= 10):
                    if (not self.distributed) or (self.distributed and (self.device == 0)):
                        print('Epoch: %d | Step: %d | loss: %.5f | Acc@1: %.5f | Lr: %.5f | Time: %.5f' % \
                              (epoch + start_epoch, batch_cnt, losses_meter.avg, top1_meter.avg,
                               self._optimizer.param_groups[0]['lr'], time() - epoch_start))
                        if batch_cnt % self.config.train.log_interval == 0:
                            if self.logger is not None:
                                log_dict = {losses_meter.name: losses_meter.avg,   top1_meter.name: top1_meter.avg}
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
                                self.save(self.config.train.save_path, 'checkpoint.pt', val_list)

                            # evaluate
                            if self.config.train.eval:
                                average_eval_loss, average_eval_top1 = self.evaluate('val', verbose=1, epoch=epoch)
                                if average_eval_loss < self.best_loss:
                                    self.best_loss = average_eval_loss

                                    if self.config.train.save:
                                        val_list = {
                                            'cur_epoch': self.epoch,
                                            'cur_step': self.step,
                                            'best_loss': best_loss,
                                            'best_acc': average_eval_top1
                                        }
                                        self.save(self.config.train.save_path, 'best_checkpoint.pt', val_list)

            if self.distributed:
                losses_meter.all_reduce()
                top1_meter.all_reduce()

            average_loss = losses_meter.avg
            average_top1 = top1_meter.avg

            if verbose:
                if (not self.distributed) or (self.distributed and (self.device == 0)):
                    print('Epoch: %d | Train Loss: %.5f | Train Acc@1: %.5f | Time: %.5f' % (
                        epoch + start_epoch, average_loss, average_top1, time() - epoch_start))

            scheduler_loss = average_loss
            if not self.config.train.eval:
                scheduler_loss = average_eval_loss

            if self.config.train.scheduler.type == "plateau":
                self._scheduler.step(scheduler_loss)
            elif self._scheduler is not None:
                self._scheduler.step()

        print('optimization finished.')
        print('Total time elapsed: %.5fs' % (time() - train_start))