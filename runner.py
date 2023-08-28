# coding: utf-8
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


class DefaultRunner(object):
        def __init__(self, train_set, val_set, model, optimizer, scheduler, tokenizer, gpus, config, distributed=False):
                self.train_set = train_set
                self.val_set = val_set
                self.gpus = gpus
                self.distributed = distributed
                self.device = gpus[0] if self.distributed else torch.device(gpus[0]) if len(gpus) > 0 else torch.device(
                        'cpu')
                self.config = config
                self._model = model
                self._optimizer = optimizer
                self._scheduler = scheduler
                self._tokenizer = tokenizer

                self.best_loss = np.inf
                self.start_epoch = 0

                if self.distributed:
                        self._model = self._model.cuda(self.device)
                        self._model = DDP(self._model, device_ids=[self.device])
                        self.batch_size = int(self.config.train.batch_size / dist.get_world_size())
                elif self.device.type == 'cuda':
                        self._model = self._model.cuda(self.device)

        def collate_fn(self, batch):
                token_encoding = self._tokenizer.batch_encode_plus([i[1] for i in batch],
                                                                   add_special_tokens=True,
                                                                   padding='longest',
                                                                   return_tensors='pt')
                return torch.tensor([i[0] for i in batch]), token_encoding

        def save(self, checkpoint, epoch=None, var_list={}):
                state = {
                        **var_list,
                        "model": self._model.state_dict(),
                        "optimizer": self._optimizer.state_dict(),
                        "scheduler": self._scheduler.state_dict(),
                        "config": self.config
                }
                epoch = str(epoch) if epoch is not None else ''
                checkpoint = os.path.join(checkpoint, 'checkpoint%s' % epoch)
                torch.save(state, checkpoint)

        def load(self, checkpoint, epoch=None, load_optimizer=False, load_scheduler=False, map_location=None):
                epoch = str(epoch) if epoch is not None else ''
                checkpoint = os.path.join(checkpoint, 'checkpoint%s' % epoch)
                print("Load checkpoint from %s" % checkpoint)

                state = torch.load(checkpoint, map_location=self.device if map_location is None else map_location)
                self._model.load_state_dict(state["model"])
                # self._model.load_state_dict(state["model"], strict=False)
                self.best_loss = state['best_loss']
                self.start_epoch = state['cur_epoch'] + 1

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
                                losses_meter.update(loss.item(), batch.shape[0])

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
                average_loss = losses_meter.avg

                if verbose:
                        if (not self.distributed) or (self.distributed and (self.device == 0)):
                                print('Evaluate %s Loss: %.5f of %d samples | Time: %.5f' % (
                                split, average_loss, losses_meter.count, time() - eval_start))

                return average_loss

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

                model = self._model
                train_losses = []
                val_losses = []
                loss_fn = torch.nn.CrossEntropyLoss()
                best_loss = self.best_loss
                start_epoch = self.start_epoch
                print('start training...')

                for epoch in range(num_epochs):
                        # train
                        if train_sampler is not None:
                                train_sampler.set_epoch(epoch)
                        model.train()
                        epoch_start = time()
                        # batch_losses = []
                        losses_meter = utils.AverageMeter('Loss', self.device, ':.2e')
                        batch_cnt = 0
                        for batch in dataloader:
                                batch_cnt += 1
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
                                losses_meter.update(loss.item(), batch.shape[0])
                                # if not isinstance(self.device, torch.device):
                                #     torch.distributed.barrier()
                                #     loss_print = reduce_mean(loss)
                                # else:
                                #     loss_print = loss
                                if not loss.requires_grad:
                                        raise RuntimeError("loss doesn't require grad")
                                self._optimizer.zero_grad()
                                loss.backward()
                                self._optimizer.step()
                                # batch_losses.append(loss_print.item())

                                if batch_cnt % self.config.train.log_interval == 0 or (epoch == 0 and batch_cnt <= 10):
                                        # if batch_cnt % self.config.train.log_interval == 0:
                                        if (not self.distributed) or (self.distributed and (self.device == 0)):
                                                print('Epoch: %d | Step: %d | loss: %.5f | Lr: %.5f' % \
                                                      (epoch + start_epoch, batch_cnt, losses_meter.val,
                                                       self._optimizer.param_groups[0]['lr']))

                        if self.distributed:
                                losses_meter.all_reduce()
                        average_loss = losses_meter.avg
                        train_losses.append(average_loss)

                        if verbose:
                                if (not self.distributed) or (self.distributed and (self.device == 0)):
                                        print('Epoch: %d | Train Loss: %.5f | Time: %.5f' % (
                                        epoch + start_epoch, average_loss, time() - epoch_start))

                        # evaluate
                        if self.config.train.eval:
                                average_eval_loss = self.evaluate('val', verbose=1, epoch=epoch)
                                val_losses.append(average_eval_loss)
                        else:
                                # use train loss as surrogate loss
                                average_eval_loss = average_loss
                                val_losses.append(average_loss)

                        if self.config.train.scheduler.type == "plateau":
                                self._scheduler.step(average_eval_loss)
                        else:
                                self._scheduler.step()

                        if val_losses[-1] < best_loss:
                                best_loss = val_losses[-1]
                                if self.config.train.save and (
                                        (not self.distributed) or (self.distributed and (self.device == 0))):
                                        val_list = {
                                                'cur_epoch': epoch + start_epoch,
                                                'best_loss': best_loss,
                                        }
                                        self.save(self.config.train.save_path, epoch + start_epoch, val_list)
                self.best_loss = best_loss
                self.start_epoch = start_epoch + num_epochs
                print('optimization finished.')
                print('Total time elapsed: %.5fs' % (time() - train_start))