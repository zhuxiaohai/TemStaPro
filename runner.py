# coding: utf-8
from time import time
import os

import numpy as np
import torch
from torch.utils.data import DataLoader


class DefaultRunner(object):
    def __init__(self, train_set, val_set, model, optimizer, scheduler, config):
        self.train_set = train_set
        self.val_set = val_set
        self.gpus = config.train.gpus
        self.device = config.train.device
        self.config = config

        self.batch_size = self.config.train.batch_size

        self._model = model
        self._optimizer = optimizer
        self._scheduler = scheduler

        self.best_loss = np.inf
        self.start_epoch = 0

        if self.device.type == 'cuda':
            self._model = self._model.cuda(self.device)

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

    def load(self, checkpoint, epoch=None, load_optimizer=False, load_scheduler=False):
        epoch = str(epoch) if epoch is not None else ''
        checkpoint = os.path.join(checkpoint, 'checkpoint%s' % epoch)
        print("Load checkpoint from %s" % checkpoint)

        state = torch.load(checkpoint, map_location=self.device)
        self._model.load_state_dict(state["model"])
        # self._model.load_state_dict(state["model"], strict=False)
        self.best_loss = state['best_loss']
        self.start_epoch = state['cur_epoch'] + 1

        if load_optimizer:
            self._optimizer.load_state_dict(state["optimizer"])
            if self.device.type == 'cuda':
                for state in self._optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.cuda(self.device)

        if load_scheduler:
            self._scheduler.load_state_dict(state["scheduler"])

    @torch.no_grad()
    def evaluate(self, split, verbose=0):
        """
        Evaluate the model.
        Parameters:
            split (str): split to evaluate. Can be ``train``, ``val`` or ``test``.
        """
        if split not in ['train', 'val', 'test']:
            raise ValueError('split should be either train, val, or test.')

        # test_set = getattr(self, "%s_set" % split)
        # dataloader = DataLoader(test_set, batch_size=self.config.train.batch_size, \
        #                         shuffle=False, num_workers=self.config.train.num_workers,
        #                         pin_memory=True)
        dataloader = getattr(self, "%s_set" % split)
        model = self._model
        model.eval()
        # code here
        eval_start = time()
        eval_losses = []
        loss_fn = torch.nn.CrossEntropyLoss()
        for labels, batch in dataloader:
            if self.device.type == "cuda":
                batch['input_ids'] = batch['input_ids'].to(self.device, non_blocking=True)
                batch['attention_mask'] = batch['attention_mask'].to(self.device, non_blocking=True)
                # batch = batch.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

            outputs = model(batch)
            loss = loss_fn(outputs, labels)
            eval_losses.append(loss.item())
        average_loss = sum(eval_losses) / len(eval_losses)

        if verbose:
            print('Evaluate %s Loss: %.5f | Time: %.5f' % (split, average_loss, time() - eval_start))
        return average_loss

    def train(self, verbose=1):
        train_start = time()

        num_epochs = self.config.train.epochs
        # dataloader = DataLoader(self.train_set, batch_size=self.config.train.batch_size,
        #                         shuffle=self.config.train.shuffle, num_workers=self.config.train.num_workers,
        #                         pin_memory=True)
        dataloader = self.train_set

        model = self._model
        train_losses = []
        val_losses = []
        loss_fn = torch.nn.CrossEntropyLoss()
        best_loss = self.best_loss
        start_epoch = self.start_epoch
        print('start training...')

        for epoch in range(num_epochs):
            # train
            model.train()
            epoch_start = time()
            batch_losses = []
            batch_cnt = 0
            for labels, batch in dataloader:
                # loader_time = time()
                # try:
                #     print(loader_time - backward_time)
                # except:
                #     print(loader_time - epoch_start)
                batch_cnt += 1
                if self.device.type == "cuda":
                    batch['input_ids'] = batch['input_ids'].to(self.device, non_blocking=True)
                    batch['attention_mask'] = batch['attention_mask'].to(self.device, non_blocking=True)
                    # batch = batch.to(self.device, non_blocking=True)
                    labels = labels.to(self.device, non_blocking=True)
                    # cuda_time = time()

                outputs = model(batch)
                # model_time = time()
                loss = loss_fn(outputs, labels)
                if not loss.requires_grad:
                    raise RuntimeError("loss doesn't require grad")
                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()
                # backward_time = time()
                batch_losses.append(loss.item())
                # print(cuda_time-loader_time, model_time-loader_time, backward_time-loader_time)

                if batch_cnt % self.config.train.log_interval == 0 or (epoch == 0 and batch_cnt <= 10):
                    # if batch_cnt % self.config.train.log_interval == 0:
                    print('Epoch: %d | Step: %d | loss: %.5f | Lr: %.5f' % \
                          (epoch + start_epoch, batch_cnt, batch_losses[-1], self._optimizer.param_groups[0]['lr']))

            average_loss = sum(batch_losses) / len(batch_losses)
            train_losses.append(average_loss)

            if verbose:
                print('Epoch: %d | Train Loss: %.5f | Time: %.5f' % (
                epoch + start_epoch, average_loss, time() - epoch_start))

            # evaluate
            if self.config.train.eval:
                average_eval_loss = self.evaluate('val', verbose=1)
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
                if self.config.train.save:
                    val_list = {
                        'cur_epoch': epoch + start_epoch,
                        'best_loss': best_loss,
                    }
                    self.save(self.config.train.save_path, epoch + start_epoch, val_list)
        self.best_loss = best_loss
        self.start_epoch = start_epoch + num_epochs
        print('optimization finished.')
        print('Total time elapsed: %.5fs' % (time() - train_start))