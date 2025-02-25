# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""

import math
import os
import sys
from typing import Iterable
from datetime import datetime
import logging
import json

import torch.distributed as dist

from util.utils import slprint, to_device

import torch

import util.misc as utils
from datasets.coco_eval import CocoEvaluator
from datasets.panoptic_eval import PanopticEvaluator
from util.data_prefetcher import data_prefetcher



learn_grad_dic = {}
learn_grad_dic_abs = {}
learn_iter = 0

forget_grad_dic = {}
forget_grad_dic_abs = {}
forget_iter = 0







def get_structure_loss(model: torch.nn.Module):
    if isinstance(model, (torch.nn.DataParallel,torch.nn.parallel.DistributedDataParallel)):
        model_without_ddp = model.module

    else:
        model_without_ddp = model
    # if isinstance(model, torch.nn.DataParallel):
    #     model_without_ddp = model.module
    # else:
    #     model_without_ddp = model
    learnable_params_name = [
        name
        for name, param in model_without_ddp.named_parameters()
        if param.requires_grad
    ]
    # print("learnable_params_name",learnable_params_name)
    # learnable_params_name = [
    #     name
    #     for name, param in model_without_ddp.named_parameters()
    # ]
    # print("learnable_params_name",learnable_params_name)
    group_layers = []

    for i in range(6):
        group_item = []
        group_item.append("transformer.encoder.layers.{}.linear1.lora_A".format(i))
        group_item.append("transformer.encoder.layers.{}.linear1.lora_B".format(i))
        group_item.append("transformer.encoder.layers.{}.linear2.lora_A".format(i))
        group_item.append("transformer.encoder.layers.{}.linear2.lora_B".format(i))
        group_layers.append(group_item)
    for i in range(6):
        group_item = []
        group_item.append("transformer.decoder.layers.{}.linear1.lora_A".format(i))
        group_item.append("transformer.decoder.layers.{}.linear1.lora_B".format(i))
        group_item.append("transformer.decoder.layers.{}.linear2.lora_A".format(i))
        group_item.append("transformer.decoder.layers.{}.linear2.lora_B".format(i))
        # 忘写了
        group_layers.append(group_item)
    layers_num=[2,2,18,2]
    for i in range(4):
        for j in range(layers_num[i]):
            group_item = []
            group_item.append(f"backbone.0.layers.{i}.blocks.{j}.mlp.fc1.lora_A")
            group_item.append(f"backbone.0.layers.{i}.blocks.{j}.mlp.fc1.lora_B")
            group_item.append(f"backbone.0.layers.{i}.blocks.{j}.mlp.fc2.lora_A")
            group_item.append(f"backbone.0.layers.{i}.blocks.{j}.mlp.fc2.lora_B")
            group_layers.append(group_item)

    # get the parameters
    params_dict = dict(model_without_ddp.named_parameters())

    group_params = []
    for group_item in group_layers:
        group_param = []
        for item in group_item:
            group_param.append(
                params_dict.get(item)
                if item in learnable_params_name
                else None
            )
        group_params.append(group_param)

    def group_sparse_multi_module(group_param):
        # group_param is a list of parameters
        # calculate the loss for a single group of parameters
        def l2_loss(param_group):
            return torch.sum(param_group**2)

        lasso_sum = 0
        for param in group_param:
            lasso_sum += l2_loss(param)
        return torch.sqrt(lasso_sum)

    group_sparse_loss = 0
    # calculate the loss for all groups of parameters
    for group_param in group_params:
        group_sparse_loss += group_sparse_multi_module(group_param)
    if torch.distributed.is_initialized():
        dist.all_reduce(group_sparse_loss, op=dist.ReduceOp.SUM)
        group_sparse_loss /= torch.distributed.get_world_size()
    # print('group_sparse_loss', group_sparse_loss)
    return group_sparse_loss

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, 
                    wo_class_error=False, lr_scheduler=None, args=None, logger=None, ema_m=None):
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    try:
        need_tgt_for_training = args.use_dn
    except:
        need_tgt_for_training = False

    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    if not wo_class_error:
        metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 50

    _cnt = 0
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header, logger=logger):

        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with torch.cuda.amp.autocast(enabled=args.amp):
            if need_tgt_for_training:
                outputs = model(samples, targets)
            else:
                outputs = model(samples)
        
            loss_dict = criterion(outputs, targets)
            weight_dict = criterion.weight_dict

            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        important_keys = ['loss', 'loss_ce', 'loss_bbox', 'loss_giou']
        loss_dict_reduced_filtered = {k: v for k, v in loss_dict_reduced.items() if
                                      k in important_keys and 'unscaled' not in k}
        # loss_dict_reduced_filtered = {k: v for k, v in loss_dict_reduced.items() if
        #                               not any(k.endswith(f'_{i}') for i in range(10))}
        # loss_dict_reduced_unscaled = {f'{k}_unscaled': v
        #                               for k, v in loss_dict_reduced_filtered.items()}
        loss_dict_reduced_unscaled = {}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced_filtered.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)


        # amp backward function
        if args.amp:
            optimizer.zero_grad()
            scaler.scale(losses).backward()
            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            # original backward function
            optimizer.zero_grad()
            losses.backward()
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()

        if args.onecyclelr:
            lr_scheduler.step()
        if args.use_ema:
            if epoch >= args.ema_epoch:
                ema_m.update(model)

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        if 'class_error' in loss_dict_reduced:
            metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        _cnt += 1
        if args.debug:
            if _cnt % 15 == 0:
                print("BREAK!"*5)
                break

    if getattr(criterion, 'loss_weight_decay', False):
        criterion.loss_weight_decay(epoch=epoch)
    if getattr(criterion, 'tuning_matching', False):
        criterion.tuning_matching(epoch)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    resstat = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
    if getattr(criterion, 'loss_weight_decay', False):
        resstat.update({f'weight_{k}': v for k,v in criterion.weight_dict.items()})
    return resstat


def train_one_epoch_learn(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader_remain: Iterable, data_loader_learn: Iterable,optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0,
                    wo_class_error=False, lr_scheduler=None, args=None, logger=None, ema_m=None,beta=0.15,alpha=0.2):
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    try:
        need_tgt_for_training = args.use_dn
    except:
        need_tgt_for_training = False

    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    if not wo_class_error:
        metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 50

    _cnt = 0

    prefetcher_learn = data_prefetcher(data_loader_learn, device, prefetch=False)
    samples_learn, targets_learn = (
        prefetcher_learn.next()
    )  # data has already been put on GPU device

    for samples_remain, targets_remain in metric_logger.log_every(data_loader_remain, print_freq, header, logger=logger):

        samples_remain = samples_remain.to(device)
        targets_remain = [{k: v.to(device) for k, v in t.items()} for t in targets_remain]

        with torch.cuda.amp.autocast(enabled=args.amp):
            if need_tgt_for_training:
                outputs_remain = model(samples_remain, targets_remain)
            else:
                outputs_remain = model(samples_remain)

            loss_dict_remain = criterion(outputs_remain, targets_remain)
            weight_dict_remain = criterion.weight_dict

            if need_tgt_for_training:
                outputs_learn = model(samples_learn, targets_learn)
            else:
                outputs_learn = model(samples_learn)

            loss_dict_learn = criterion(outputs_learn, targets_learn)
            weight_dict_learn = criterion.weight_dict

            losses_remain = sum(loss_dict_remain[k] * weight_dict_remain[k] for k in loss_dict_remain.keys() if k in weight_dict_remain)
            losses_learn = sum(loss_dict_learn[k] * weight_dict_learn[k] for k in loss_dict_learn.keys() if k in weight_dict_learn)

            beta = 0.9
            alpha = 0.01
            structure_loss = get_structure_loss(model)
            # print("fdsafdasfsafasdfds")
            # print(
            #     f"structure_loss: {structure_loss}, shape: {structure_loss.shape}, requires_grad: {structure_loss.requires_grad}")
            # print(
            #     f"losses_remain.device: {losses_remain.device}, losses_learn.device: {losses_learn.device}, structure_loss.device: {structure_loss.device}")

            # losses = losses_remain + losses_learn*beta + structure_loss * alpha
            # losses = losses_remain + losses_learn*beta*2
            losses = losses_remain + losses_learn * beta + structure_loss * alpha
            # losses = losses_remain + losses_learn *beta
            if _cnt % 50 ==0:
                logger.info(f'iter:{_cnt} loss_remain: {losses_remain.item()} loss_learn: {losses_learn.item()} structure_loss: {structure_loss.item()}')
            loss_dict_remain.update(loss_dict_learn)


        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict_remain)
        important_keys = ['loss', 'loss_ce', 'loss_bbox', 'loss_giou']
        loss_dict_reduced_filtered ={k:v for  k,v in loss_dict_reduced.items() if k in important_keys and 'unscaled' not in k}
        # loss_dict_reduced_filtered = {k: v for k, v in loss_dict_reduced.items() if
        #                               not any(k.endswith(f'_{i}') for i in range(10))}
        # loss_dict_reduced_unscaled = {f'{k}_unscaled': v
        #                               for k, v in loss_dict_reduced_filtered.items()}
        loss_dict_reduced_unscaled = {}
        loss_dict_reduced_scaled = {k: v * weight_dict_remain[k]
                                    for k, v in loss_dict_reduced_filtered.items() if k in weight_dict_remain}


        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        # amp backward function
        if args.amp:
            optimizer.zero_grad()
            scaler.scale(losses).backward()
            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            # original backward function
            with torch.autograd.set_detect_anomaly(True):

                optimizer.zero_grad()
                losses.backward()
                if max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                optimizer.step()

        if args.onecyclelr:
            lr_scheduler.step()


        metric_logger.update(loss=loss_value, **loss_dict_reduced_filtered, **loss_dict_reduced_unscaled)
        if 'class_error' in loss_dict_reduced:
            metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        _cnt += 1
        # 取数据
        samples_learn, targets_learn = prefetcher_learn.next()
        if samples_learn is None:
            prefetcher_learn = data_prefetcher(
                data_loader_learn, device, prefetch=False
            )
            samples_learn, targets_learn = prefetcher_learn.next()
        if args.debug:
            if _cnt % 15 == 0:
                print("BREAK!" * 5)
                break

    if getattr(criterion, 'loss_weight_decay', False):
        criterion.loss_weight_decay(epoch=epoch)
    if getattr(criterion, 'tuning_matching', False):
        criterion.tuning_matching(epoch)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    resstat = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
    if getattr(criterion, 'loss_weight_decay', False):
        resstat.update({f'weight_{k}': v for k, v in criterion.weight_dict.items()})
    return resstat




def train_one_epoch_learn_test(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader_remain: Iterable, data_loader_learn: Iterable,optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0,
                    wo_class_error=False, lr_scheduler=None, args=None, logger=None, ema_m=None,beta=0.15,alpha=0.2):
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    global learn_iter
    global learn_grad_dic
    global learn_grad_dic_abs
    try:
        need_tgt_for_training = args.use_dn
    except:
        need_tgt_for_training = False

    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    if not wo_class_error:
        metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 50

    _cnt = 0

    prefetcher_learn = data_prefetcher(data_loader_learn, device, prefetch=False)
    samples_learn, targets_learn = (
        prefetcher_learn.next()
    )  # data has already been put on GPU device

    for samples_remain, targets_remain in metric_logger.log_every(data_loader_remain, print_freq, header, logger=logger):

        samples_remain = samples_remain.to(device)
        targets_remain = [{k: v.to(device) for k, v in t.items()} for t in targets_remain]

        with torch.cuda.amp.autocast(enabled=args.amp):
            if need_tgt_for_training:
                outputs_remain,_ = model(samples_remain, targets_remain)
            else:
                outputs_remain = model(samples_remain)

            loss_dict_remain = criterion(outputs_remain, targets_remain)
            weight_dict_remain = criterion.weight_dict

            if need_tgt_for_training:
                outputs_learn,structure_loss = model(samples_learn, targets_learn)
            else:
                outputs_learn = model(samples_learn)

            loss_dict_learn = criterion(outputs_learn, targets_learn)
            weight_dict_learn = criterion.weight_dict

            losses_remain = sum(loss_dict_remain[k] * weight_dict_remain[k] for k in loss_dict_remain.keys() if k in weight_dict_remain)
            losses_learn = sum(loss_dict_learn[k] * weight_dict_learn[k] for k in loss_dict_learn.keys() if k in weight_dict_learn)
            beta = 0.9
            alpha = 0.01
            # structure_loss = get_structure_loss(model)
            # print("fdsafdasfsafasdfds")
            # print(
            #     f"structure_loss: {structure_loss}, shape: {structure_loss.shape}, requires_grad: {structure_loss.requires_grad}")
            # print(
            #     f"losses_remain.device: {losses_remain.device}, losses_learn.device: {losses_learn.device}, structure_loss.device: {structure_loss.device}")

            # losses = losses_remain + losses_learn*beta + structure_loss * alpha
            # losses = losses_remain + losses_learn*beta*2
            losses = losses_remain + losses_learn * beta + structure_loss * alpha
            # losses = losses_remain + losses_learn *beta
            if _cnt % 50 ==0:
                logger.info(f'iter:{_cnt} loss_remain: {losses_remain.item()} loss_learn: {losses_learn.item()} structure_loss: {structure_loss.item()}')
            loss_dict_remain.update(loss_dict_learn)


        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict_remain)
        important_keys = ['loss', 'loss_ce', 'loss_bbox', 'loss_giou']
        loss_dict_reduced_filtered ={k:v for  k,v in loss_dict_reduced.items() if k in important_keys and 'unscaled' not in k}
        # loss_dict_reduced_filtered = {k: v for k, v in loss_dict_reduced.items() if
        #                               not any(k.endswith(f'_{i}') for i in range(10))}
        # loss_dict_reduced_unscaled = {f'{k}_unscaled': v
        #                               for k, v in loss_dict_reduced_filtered.items()}
        loss_dict_reduced_unscaled = {}
        loss_dict_reduced_scaled = {k: v * weight_dict_remain[k]
                                    for k, v in loss_dict_reduced_filtered.items() if k in weight_dict_remain}


        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        # amp backward function
        if args.amp:
            optimizer.zero_grad()
            scaler.scale(losses).backward()
            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            # original backward function
            with torch.autograd.set_detect_anomaly(True):

                optimizer.zero_grad()
                losses.backward()

                learn_iter +=1
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        # 先去
                        grad_mean = param.grad.mean().item()
                        learn_grad_dic[name] = grad_mean + learn_grad_dic.get(name, 0.0)
                        grad_mean_abs = param.grad.abs().mean().item()
                        learn_grad_dic_abs[name] = grad_mean_abs + learn_grad_dic_abs.get(name, 0.0)


                if max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                optimizer.step()

        if args.onecyclelr:
            lr_scheduler.step()


        metric_logger.update(loss=loss_value, **loss_dict_reduced_filtered, **loss_dict_reduced_unscaled)
        if 'class_error' in loss_dict_reduced:
            metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        _cnt += 1
        # 取数据
        samples_learn, targets_learn = prefetcher_learn.next()
        if samples_learn is None:
            prefetcher_learn = data_prefetcher(
                data_loader_learn, device, prefetch=False
            )
            samples_learn, targets_learn = prefetcher_learn.next()
        if args.debug:
            if _cnt % 15 == 0:
                print("BREAK!" * 5)
                break

    if (epoch+1) % 5 == 0:
        for key, val in learn_grad_dic.items():
            learn_grad_dic[key] = val / learn_iter
        save_path = os.path.join(args.output_dir, f"learn_grad_{epoch+1}.json")
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(learn_grad_dic, f, ensure_ascii=False, indent=4)

        for key, val in learn_grad_dic_abs.items():
            learn_grad_dic_abs[key] = val / learn_iter
        save_path = os.path.join(args.output_dir, f"learn_grad_{epoch+1}_abs.json")
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(learn_grad_dic_abs, f, ensure_ascii=False, indent=4)

    if getattr(criterion, 'loss_weight_decay', False):
        criterion.loss_weight_decay(epoch=epoch)
    if getattr(criterion, 'tuning_matching', False):
        criterion.tuning_matching(epoch)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    resstat = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
    if getattr(criterion, 'loss_weight_decay', False):
        resstat.update({f'weight_{k}': v for k, v in criterion.weight_dict.items()})
    return resstat





def train_one_epoch_learn_new(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader_remain: Iterable, data_loader_learn: Iterable,optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0,
                    wo_class_error=False, lr_scheduler=None, args=None, logger=None, ema_m=None,beta=0.15,alpha=0.2):
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    try:
        need_tgt_for_training = args.use_dn
    except:
        need_tgt_for_training = False

    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    if not wo_class_error:
        metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 50

    _cnt = 0

    prefetcher_learn = data_prefetcher(data_loader_learn, device, prefetch=False)
    samples_learn, targets_learn = (
        prefetcher_learn.next()
    )  # data has already been put on GPU device

    for samples_remain, targets_remain in metric_logger.log_every(data_loader_remain, print_freq, header, logger=logger):

        samples_remain = samples_remain.to(device)
        targets_remain = [{k: v.to(device) for k, v in t.items()} for t in targets_remain]

        with torch.cuda.amp.autocast(enabled=args.amp):
            if need_tgt_for_training:
                outputs_remain = model(samples_remain, targets_remain)
            else:
                outputs_remain = model(samples_remain)

            loss_dict_remain = criterion(outputs_remain, targets_remain)
            weight_dict_remain = criterion.weight_dict

            if need_tgt_for_training:
                outputs_learn = model(samples_learn, targets_learn)
            else:
                outputs_learn = model(samples_learn)

            loss_dict_learn = criterion(outputs_learn, targets_learn)
            weight_dict_learn = criterion.weight_dict

            losses_remain = sum(loss_dict_remain[k] * weight_dict_remain[k] for k in loss_dict_remain.keys() if k in weight_dict_remain)
            losses_learn = sum(loss_dict_learn[k] * weight_dict_learn[k] for k in loss_dict_learn.keys() if k in weight_dict_learn)

            beta = beta
            alpha= alpha
            structure_loss = get_structure_loss(model)
            beta = min(epoch / 5, 0.5)  # 随epoch逐渐增加到0.5

            # 归一化损失
            losses = (losses_remain + losses_learn * beta) / (1 + beta) + structure_loss * alpha
            # losses = losses_remain + losses_learn*beta + structure_loss * alpha
            loss_dict_remain.update(loss_dict_learn)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict_remain)
        important_keys = ['loss', 'loss_ce', 'loss_bbox', 'loss_giou']
        loss_dict_reduced_filtered ={k:v for  k,v in loss_dict_reduced.items() if k in important_keys and 'unscaled' not in k}
        # loss_dict_reduced_filtered = {k: v for k, v in loss_dict_reduced.items() if
        #                               not any(k.endswith(f'_{i}') for i in range(10))}
        # loss_dict_reduced_unscaled = {f'{k}_unscaled': v
        #                               for k, v in loss_dict_reduced_filtered.items()}
        loss_dict_reduced_unscaled = {}
        loss_dict_reduced_scaled = {k: v * weight_dict_remain[k]
                                    for k, v in loss_dict_reduced_filtered.items() if k in weight_dict_remain}


        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        # amp backward function
        if args.amp:
            optimizer.zero_grad()
            scaler.scale(losses).backward()
            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            # original backward function
            optimizer.zero_grad()
            losses.backward()
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()

        if args.onecyclelr:
            lr_scheduler.step()


        metric_logger.update(loss=loss_value, **loss_dict_reduced_filtered, **loss_dict_reduced_unscaled)
        if 'class_error' in loss_dict_reduced:
            metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        _cnt += 1
        # 取数据
        samples_learn, targets_learn = prefetcher_learn.next()
        if samples_learn is None:
            prefetcher_learn = data_prefetcher(
                data_loader_learn, device, prefetch=False
            )
            samples_learn, targets_learn = prefetcher_learn.next()
        if args.debug:
            if _cnt % 15 == 0:
                print("BREAK!" * 5)
                break

    if getattr(criterion, 'loss_weight_decay', False):
        criterion.loss_weight_decay(epoch=epoch)
    if getattr(criterion, 'tuning_matching', False):
        criterion.tuning_matching(epoch)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    resstat = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
    if getattr(criterion, 'loss_weight_decay', False):
        resstat.update({f'weight_{k}': v for k, v in criterion.weight_dict.items()})
    return resstat

def train_one_epoch_forget(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader_remain: Iterable, data_loader_forget: Iterable,optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0,
                    wo_class_error=False, lr_scheduler=None, args=None, logger=None, ema_m=None,beta=0.15,alpha=0.2,BND=115):
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    global forget_iter
    global forget_grad_dic
    global forget_grad_dic_abs

    try:
        need_tgt_for_training = args.use_dn
    except:
        need_tgt_for_training = False

    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    if not wo_class_error:
        metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 50

    _cnt = 0

    prefetcher_forget = data_prefetcher(data_loader_forget, device, prefetch=False)
    samples_forget, targets_forget = (
        prefetcher_forget.next()
    )  # data has already been put on GPU device

    for samples_remain, targets_remain in metric_logger.log_every(data_loader_remain, print_freq, header, logger=logger):

        samples_remain = samples_remain.to(device)
        targets_remain = [{k: v.to(device) for k, v in t.items()} for t in targets_remain]

        with torch.cuda.amp.autocast(enabled=args.amp):
            if need_tgt_for_training:
                outputs_remain,_= model(samples_remain, targets_remain)
            else:
                outputs_remain,_= model(samples_remain)

            loss_dict_remain = criterion(outputs_remain, targets_remain)
            weight_dict_remain = criterion.weight_dict

            if need_tgt_for_training:
                outputs_forget,structure_loss= model(samples_forget, targets_forget)
            else:
                outputs_forget,structure_loss = model(samples_forget)

            loss_dict_forget = criterion(outputs_forget, targets_forget)
            weight_dict_forget = criterion.weight_dict
            BND = 15
            losses_remain = sum(loss_dict_remain[k] * weight_dict_remain[k] for k in loss_dict_remain.keys() if k in weight_dict_remain)
            losses_forget = sum(loss_dict_forget[k] * weight_dict_forget[k] for k in loss_dict_forget.keys() if k in weight_dict_forget)
            if _cnt % 50 == 0:
                logger.info(f'iter:{_cnt} loss_forget: {losses_forget.item()}')
            losses_forget=torch.functional.F.relu(BND - losses_forget)
            beta = 0.2
            alpha= 0.01

            losses = losses_remain + losses_forget*beta + structure_loss * alpha
            if _cnt % 50 == 0:
                logger.info(
                    f'iter:{_cnt} loss_remain: {losses_remain.item()} loss_forget: {losses_forget.item()} structure_loss: {structure_loss.item()}')
            loss_dict_remain.update(loss_dict_forget)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict_remain)
        important_keys = ['loss', 'loss_ce', 'loss_bbox', 'loss_giou']
        loss_dict_reduced_filtered = {k: v for k, v in loss_dict_reduced.items() if
                                      k in important_keys and 'unscaled' not in k}
        # loss_dict_reduced_unscaled = {f'{k}_unscaled': v
        #                               for k, v in loss_dict_reduced.items()}
        # loss_dict_reduced_scaled = {k: v * weight_dict_remain[k]
        #                             for k, v in loss_dict_reduced.items() if k in weight_dict_remain}
        loss_dict_reduced_unscaled = {}
        loss_dict_reduced_scaled = {k: v * weight_dict_remain[k]
                                    for k, v in loss_dict_reduced_filtered.items() if k in weight_dict_remain}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        # amp backward function 反向传播
        if args.amp:
            optimizer.zero_grad()
            scaler.scale(losses).backward()



            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            # original backward function
            optimizer.zero_grad()
            losses.backward()

            forget_iter += 1
            for name, param in model.named_parameters():
                if param.grad is not None:
                    # 先去
                    grad_mean = param.grad.mean().item()
                    forget_grad_dic[name] = grad_mean + forget_grad_dic.get(name, 0.0)
                    grad_mean_abs = param.grad.abs().mean().item()
                    forget_grad_dic_abs[name] = grad_mean_abs + forget_grad_dic_abs.get(name, 0.0)
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()

        if args.onecyclelr:
            lr_scheduler.step()
        if args.use_ema:
            if epoch >= args.ema_epoch:
                ema_m.update(model)

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        if 'class_error' in loss_dict_reduced:
            metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        _cnt += 1

        samples_forget, targets_forget = prefetcher_forget.next()
        if samples_forget is None:
            prefetcher_forget = data_prefetcher(
                data_loader_forget, device, prefetch=False
            )
            samples_forget, targets_forget = prefetcher_forget.next()
        if args.debug:
            if _cnt % 15 == 0:
                print("BREAK!" * 5)
                break

    if (epoch + 1) % 5 == 0:
        for key, val in forget_grad_dic.items():
            forget_grad_dic[key] = val / forget_iter
        save_path = os.path.join(args.output_dir, f"forget_grad_{epoch + 1}.json")
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(forget_grad_dic, f, ensure_ascii=False, indent=4)

        for key, val in forget_grad_dic_abs.items():
            forget_grad_dic_abs[key] = val / forget_iter
        save_path = os.path.join(args.output_dir, f"forget_grad_{epoch + 1}_abs.json")
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(forget_grad_dic_abs, f, ensure_ascii=False, indent=4)

    if getattr(criterion, 'loss_weight_decay', False):
        criterion.loss_weight_decay(epoch=epoch)
    if getattr(criterion, 'tuning_matching', False):
        criterion.tuning_matching(epoch)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    resstat = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
    if getattr(criterion, 'loss_weight_decay', False):
        resstat.update({f'weight_{k}': v for k, v in criterion.weight_dict.items()})
    return resstat

def train_one_epoch_forget_twice(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader_remain: Iterable, data_loader_forget: Iterable,optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0,
                    wo_class_error=False, lr_scheduler=None, args=None, logger=None, ema_m=None,beta=0.15,alpha=0.2,BND=115):
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    global forget_twice_iter
    global forget_twice_grad_dic
    global forget_twice_grad_dic_abs

    try:
        need_tgt_for_training = args.use_dn
    except:
        need_tgt_for_training = False

    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    if not wo_class_error:
        metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 50

    _cnt = 0

    prefetcher_forget = data_prefetcher(data_loader_forget, device, prefetch=False)
    samples_forget, targets_forget = (
        prefetcher_forget.next()
    )  # data has already been put on GPU device

    for samples_remain, targets_remain in metric_logger.log_every(data_loader_remain, print_freq, header, logger=logger):

        samples_remain = samples_remain.to(device)
        targets_remain = [{k: v.to(device) for k, v in t.items()} for t in targets_remain]

        with torch.cuda.amp.autocast(enabled=args.amp):
            if need_tgt_for_training:
                outputs_remain,_= model(samples_remain, targets_remain)
            else:
                outputs_remain,_= model(samples_remain)

            loss_dict_remain = criterion(outputs_remain, targets_remain)
            weight_dict_remain = criterion.weight_dict

            if need_tgt_for_training:
                outputs_forget,structure_loss= model(samples_forget, targets_forget)
            else:
                outputs_forget,structure_loss = model(samples_forget)

            loss_dict_forget = criterion(outputs_forget, targets_forget)
            weight_dict_forget = criterion.weight_dict
            BND = 15
            losses_remain = sum(loss_dict_remain[k] * weight_dict_remain[k] for k in loss_dict_remain.keys() if k in weight_dict_remain)
            losses_forget = sum(loss_dict_forget[k] * weight_dict_forget[k] for k in loss_dict_forget.keys() if k in weight_dict_forget)
            if _cnt % 50 == 0:
                logger.info(f'iter:{_cnt} loss_forget: {losses_forget.item()}')
            losses_forget=torch.functional.F.relu(BND - losses_forget)
            beta = 0.2
            alpha= 0.01

            losses = losses_remain + losses_forget*beta + structure_loss * alpha
            if _cnt % 50 == 0:
                logger.info(
                    f'iter:{_cnt} loss_remain: {losses_remain.item()} loss_forget: {losses_forget.item()} structure_loss: {structure_loss.item()}')
            loss_dict_remain.update(loss_dict_forget)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict_remain)
        important_keys = ['loss', 'loss_ce', 'loss_bbox', 'loss_giou']
        loss_dict_reduced_filtered = {k: v for k, v in loss_dict_reduced.items() if
                                      k in important_keys and 'unscaled' not in k}
        # loss_dict_reduced_unscaled = {f'{k}_unscaled': v
        #                               for k, v in loss_dict_reduced.items()}
        # loss_dict_reduced_scaled = {k: v * weight_dict_remain[k]
        #                             for k, v in loss_dict_reduced.items() if k in weight_dict_remain}
        loss_dict_reduced_unscaled = {}
        loss_dict_reduced_scaled = {k: v * weight_dict_remain[k]
                                    for k, v in loss_dict_reduced_filtered.items() if k in weight_dict_remain}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        # amp backward function 反向传播
        if args.amp:
            optimizer.zero_grad()
            scaler.scale(losses).backward()



            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            # original backward function
            optimizer.zero_grad()
            losses.backward()
            #
            # forget_iter += 1
            # for name, param in model.named_parameters():
            #     if param.grad is not None:
            #         # 先去
            #         grad_mean = param.grad.mean().item()
            #         forget_grad_dic[name] = grad_mean + forget_grad_dic.get(name, 0.0)
            #         grad_mean_abs = param.grad.abs().mean().item()
            #         forget_grad_dic_abs[name] = grad_mean_abs + forget_grad_dic_abs.get(name, 0.0)
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()

        if args.onecyclelr:
            lr_scheduler.step()
        if args.use_ema:
            if epoch >= args.ema_epoch:
                ema_m.update(model)

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        if 'class_error' in loss_dict_reduced:
            metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        _cnt += 1

        samples_forget, targets_forget = prefetcher_forget.next()
        if samples_forget is None:
            prefetcher_forget = data_prefetcher(
                data_loader_forget, device, prefetch=False
            )
            samples_forget, targets_forget = prefetcher_forget.next()
        if args.debug:
            if _cnt % 15 == 0:
                print("BREAK!" * 5)
                break

    # if (epoch + 1) % 5 == 0:
    #     for key, val in forget_grad_dic.items():
    #         forget_grad_dic[key] = val / forget_iter
    #     save_path = os.path.join(args.output_dir, f"forget_grad_{epoch + 1}.json")
    #     with open(save_path, "w", encoding="utf-8") as f:
    #         json.dump(forget_grad_dic, f, ensure_ascii=False, indent=4)
    #
    #     for key, val in forget_grad_dic_abs.items():
    #         forget_grad_dic_abs[key] = val / forget_iter
    #     save_path = os.path.join(args.output_dir, f"forget_grad_{epoch + 1}_abs.json")
    #     with open(save_path, "w", encoding="utf-8") as f:
    #         json.dump(forget_grad_dic_abs, f, ensure_ascii=False, indent=4)

    if getattr(criterion, 'loss_weight_decay', False):
        criterion.loss_weight_decay(epoch=epoch)
    if getattr(criterion, 'tuning_matching', False):
        criterion.tuning_matching(epoch)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    resstat = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
    if getattr(criterion, 'loss_weight_decay', False):
        resstat.update({f'weight_{k}': v for k, v in criterion.weight_dict.items()})
    return resstat


class COCOEvalLogger:
    def __init__(self, log_dir, exp_name=None):
        """
        Initialize the COCO evaluation logger

        Args:
            log_dir (str): Directory to save log files
            exp_name (str, optional): Experiment name for the log file prefix
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        # Create experiment name with timestamp if not provided
        if exp_name is None:
            exp_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.exp_name = exp_name

        # Set up logging
        self.logger = logging.getLogger(f'coco_eval_{exp_name}')
        self.logger.setLevel(logging.INFO)

        # File handler for detailed logs
        log_file = os.path.join(log_dir, f'coco_eval_{exp_name}.log')
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)

        # Format for logging
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        # Also keep metrics in a dictionary for JSON export
        self.metrics_history = []

    def log_coco_eval_metrics(self, coco_eval, epoch=None, iteration=None):
        """
        Log COCO evaluation metrics

        Args:
            coco_eval: COCO evaluator object
            epoch (int, optional): Current epoch number
            iteration (int, optional): Current iteration number
        """
        metrics = {
            'epoch': epoch,
            'iteration': iteration,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        # Standard COCO metric names
        metric_names = [
            'AP @ IoU=0.50:0.95 | area=all | maxDets=100',
            'AP @ IoU=0.50 | area=all | maxDets=100',
            'AP @ IoU=0.75 | area=all | maxDets=100',
            'AP @ IoU=0.50:0.95 | area=small | maxDets=100',
            'AP @ IoU=0.50:0.95 | area=medium | maxDets=100',
            'AP @ IoU=0.50:0.95 | area=large | maxDets=100',
            'AR @ IoU=0.50:0.95 | area=all | maxDets=1',
            'AR @ IoU=0.50:0.95 | area=all | maxDets=10',
            'AR @ IoU=0.50:0.95 | area=all | maxDets=100',
            'AR @ IoU=0.50:0.95 | area=small | maxDets=100',
            'AR @ IoU=0.50:0.95 | area=medium | maxDets=100',
            'AR @ IoU=0.50:0.95 | area=large | maxDets=100'
        ]

        # Log each metric
        self.logger.info(f"{'=' * 20} COCO Evaluation Metrics {'=' * 20}")
        if epoch is not None:
            self.logger.info(f"Epoch: {epoch}")
        if iteration is not None:
            self.logger.info(f"Iteration: {iteration}")

        stats = coco_eval.stats.tolist()
        for name, value in zip(metric_names, stats):
            self.logger.info(f"{name}: {value:.3f}")
            metrics[name.replace(' | ', '_').replace('=', '').replace(' ', '_')] = value

        self.metrics_history.append(metrics)
        self.logger.info("=" * 60)

        # Save metrics to JSON
        self._save_metrics_to_json()

    def _save_metrics_to_json(self):
        """Save all metrics history to a JSON file"""
        json_file = os.path.join(self.log_dir, f'coco_eval_metrics_{self.exp_name}.json')
        with open(json_file, 'w') as f:
            json.dump(self.metrics_history, f, indent=4)

    def get_latest_metrics(self):
        """Return the most recent metrics"""
        return self.metrics_history[-1] if self.metrics_history else None
@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir, wo_class_error=False, args=None, logger=None):
    try:
        need_tgt_for_training = args.use_dn
    except:
        need_tgt_for_training = False

    model.eval()
    criterion.eval()

    # coco_logger = COCOEvalLogger(
    #     log_dir=os.path.join(output_dir, 'eval_logs'),
    #     exp_name=args.exp_name if hasattr(args, 'exp_name') else None
    # )

    metric_logger = utils.MetricLogger(delimiter="  ")
    if not wo_class_error:
        metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    useCats = True
    try:
        useCats = args.useCats
    except:
        useCats = True
    if not useCats:
        print("useCats: {} !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!".format(useCats))
    coco_evaluator = CocoEvaluator(base_ds, iou_types, useCats=useCats)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )

    _cnt = 0
    output_state_dict = {} # for debug only
    print_feq=50
    for samples, targets in metric_logger.log_every(data_loader, 100, header, logger=logger):
        samples = samples.to(device)

        # targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        targets = [{k: to_device(v, device) for k, v in t.items()} for t in targets]

        with torch.cuda.amp.autocast(enabled=args.amp):
            if need_tgt_for_training:
                outputs,_ = model(samples, targets)
            else:
                outputs,_= model(samples)
            # outputs = model(samples)

            loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        important_keys = ['loss', 'loss_ce', 'loss_bbox', 'loss_giou']
        loss_dict_reduced_filtered = {k: v for k, v in loss_dict_reduced.items() if
                                      k in important_keys and 'unscaled' not in k}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced_filtered.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        if 'class_error' in loss_dict_reduced:
            metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        # [scores: [100], labels: [100], boxes: [100, 4]] x B
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}

        if coco_evaluator is not None:
            coco_evaluator.update(res)

        if panoptic_evaluator is not None:
            res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
            for i, target in enumerate(targets):
                image_id = target["image_id"].item()
                file_name = f"{image_id:012d}.png"
                res_pano[i]["image_id"] = image_id
                res_pano[i]["file_name"] = file_name

            panoptic_evaluator.update(res_pano)
        
        if args.save_results:
            # res_score = outputs['res_score']
            # res_label = outputs['res_label']
            # res_bbox = outputs['res_bbox']
            # res_idx = outputs['res_idx']


            for i, (tgt, res, outbbox) in enumerate(zip(targets, results, outputs['pred_boxes'])):
                """
                pred vars:
                    K: number of bbox pred
                    score: Tensor(K),
                    label: list(len: K),
                    bbox: Tensor(K, 4)
                    idx: list(len: K)
                tgt: dict.

                """
                # compare gt and res (after postprocess)
                gt_bbox = tgt['boxes']
                gt_label = tgt['labels']
                gt_info = torch.cat((gt_bbox, gt_label.unsqueeze(-1)), 1)
                
                # img_h, img_w = tgt['orig_size'].unbind()
                # scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=0)
                # _res_bbox = res['boxes'] / scale_fct
                _res_bbox = outbbox
                _res_prob = res['scores']
                _res_label = res['labels']
                res_info = torch.cat((_res_bbox, _res_prob.unsqueeze(-1), _res_label.unsqueeze(-1)), 1)
                # import ipdb;ipdb.set_trace()

                if 'gt_info' not in output_state_dict:
                    output_state_dict['gt_info'] = []
                output_state_dict['gt_info'].append(gt_info.cpu())

                if 'res_info' not in output_state_dict:
                    output_state_dict['res_info'] = []
                output_state_dict['res_info'].append(res_info.cpu())

            # # for debug only
            # import random
            # if random.random() > 0.7:
            #     print("Now let's break")
            #     break

        _cnt += 1
        if args.debug:
            if _cnt % 15 == 0:
                print("BREAK!"*5)
                break

    if args.save_results:
        import os.path as osp
        
        # output_state_dict['gt_info'] = torch.cat(output_state_dict['gt_info'])
        # output_state_dict['res_info'] = torch.cat(output_state_dict['res_info'])
        savepath = osp.join(args.output_dir, 'results-{}.pkl'.format(utils.get_rank()))
        print("Saving res to {}".format(savepath))
        torch.save(output_state_dict, savepath)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    category_ap = {}
    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
        # 计算并记录按类别的 AP
        if 'bbox' in coco_evaluator.coco_eval:

            coco_eval_bbox = coco_evaluator.coco_eval['bbox']
            # 每个类别的 AP 存储在 coco_eval_bbox.eval['precision']
            # precision shape: [IoU阈值, Recall, 类别索引, area范围, maxDets范围]
            precision = coco_eval_bbox.eval['precision']
            categories = base_ds.cats  # COCO 数据集的类别信息
            category_mapping = {cat_id: i for i, cat_id in enumerate(sorted(base_ds.cats.keys()))}

            for cat_id, cat_info in categories.items():
                # 按类别索引提取 AP，IoU 范围是 0.5:0.95 平均
                category_index = category_mapping[cat_id]  # 使用映射后的连续索引  # COCO 的类别 ID 是从 1 开始的
                category_precision = precision[:, :, category_index, 0, -1]  # area=all, maxDets=max
                category_ap[cat_info['name']] = category_precision.mean()  # 计算类别的平均 AP

            # 将按类别 AP 结果记录到 stats 字典中

    panoptic_res = None
    if panoptic_evaluator is not None:
        panoptic_res = panoptic_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()



    if panoptic_res is not None:
        stats['PQ_all'] = panoptic_res["All"]
        stats['PQ_th'] = panoptic_res["Things"]
        stats['PQ_st'] = panoptic_res["Stuff"]

    stats['category_ap'] = category_ap

    return stats, coco_evaluator


@torch.no_grad()
def test(model, criterion, postprocessors, data_loader, base_ds, device, output_dir, wo_class_error=False, args=None, logger=None):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    # if not wo_class_error:
    #     metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    # coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )

    final_res = []
    for samples, targets in metric_logger.log_every(data_loader, 10, header, logger=logger):
        samples = samples.to(device)

        # targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        targets = [{k: to_device(v, device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        # loss_dict = criterion(outputs, targets)
        # weight_dict = criterion.weight_dict

        # # reduce losses over all GPUs for logging purposes
        # loss_dict_reduced = utils.reduce_dict(loss_dict)
        # loss_dict_reduced_scaled = {k: v * weight_dict[k]
        #                             for k, v in loss_dict_reduced.items() if k in weight_dict}
        # loss_dict_reduced_unscaled = {f'{k}_unscaled': v
        #                               for k, v in loss_dict_reduced.items()}
        # metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
        #                      **loss_dict_reduced_scaled,
        #                      **loss_dict_reduced_unscaled)
        # if 'class_error' in loss_dict_reduced:
        #     metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes, not_to_xyxy=True)
        # [scores: [100], labels: [100], boxes: [100, 4]] x B
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        for image_id, outputs in res.items():
            _scores = outputs['scores'].tolist()
            _labels = outputs['labels'].tolist()
            _boxes = outputs['boxes'].tolist()
            for s, l, b in zip(_scores, _labels, _boxes):
                assert isinstance(l, int)
                itemdict = {
                        "image_id": int(image_id), 
                        "category_id": l, 
                        "bbox": b, 
                        "score": s,
                        }
                final_res.append(itemdict)

    if args.output_dir:
        import json
        with open(args.output_dir + f'/results{args.rank}.json', 'w') as f:
            json.dump(final_res, f)        

    return final_res
