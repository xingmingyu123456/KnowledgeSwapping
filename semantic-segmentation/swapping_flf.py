# ------------------------------------------------------------------------
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# by Feng Li and Hao Zhang.
# ------------------------------------------------------------------------
"""
MaskDINO Training Script based on Mask2Former.
"""
import json
import warnings

from detectron2.evaluation.cityscapes_evaluation import CityscapesEvaluator
from torch.optim.lr_scheduler import StepLR

from testmapper import testMapper

# 忽略所有警告ss
warnings.filterwarnings("ignore")

import copy
import itertools
import logging
import os

from collections import OrderedDict
from typing import Any, Dict, List, Set

import torch

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, build_detection_train_loader,build_detection_test_loader

from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    SemSegEvaluator,
    verify_results, print_csv_format,
)
from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.utils.logger import setup_logger
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
# MaskDINO
from mask2former import (
    COCOInstanceNewBaselineDatasetMapper,
    COCOPanopticNewBaselineDatasetMapper,
    InstanceSegEvaluator,
    MaskFormerInstanceDatasetMapper,
    MaskFormerPanopticDatasetMapper,
    MaskFormerSemanticDatasetMapper,
    SemanticSegmentorWithTTA,
    add_maskformer2_config,
    CustomMaskFormerSemanticDatasetMapper,
)
import random
from detectron2.engine import (
    DefaultTrainer,
    default_argument_parser,
    default_setup,
    hooks,
    launch,
    create_ddp_model,
    AMPTrainer,
    SimpleTrainer
)
import weakref
from datasets.register_pet import register_pet
from datasets.register_coco import register_coco
from datasets.register_sat_new import  register_sat_new
from datasets.register_sat import register_sat
from datasets.register_forget import register_forget
from datasets.register_remain import register_remain
from datasets.register_forget_ade import register_forget_ade
from datasets.register_remain_ade import register_remain_ade

import numpy as np
import torch.nn as nn
import math
from datetime import datetime
import csv
import loralib as lora

current = datetime.now()
time_str = current.strftime("%Y-%m-%d-%H-%M-%S")
# time_str = "2024-11-04-10-56-55"
def reinitialize_lora_parameters(model):

    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora" in name:
                if isinstance(param, nn.Parameter):
                    if "lora_A" in name:
                        nn.init.kaiming_uniform_(param, a=math.sqrt(50))
                    elif "lora_B" in name:
                        nn.init.zeros_(param)
                else:
                    raise ValueError(
                        f"Parameter {name} is not an instance of nn.Parameter."
                    )

def get_structure_loss(model: torch.nn.Module):
    # DistributedDataParallel 是这个类型
    if isinstance(model, (torch.nn.DataParallel,torch.nn.parallel.DistributedDataParallel)):
        model_without_ddp = model.module

    else:
        model_without_ddp = model
    learnable_params_name = [
        name
        for name, param in model_without_ddp.named_parameters()
        if param.requires_grad
    ]
    # print("learable_params_name",learnable_params_name)
    # learnable_params_name = [
    #     name
    #     for name, param in model_without_ddp.named_parameters()
    # ]
    # print("learnable_params_name",learnable_params_name)
    group_layers = []

    layers = [0,1,3]

    for i in layers:
        for j in range(2):
            group_item = []
            group_item.append(f"backbone.layers.{i}.blocks.{j}.mlp.fc1.lora_A")
            group_item.append(f"backbone.layers.{i}.blocks.{j}.mlp.fc1.lora_B")
            group_item.append(f"backbone.layers.{i}.blocks.{j}.mlp.fc2.lora_A")
            group_item.append(f"backbone.layers.{i}.blocks.{j}.mlp.fc2.lora_B")
            group_layers.append(group_item)

    for j in range(18): #原来这里可能写错了用的是2（用的是cityscape的时候）
        group_item = []
        group_item.append(f"backbone.layers.2.blocks.{j}.mlp.fc1.lora_A")
        group_item.append(f"backbone.layers.2.blocks.{j}.mlp.fc1.lora_B")
        group_item.append(f"backbone.layers.2.blocks.{j}.mlp.fc2.lora_A")
        group_item.append(f"backbone.layers.2.blocks.{j}.mlp.fc2.lora_B")
        group_layers.append(group_item)

    for i in range(6):
        group_item = []
        group_item.append(f"sem_seg_head.pixel_decoder.transformer.encoder.layers.{i}.linear1.lora_A")
        group_item.append(f"sem_seg_head.pixel_decoder.transformer.encoder.layers.{i}.linear1.lora_B")
        group_item.append(f"sem_seg_head.pixel_decoder.transformer.encoder.layers.{i}.linear2.lora_A")
        group_item.append(f"sem_seg_head.pixel_decoder.transformer.encoder.layers.{i}.linear2.lora_B")
        group_layers.append(group_item)

    for i in range(9):
        group_item = []
        group_item.append(f"sem_seg_head.predictor.transformer_ffn_layers.{i}.linear1.lora_A")
        group_item.append(f"sem_seg_head.predictor.transformer_ffn_layers.{i}.linear1.lora_B")
        group_item.append(f"sem_seg_head.predictor.transformer_ffn_layers.{i}.linear2.lora_A")
        group_item.append(f"sem_seg_head.predictor.transformer_ffn_layers.{i}.linear2.lora_B")
        group_layers.append(group_item)


    # get the parameters
    group_params = []
    for group_item in group_layers:
        group_param = []
        for item in group_item:
            group_param.append(
                model_without_ddp.get_parameter(item)
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
    # print('group_sparse_loss', group_sparse_loss)
    return group_sparse_loss



class Trainer_learn(DefaultTrainer):
    """
    Extension of the Trainer class adapted to MaskFormer.
    """
    def __init__(self, cfg):
        super().__init__(cfg)
        self.logger = logging.getLogger("detectron2")
        if not self.logger.isEnabledFor(logging.INFO):  # setup_logger is not called for d2
            self.logger =setup_logger()
        # logger = setup_logger()  # 不传 level 参数
        # logger.setLevel(logging.INFO)  # 手动设置日志级别为 WARNING

        cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())

        # Assume these objects must be constructed in this order.
        self.model = self.build_model(cfg)
        self.optimizer = self.build_optimizer(cfg, self.model)



        self.data_loader_learn,self.data_loader_remain = self.build_train_loader(cfg)
        # dataloader
        self.data_loader_learn_iter = iter(self.data_loader_learn)
        self.data_loader_remain_iter = iter(self.data_loader_remain)

        self.model = create_ddp_model(self.model, broadcast_buffers=False,find_unused_parameters=True)

        self.scheduler = self.build_lr_scheduler(cfg, self.optimizer)
        self._trainer = SimpleTrainer(
            self.model, self.data_loader_learn, self.optimizer
        )
        # add model EMA
        kwargs = {
            'trainer': weakref.proxy(self),
        }
        # kwargs.update(model_ema.may_get_ema_checkpointer(cfg, model)) TODO: release ema training for large models
        self.checkpointer = DetectionCheckpointer(
            # Assume you want to save checkpoints together with logs/statistics
            self.model,
            cfg.OUTPUT_DIR,
            **kwargs,
        )
        self.start_iter = 0

        self.cfg = cfg


        # # 保存权重狗子 输出日志狗子 测试钩子
        # self.register_hooks(self.build_hooks())
        # TODO: release model conversion checkpointer from DINO to MaskDINO
        self.checkpointer = DetectionCheckpointer(
            # Assume you want to save checkpoints together with logs/statistics
            self.model,
            cfg.OUTPUT_DIR,
            **kwargs,
        )
        # TODO: release GPU cluster submit scripts based on submitit for multi-node training
        self.current_iter = 0

        self.grad_dic = {}
        self.grad_dic_abs = {}

    def save_checkpoint(self, name):
        """
        Save checkpoint with validation
        """
        logger = logging.getLogger("detectron2.trainer")

        # 检查当前模型状态
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if 'norm' in name:
                    if 'weight' in name and torch.allclose(param, torch.ones_like(param)):
                        logger.error(f"Error: {name} weights are all ones before saving!")
                    if 'bias' in name and torch.allclose(param, torch.zeros_like(param)):
                        logger.error(f"Error: {name} biases are all zeros before saving!")
                if 'lora_B' in name and torch.allclose(param, torch.zeros_like(param)):
                    logger.error(f"Error: {name} is all zeros before saving!")

                # 打印参数统计信息
                logger.info(f"Parameter {name} stats before saving: "
                            f"mean={param.mean().item():.4f}, "
                            f"std={param.std().item():.4f}, "
                            f"min={param.min().item():.4f}, "
                            f"max={param.max().item():.4f}")

        # 保存检查点
        save_dict = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler else None,
            "iteration": self.iter,
        }

        save_path = os.path.join(self.cfg.OUTPUT_DIR, f"{name}.pth")
        torch.save(save_dict, save_path)

        # 验证保存的检查点
        loaded = torch.load(save_path)
        if "model" in loaded:
            model_state = loaded["model"]
            for key, value in model_state.items():
                if 'norm' in key:
                    if 'weight' in key and torch.allclose(value, torch.ones_like(value)):
                        logger.error(f"Error: Saved {key} weights are all ones!")
                    if 'bias' in key and torch.allclose(value, torch.zeros_like(value)):
                        logger.error(f"Error: Saved {key} biases are all zeros!")
                if 'lora_B' in key and torch.allclose(value, torch.zeros_like(value)):
                    logger.error(f"Error: Saved {key} is all zeros!")

    def train(self):
        # 使用自定义 run_step 的训练循环
        super().train()  # 调用父类的 train 方法，会使用自定义的 run_step



    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        为不同数据集创建对应的评估器
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")

        return SemSegEvaluator(
            dataset_name,
            output_dir=output_folder,
        )

    @classmethod
    def build_train_loader(cls, cfg):
        """
        Create two data loaders, each for a different dataset.
        """
        # Configure data loader for dataset1
        cfg1 = cfg.clone()
        cfg1.defrost()
        cfg1.DATASETS.TRAIN = ("learn_dataset_train",)  # 修改为第一个数据集的名称

        # cfg1.MODEL.PIXEL_MEAN = [117.75762846,113.53144235,98.78248971]
        # cfg1.MODEL.PIXEL_STD = [58.52638834,58.14435355,59.14101012]
        mapper1 = CustomMaskFormerSemanticDatasetMapper.from_config(cfg1, is_train=True)
        data_loader_learn = build_detection_train_loader(cfg1, mapper=mapper1)
        cfg2 = cfg.clone()
        cfg2.defrost()
        cfg2.DATASETS.TRAIN = ("remain_dataset_train",)  # 修改为第一个数据集的名称

        mapper2 = CustomMaskFormerSemanticDatasetMapper.from_config(cfg2, is_train=True)
        data_loader_remain = build_detection_train_loader(cfg2, mapper=mapper2)
        return data_loader_learn, data_loader_remain

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        if dataset_name == "remain_dataset_val":
            pixel_mean = torch.tensor([123.675, 116.280, 103.530], dtype=torch.float32).view(3, 1, 1)
            pixel_std = torch.tensor([58.395, 57.120, 57.375], dtype=torch.float32).view(3, 1, 1)

        elif dataset_name == "learn_dataset_val":
            pixel_mean = torch.tensor([119.73433564, 109.83269262, 94.84763627], dtype=torch.float32).view(3, 1, 1)
            pixel_std = torch.tensor([57.6707407, 57.19914482, 58.46454528], dtype=torch.float32).view(3, 1, 1)
        elif dataset_name == "forget_dataset_val":
            pixel_mean = torch.tensor([123.675, 116.280, 103.530], dtype=torch.float32).view(3, 1, 1)
            pixel_std = torch.tensor([58.395, 57.120, 57.375], dtype=torch.float32).view(3, 1, 1)
        else:
            raise ValueError(f"Unknown dataset name: {dataset_name}")
        # print(f"数据集：{dataset_name},pixel_mean：{pixel_mean},pixel_std：{pixel_std}")
        # print("fdsaafdasfdasfdas111111111111")
        # print()
        mapper = testMapper(mean=pixel_mean, std=pixel_std)
        test_dataset_loader = build_detection_test_loader(cfg, dataset_name,mapper=mapper)
        return  test_dataset_loader


    def run_step(self):
        """
        Modified training step to load data from two loaders and calculate losses separately.
        """
        assert self.model.training, "[Trainer] Model is not in training mode."

        # Get data from both data loaders
        data_learn = next(self.data_loader_learn_iter, None)
        data_remain = next(self.data_loader_remain_iter, None)

        # Reset iterator if any data loader is exhausted
        if data_learn is None:
            self.data_loader_learn_iter = iter(self.data_loader_learn)
            data_learn = next(self.data_loader_learn_iter)

        if data_remain is None:
            self.data_loader_remain_iter = iter(self.data_loader_remain)
            data_remain = next(self.data_loader_remain_iter)

        # Compute losses separately
        loss_dict_learn = self.model(data_learn)
        loss_dict_remain = self.model(data_remain)
        structure_loss=get_structure_loss(self.model)
        # Calculate total loss
        losses_learn = sum(loss for loss in loss_dict_learn.values())
        losses_remain = sum(loss for loss in loss_dict_remain.values())
        if self.iter%20==0:
            self.logger.info(f"当前迭代：{self.iter} losses_learn: {losses_learn}, losses_remain: {losses_remain}, structure_loss: {structure_loss}")

        beta = self.cfg.LEARN_BETA
        alpha =self.cfg.LEARN_ALPHA
        # total_loss = beta*losses_learn + losses_remain + alpha*structure_loss
        total_loss = beta * losses_learn + losses_remain + alpha*structure_loss

        # Backward and optimizer step
        self.optimizer.zero_grad()
        total_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        for name, param in self.model.named_parameters():
            if param.grad is not None:
                # 先去
                grad_mean = param.grad.mean().item()
                self.grad_dic[name] = grad_mean + self.grad_dic.get(name, 0.0)
                grad_mean_abs = param.grad.abs().mean().item()
                self.grad_dic_abs[name] = grad_mean_abs + self.grad_dic_abs.get(name, 0.0)

        self.optimizer.step()
        # Update learning rate only if within iteration limits
        # if self.current_iter < self.max_iter:
        #     if self.scheduler is not None:
        #         self.scheduler.step()



        if self.iter % 2000 == 1000 and self.iter != 0:
            self.custom_save(f"learn_model_{self.current_iter:07d}")
        if self.iter % 1000 == 0 and self.iter != 0:
            for key,val in self.grad_dic.items():
                self.grad_dic[key] = val/self.iter
            save_path = os.path.join(self.cfg.OUTPUT_DIR, f"learn_grad_{self.iter}.json")
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(self.grad_dic, f, ensure_ascii=False,indent=4)

            for key,val in self.grad_dic_abs.items():
                self.grad_dic_abs[key] = val/self.iter
            save_path = os.path.join(self.cfg.OUTPUT_DIR, f"learn_grad_{self.iter}_abs.json")
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(self.grad_dic_abs, f, ensure_ascii=False,indent=4)

        # Update iteration counter
        self.current_iter += 1

    def custom_save(self, name):
        """
        自定义保存函数，保存完整的模型状态
        """
        logger = logging.getLogger("detectron2.trainer")
        logger.info(f"Saving model checkpoint: {name}")
        if isinstance(self.model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
            model_without_ddp = self.model.module
        else:
            model_without_ddp = self.model
        save_dict = {
            "model": model_without_ddp.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler else None,
            "iteration": self.iter,
        }

        save_path = os.path.join(self.cfg.OUTPUT_DIR, f"{name}.pth")
        torch.save(save_dict, save_path)

        # 验证保存的检查点
        loaded = torch.load(save_path)
        if "model" in loaded:
            logger.info(f"Successfully saved checkpoint: {save_path}")
        else:
            logger.error(f"Failed to save checkpoint: {save_path}")

    def custom_load(self, path,resume=False):
        """
        自定义加载函数
        """
        logger = logging.getLogger("detectron2.trainer")
        logger.info(f"Loading checkpoint from: {path}")

        checkpoint = torch.load(path, map_location=torch.device("cpu"))

        # 加载模型权重
        # self.model.load_state_dict(checkpoint["model"],strict=False)
        if isinstance(self.model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
            missing_keys, unexpected_keys = self.model.module.load_state_dict(checkpoint["model"], strict=False)
        else:
            missing_keys, unexpected_keys = self.model.load_state_dict(checkpoint["model"], strict=False)


        if len(missing_keys) > 0:
            logger.warning(f"Missing keys in checkpoint: {missing_keys}")
        if len(unexpected_keys) > 0:
            logger.warning(f"Unexpected keys in checkpoint: {unexpected_keys}")
        if resume:
            # 加载优化器状态
            if "optimizer" in checkpoint and self.optimizer:
                logger.info("Loading optimizer from checkpoint")
                self.optimizer.load_state_dict(checkpoint["optimizer"])

            # 加载调度器状态
            if "scheduler" in checkpoint and self.scheduler:
                logger.info("Loading scheduler from checkpoint")
                self.scheduler.load_state_dict(checkpoint["scheduler"])

            # 恢复迭代次数
            if "iteration" in checkpoint:
                self.iter = checkpoint["iteration"]

        return checkpoint
    def resume_or_load(self, resume=False):
        """
        重写resume_or_load方法以使用自定义的加载函数
        """
        if resume:
            if self.cfg.MODEL.WEIGHTS:
                check = self.custom_load(self.cfg.MODEL.WEIGHTS,resume=True)
                self.start_iter = self.iter + 1
                return check
        else:
            # 如果是初始化，加载指定的权重
            if self.cfg.MODEL.WEIGHTS:
                return self.custom_load(self.cfg.MODEL.WEIGHTS,resume=False)
        return {}


    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        """
        It now calls :func:`detectron2.solver.build_lr_scheduler`.
        Overwrite it if you'd like a different scheduler.
        """
        return StepLR(optimizer, step_size=10000, gamma=1)
        # return build_lr_scheduler(cfg, optimizer)

    @classmethod
    def build_optimizer(cls, cfg, model):
        weight_decay_norm = cfg.SOLVER.WEIGHT_DECAY_NORM
        weight_decay_embed = cfg.SOLVER.WEIGHT_DECAY_EMBED

        defaults = {}
        defaults["lr"] = cfg.SOLVER.BASE_LR
        defaults["weight_decay"] = cfg.SOLVER.WEIGHT_DECAY

        norm_module_types = (
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
            torch.nn.BatchNorm3d,
            torch.nn.SyncBatchNorm,
            # NaiveSyncBatchNorm inherits from BatchNorm2d
            torch.nn.GroupNorm,
            torch.nn.InstanceNorm1d,
            torch.nn.InstanceNorm2d,
            torch.nn.InstanceNorm3d,
            torch.nn.LayerNorm,
            torch.nn.LocalResponseNorm,
        )

        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        for module_name, module in model.named_modules():
            for module_param_name, value in module.named_parameters(recurse=False):
                if not value.requires_grad:
                    continue
                # Avoid duplicating parameters
                if value in memo:
                    continue
                memo.add(value)

                hyperparams = copy.copy(defaults)
                if "backbone" in module_name:
                    hyperparams["lr"] = hyperparams["lr"] * cfg.SOLVER.BACKBONE_MULTIPLIER
                if (
                    "relative_position_bias_table" in module_param_name
                    or "absolute_pos_embed" in module_param_name
                ):
                    # print(module_param_name)
                    hyperparams["weight_decay"] = 0.0
                if isinstance(module, norm_module_types):
                    hyperparams["weight_decay"] = weight_decay_norm
                if isinstance(module, torch.nn.Embedding):
                    hyperparams["weight_decay"] = weight_decay_embed
                params.append({"params": [value], **hyperparams})

        def maybe_add_full_model_gradient_clipping(optim):
            # detectron2 doesn't have full model gradient clipping now
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
            )
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer

class Trainer_forget(DefaultTrainer):
    """
    Extension of the Trainer class adapted to MaskFormer.
    """
    def __init__(self, cfg):
        super().__init__(cfg)
        self.logger = logging.getLogger("detectron2")
        if not self.logger.isEnabledFor(logging.INFO):  # setup_logger is not called for d2
            self.logger=setup_logger()
        cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())

        # Assume these objects must be constructed in this order.
        self.model = self.build_model(cfg)
        self.optimizer = self.build_optimizer(cfg, self.model)

        self.data_loader_forget,self.data_loader_remain = self.build_train_loader(cfg)
        # dataloader
        self.data_loader_forget_iter = iter(self.data_loader_forget)
        self.data_loader_remain_iter = iter(self.data_loader_remain)

        self.model = create_ddp_model(self.model, broadcast_buffers=False,find_unused_parameters=True)

        # self._trainer = (AMPTrainer if cfg.SOLVER.AMP.ENABLED else SimpleTrainer)(
        #     self.model, self.data_loader_remain, self.optimizer
        # )

        self.scheduler = self.build_lr_scheduler(cfg, self.optimizer)
        self._trainer = SimpleTrainer(
            self.model, self.data_loader_forget, self.optimizer
        )
        # add model EMA
        kwargs = {
            'trainer': weakref.proxy(self),
        }
        # kwargs.update(model_ema.may_get_ema_checkpointer(cfg, model)) TODO: release ema training for large models
        self.checkpointer = DetectionCheckpointer(
            # Assume you want to save checkpoints together with logs/statistics
            self.model,
            cfg.OUTPUT_DIR,
            **kwargs,
        )
        self.start_iter = 0
        self.cfg = cfg
        # TODO: release model conversion checkpointer from DINO to MaskDINO
        self.checkpointer = DetectionCheckpointer(
            # Assume you want to save checkpoints together with logs/statistics
            self.model,
            cfg.OUTPUT_DIR,
            **kwargs,
        )
        # TODO: release GPU cluster submit scripts based on submitit for multi-node training
        self.current_iter = 0
        self.grad_dic = {}
        self.grad_dic_abs ={}

    def train(self):
        # 使用自定义 run_step 的训练循环
        super().train()  # 调用父类的 train 方法，会使用自定义的 run_step


    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each
        builtin dataset. For your own dataset, you can simply create an
        evaluator manually in your script and do not have to worry about the
        hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")

        return SemSegEvaluator(
            dataset_name,
            output_dir=output_folder,
        )

    @classmethod
    def build_train_loader(cls, cfg):
        """
        Create two data loaders, each for a different dataset.
        """
        # Configure data loader for dataset1
        cfg1 = cfg.clone()
        cfg1.defrost()
        cfg1.DATASETS.TRAIN = ("forget_dataset_train",)  # 修改为第一个数据集的名称

        mapper1 = CustomMaskFormerSemanticDatasetMapper.from_config(cfg1, is_train=True)
        data_loader_forget = build_detection_train_loader(cfg1, mapper=mapper1)
        cfg2 = cfg.clone()
        cfg2.defrost()
        cfg2.DATASETS.TRAIN = ("remain_dataset_train",)
        # 这个是给小数据集voc用的
        # cfg2.DATASETS.SAMPLE_WEIGHTS = (1.0, 3.0)  # 按数据集大小比例设置
        # # 这个是给小数据集sat卫星用的
        # cfg2.DATASETS.SAMPLE_WEIGHTS = (1.0, 3.0)  # 按数据集大小比例设置
        # # 这个是给小数据集coco用的
        # cfg2.DATASETS.SAMPLE_WEIGHTS = (1.0, 1.5)  # 按数据集大小比例设置
        # 这个是给小数据集pet用的
        # cfg2.DATASETS.SAMPLE_WEIGHTS = (1.0, 2.0)  # 按数据集大小比例设置
        # cfg2.DATASETS.TRAIN = ("remain_dataset_train",)

        mapper2 = CustomMaskFormerSemanticDatasetMapper.from_config(cfg2, is_train=True)
        data_loader_remain = build_detection_train_loader(cfg2, mapper=mapper2)
        return data_loader_forget, data_loader_remain

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        if dataset_name == "remain_dataset_val":
            pixel_mean = torch.tensor([123.675, 116.280, 103.530], dtype=torch.float32).view(3, 1, 1)
            pixel_std = torch.tensor([58.395, 57.120, 57.375], dtype=torch.float32).view(3, 1, 1)

        elif dataset_name == "learn_dataset_val":
            pixel_mean = torch.tensor([119.73433564, 109.83269262, 94.84763627], dtype=torch.float32).view(3, 1, 1)
            pixel_std = torch.tensor([57.6707407, 57.19914482, 58.46454528], dtype=torch.float32).view(3, 1, 1)
        elif dataset_name == "forget_dataset_val":
            pixel_mean = torch.tensor([123.675, 116.280, 103.530], dtype=torch.float32).view(3, 1, 1)
            pixel_std = torch.tensor([58.395, 57.120, 57.375], dtype=torch.float32).view(3, 1, 1)
        else:
            raise ValueError(f"Unknown dataset name: {dataset_name}")
            # print(f"数据集：{dataset_name},pixel_mean：{pixel_mean},pixel_std：{pixel_std}")
            # print("fdsaafdasfdasfdas111111111111")
            # print()
        mapper = testMapper(mean=pixel_mean, std=pixel_std)
        test_dataset_loader = build_detection_test_loader(cfg, dataset_name, mapper=mapper)
        return test_dataset_loader
    def run_step(self):
        """
        Modified training step to load data from two loaders and calculate losses separately.
        """
        assert self.model.training, "[Trainer] Model is not in training mode."

        # Get data from both data loaders
        data_forget = next(self.data_loader_forget_iter, None)
        data_remain = next(self.data_loader_remain_iter, None)

        # Reset iterator if any data loader is exhausted
        if data_forget is None:
            self.data_loader_forget_iter = iter(self.data_loader_forget)
            data_forget = next(self.data_loader_forget_iter)

        if data_remain is None:
            self.data_loader_remain_iter = iter(self.data_loader_remain)
            data_remain = next(self.data_loader_remain_iter)

        # Compute losses separately
        loss_dict_forget = self.model(data_forget)
        loss_dict_remain = self.model(data_remain)
        structure_loss = get_structure_loss(self.model)
        # Calculate total loss

        losses_forget = sum(loss for loss in loss_dict_forget.values())
        losses_remain = sum(loss for loss in loss_dict_remain.values())
        BND = self.cfg.FORGET_BND
        beta = self.cfg.FORGET_BETA
        alpha = self.cfg.FORGET_ALPHA
        relu_forget =torch.functional.F.relu(BND-losses_forget)

        total_loss = beta*relu_forget + losses_remain + alpha*structure_loss
        if self.iter % 20 == 0:
            self.logger.info(
                f"当前迭代：{self.iter} losses_forget: {losses_forget}, losses_remain: {losses_remain}, structure_loss: {structure_loss}")

        # Backward and optimizer steps
        self.optimizer.zero_grad()
        total_loss.backward()

        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad_mean = param.grad.mean().item()
                grad_mean_abs = param.grad.abs().mean().item()
                self.grad_dic[name] = grad_mean + self.grad_dic.get(name, 0.0)
                self.grad_dic_abs[name] = grad_mean_abs + self.grad_dic_abs.get(name, 0.0)

        self.optimizer.step()
        # Update learning rate
        # self.scheduler.step()

        if self.iter % 2000 == 1000 and self.iter != 0:
            self.custom_save(f"forget_model_{self.current_iter:07d}")
            # Update iteration counter
        if self.iter%1000 == 0 and self.iter != 0:
            for key, val in self.grad_dic.items():
                self.grad_dic[key] = val / self.iter
            save_path = os.path.join(self.cfg.OUTPUT_DIR, f"forget_grad_{self.iter}.json")
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(self.grad_dic, f, ensure_ascii=False, indent=4)

            for key, val in self.grad_dic_abs.items():
                self.grad_dic_abs[key] = val / self.iter
            save_path = os.path.join(self.cfg.OUTPUT_DIR, f"forget_grad_{self.iter}_abs.json")
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(self.grad_dic_abs, f, ensure_ascii=False, indent=4)

        self.current_iter += 1

    def custom_save(self, name):
        """
        自定义保存函数，保存完整的模型状态
        """
        logger = logging.getLogger("detectron2.trainer")
        logger.info(f"Saving model checkpoint: {name}")
        if isinstance(self.model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
            model_without_ddp = self.model.module
        else:
            model_without_ddp = self.model
        save_dict = {
            "model": model_without_ddp.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler else None,
            "iteration": self.iter,
        }

        save_path = os.path.join(self.cfg.OUTPUT_DIR, f"{name}.pth")
        torch.save(save_dict, save_path)

        # 验证保存的检查点
        loaded = torch.load(save_path)
        if "model" in loaded:
            logger.info(f"Successfully saved checkpoint: {save_path}")
        else:
            logger.error(f"Failed to save checkpoint: {save_path}")

    def custom_load(self, path, resume=False):
        """
        自定义加载函数
        """
        logger = logging.getLogger("detectron2.trainer")
        logger.info(f"Loading checkpoint from: {path}")

        checkpoint = torch.load(path, map_location=torch.device("cpu"))

        # 加载模型权重
        # self.model.load_state_dict(checkpoint["model"],strict=False)
        if isinstance(self.model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
            missing_keys, unexpected_keys = self.model.module.load_state_dict(checkpoint["model"], strict=False)
        else:
            missing_keys, unexpected_keys = self.model.load_state_dict(checkpoint["model"], strict=False)

        if len(missing_keys) > 0:
            logger.warning(f"Missing keys in checkpoint: {missing_keys}")
        if len(unexpected_keys) > 0:
            logger.warning(f"Unexpected keys in checkpoint: {unexpected_keys}")
        if resume:
            # 加载优化器状态
            if "optimizer" in checkpoint and self.optimizer:
                logger.info("Loading optimizer from checkpoint")
                self.optimizer.load_state_dict(checkpoint["optimizer"])

            # 加载调度器状态
            if "scheduler" in checkpoint and self.scheduler:
                logger.info("Loading scheduler from checkpoint")
                self.scheduler.load_state_dict(checkpoint["scheduler"])

            # 恢复迭代次数
            if "iteration" in checkpoint:
                self.iter = checkpoint["iteration"]

        return checkpoint

    def resume_or_load(self, resume=False):
        """
                重写resume_or_load方法以使用自定义的加载函数
                """
        if resume:
            if self.cfg.MODEL.WEIGHTS:
                check = self.custom_load(self.cfg.MODEL.WEIGHTS, resume=True)
                self.start_iter = self.iter + 1
                return check
        else:
            # 如果是初始化，加载指定的权重
            if self.cfg.MODEL.WEIGHTS:
                return self.custom_load(self.cfg.MODEL.WEIGHTS, resume=False)
        return {}

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        """
        It now calls :func:`detectron2.solver.build_lr_scheduler`.
        Overwrite it if you'd like a different scheduler.
        """
        return StepLR(optimizer, step_size=10000, gamma=1)
        # return build_lr_scheduler(cfg, optimizer)

    @classmethod
    def build_optimizer(cls, cfg, model):
        weight_decay_norm = cfg.SOLVER.WEIGHT_DECAY_NORM
        weight_decay_embed = cfg.SOLVER.WEIGHT_DECAY_EMBED

        defaults = {}
        defaults["lr"] = cfg.SOLVER.BASE_LR
        defaults["weight_decay"] = cfg.SOLVER.WEIGHT_DECAY

        norm_module_types = (
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
            torch.nn.BatchNorm3d,
            torch.nn.SyncBatchNorm,
            # NaiveSyncBatchNorm inherits from BatchNorm2d
            torch.nn.GroupNorm,
            torch.nn.InstanceNorm1d,
            torch.nn.InstanceNorm2d,
            torch.nn.InstanceNorm3d,
            torch.nn.LayerNorm,
            torch.nn.LocalResponseNorm,
        )

        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        for module_name, module in model.named_modules():
            for module_param_name, value in module.named_parameters(recurse=False):
                if not value.requires_grad:
                    continue
                # Avoid duplicating parameters
                if value in memo:
                    continue
                memo.add(value)

                hyperparams = copy.copy(defaults)
                if "backbone" in module_name:
                    hyperparams["lr"] = hyperparams["lr"] * cfg.SOLVER.BACKBONE_MULTIPLIER
                if (
                    "relative_position_bias_table" in module_param_name
                    or "absolute_pos_embed" in module_param_name
                ):
                    # print(module_param_name)
                    hyperparams["weight_decay"] = 0.0
                if isinstance(module, norm_module_types):
                    hyperparams["weight_decay"] = weight_decay_norm
                if isinstance(module, torch.nn.Embedding):
                    hyperparams["weight_decay"] = weight_decay_embed
                params.append({"params": [value], **hyperparams})

        def maybe_add_full_model_gradient_clipping(optim):
            # detectron2 doesn't have full model gradient clipping now
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
            )
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer


class Trainer_forget_twice(DefaultTrainer):
    """
    Extension of the Trainer class adapted to MaskFormer.
    """
    def __init__(self, cfg):
        super().__init__(cfg)
        self.logger = logging.getLogger("detectron2")
        if not self.logger.isEnabledFor(logging.INFO):  # setup_logger is not called for d2
            self.logger=setup_logger()
        cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())

        # Assume these objects must be constructed in this order.
        self.model = self.build_model(cfg)
        self.optimizer = self.build_optimizer(cfg, self.model)

        self.data_loader_forget,self.data_loader_remain = self.build_train_loader(cfg)
        # dataloader
        self.data_loader_forget_iter = iter(self.data_loader_forget)
        self.data_loader_remain_iter = iter(self.data_loader_remain)

        self.model = create_ddp_model(self.model, broadcast_buffers=False,find_unused_parameters=True)

        # self._trainer = (AMPTrainer if cfg.SOLVER.AMP.ENABLED else SimpleTrainer)(
        #     self.model, self.data_loader_remain, self.optimizer
        # )

        self.scheduler = self.build_lr_scheduler(cfg, self.optimizer)
        self._trainer = SimpleTrainer(
            self.model, self.data_loader_forget, self.optimizer
        )
        # add model EMA
        kwargs = {
            'trainer': weakref.proxy(self),
        }
        # kwargs.update(model_ema.may_get_ema_checkpointer(cfg, model)) TODO: release ema training for large models
        self.checkpointer = DetectionCheckpointer(
            # Assume you want to save checkpoints together with logs/statistics
            self.model,
            cfg.OUTPUT_DIR,
            **kwargs,
        )
        self.start_iter = 0
        self.cfg = cfg
        # TODO: release model conversion checkpointer from DINO to MaskDINO
        self.checkpointer = DetectionCheckpointer(
            # Assume you want to save checkpoints together with logs/statistics
            self.model,
            cfg.OUTPUT_DIR,
            **kwargs,
        )
        # TODO: release GPU cluster submit scripts based on submitit for multi-node training
        self.current_iter = 0

        self.grad_dic = {}
        self.grad_dic_abs = {}

    def train(self):
        # 使用自定义 run_step 的训练循环
        super().train()  # 调用父类的 train 方法，会使用自定义的 run_step


    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each
        builtin dataset. For your own dataset, you can simply create an
        evaluator manually in your script and do not have to worry about the
        hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")

        return SemSegEvaluator(
            dataset_name,
            output_dir=output_folder,
        )

    @classmethod
    def build_train_loader(cls, cfg):
        """
        Create two data loaders, each for a different dataset.
        """
        # Configure data loader for dataset1
        cfg1 = cfg.clone()
        cfg1.defrost()
        cfg1.DATASETS.TRAIN = ("forget_dataset_train",)  # 修改为第一个数据集的名称

        mapper1 = CustomMaskFormerSemanticDatasetMapper.from_config(cfg1, is_train=True)
        data_loader_forget = build_detection_train_loader(cfg1, mapper=mapper1)
        cfg2 = cfg.clone()
        cfg2.defrost()
        cfg2.DATASETS.TRAIN = ("remain_dataset_train","learn_dataset_train")
        if cfg2.DATASETS.LEARN_DATASET_NAME == "voc":
        # 这个是给小数据集voc用的
            cfg2.DATASETS.SAMPLE_WEIGHTS = (1.0, 3.0)  # 按数据集大小比例设置
        elif cfg2.DATASETS.LEARN_DATASET_NAME == "sat":
        # # 这个是给小数据集sat卫星用的
            cfg2.DATASETS.SAMPLE_WEIGHTS = (1.0, 3.0)  # 按数据集大小比例设置
        elif cfg2.DATASETS.LEARN_DATASET_NAME == "coco":
        # # 这个是给小数据集coco用的
            cfg2.DATASETS.SAMPLE_WEIGHTS = (1.0, 1.5)  # 按数据集大小比例设置
        elif cfg2.DATASETS.LEARN_DATASET_NAME == "pet":
        # 这个是给小数据集pet用的
            cfg2.DATASETS.SAMPLE_WEIGHTS = (1.0, 2.0)  # 按数据集大小比例设置



        # cfg2.DATASETS.TRAIN = ("remain_dataset_train",)
        mapper2 = CustomMaskFormerSemanticDatasetMapper.from_config(cfg2, is_train=True)
        data_loader_remain = build_detection_train_loader(cfg2, mapper=mapper2)
        return data_loader_forget, data_loader_remain

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        if dataset_name == "remain_dataset_val":
            pixel_mean = torch.tensor([123.675, 116.280, 103.530], dtype=torch.float32).view(3, 1, 1)
            pixel_std = torch.tensor([58.395, 57.120, 57.375], dtype=torch.float32).view(3, 1, 1)

        elif dataset_name == "learn_dataset_val":
            pixel_mean = torch.tensor([119.73433564, 109.83269262, 94.84763627], dtype=torch.float32).view(3, 1, 1)
            pixel_std = torch.tensor([57.6707407, 57.19914482, 58.46454528], dtype=torch.float32).view(3, 1, 1)
        elif dataset_name == "forget_dataset_val":
            pixel_mean = torch.tensor([123.675, 116.280, 103.530], dtype=torch.float32).view(3, 1, 1)
            pixel_std = torch.tensor([58.395, 57.120, 57.375], dtype=torch.float32).view(3, 1, 1)
        else:
            raise ValueError(f"Unknown dataset name: {dataset_name}")

        mapper = testMapper(mean=pixel_mean, std=pixel_std)
        test_dataset_loader = build_detection_test_loader(cfg, dataset_name, mapper=mapper)
        return test_dataset_loader
    def run_step(self):
        """
        Modified training step to load data from two loaders and calculate losses separately.
        """
        assert self.model.training, "[Trainer] Model is not in training mode."

        # Get data from both data loaders
        data_forget = next(self.data_loader_forget_iter, None)
        data_remain = next(self.data_loader_remain_iter, None)

        # Reset iterator if any data loader is exhausted
        if data_forget is None:
            self.data_loader_forget_iter = iter(self.data_loader_forget)
            data_forget = next(self.data_loader_forget_iter)

        if data_remain is None:
            self.data_loader_remain_iter = iter(self.data_loader_remain)
            data_remain = next(self.data_loader_remain_iter)

        # Compute losses separately
        loss_dict_forget = self.model(data_forget)
        loss_dict_remain = self.model(data_remain)
        structure_loss = get_structure_loss(self.model)
        # Calculate total loss

        losses_forget = sum(loss for loss in loss_dict_forget.values())
        losses_remain = sum(loss for loss in loss_dict_remain.values())
        BND = self.cfg.FORGET_BND
        beta = self.cfg.FORGET_BETA
        alpha = self.cfg.FORGET_ALPHA
        relu_forget =torch.functional.F.relu(BND-losses_forget)

        total_loss = beta*relu_forget + losses_remain + alpha*structure_loss
        if self.iter % 20 == 0:
            self.logger.info(
                f"当前迭代：{self.iter} losses_forget: {losses_forget}, losses_remain: {losses_remain}, structure_loss: {structure_loss}")

        # Backward and optimizer steps
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad_mean = param.grad.mean().item()
                grad_mean_abs = param.grad.abs().mean().item()
                self.grad_dic[name] = grad_mean + self.grad_dic.get(name, 0.0)
                self.grad_dic_abs[name] = grad_mean_abs + self.grad_dic_abs.get(name, 0.0)



        # Update learning rate
        # self.scheduler.step()

        if self.iter % 2000 == 1000 and self.iter != 0:
            self.custom_save(f"forget_twice_model_{self.iter:07d}")

        # Update iteration counter
        self.current_iter += 1

    def custom_save(self, name):
        """
        自定义保存函数，保存完整的模型状态
        """
        logger = logging.getLogger("detectron2.trainer")
        logger.info(f"Saving model checkpoint: {name}")
        if isinstance(self.model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
            model_without_ddp = self.model.module
        else:
            model_without_ddp = self.model
        save_dict = {
            "model": model_without_ddp.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler else None,
            "iteration": self.iter,
        }

        save_path = os.path.join(self.cfg.OUTPUT_DIR, f"{name}.pth")
        torch.save(save_dict, save_path)

        # 验证保存的检查点
        loaded = torch.load(save_path)
        if "model" in loaded:
            logger.info(f"Successfully saved checkpoint: {save_path}")
        else:
            logger.error(f"Failed to save checkpoint: {save_path}")

    def custom_load(self, path, resume=False):
        """
        自定义加载函数
        """
        logger = logging.getLogger("detectron2.trainer")
        logger.info(f"Loading checkpoint from: {path}")

        checkpoint = torch.load(path, map_location=torch.device("cpu"))

        # 加载模型权重
        # self.model.load_state_dict(checkpoint["model"],strict=False)
        if isinstance(self.model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
            missing_keys, unexpected_keys = self.model.module.load_state_dict(checkpoint["model"], strict=False)
        else:
            missing_keys, unexpected_keys = self.model.load_state_dict(checkpoint["model"], strict=False)

        if len(missing_keys) > 0:
            logger.warning(f"Missing keys in checkpoint: {missing_keys}")
        if len(unexpected_keys) > 0:
            logger.warning(f"Unexpected keys in checkpoint: {unexpected_keys}")
        if resume:
            # 加载优化器状态
            if "optimizer" in checkpoint and self.optimizer:
                logger.info("Loading optimizer from checkpoint")
                self.optimizer.load_state_dict(checkpoint["optimizer"])

            # 加载调度器状态
            if "scheduler" in checkpoint and self.scheduler:
                logger.info("Loading scheduler from checkpoint")
                self.scheduler.load_state_dict(checkpoint["scheduler"])

            # 恢复迭代次数
            if "iteration" in checkpoint:
                self.iter = checkpoint["iteration"]

        return checkpoint

    def resume_or_load(self, resume=False):
        """
                重写resume_or_load方法以使用自定义的加载函数
                """
        if resume:
            if self.cfg.MODEL.WEIGHTS:
                check = self.custom_load(self.cfg.MODEL.WEIGHTS, resume=True)
                self.start_iter = self.iter + 1
                return check
        else:
            # 如果是初始化，加载指定的权重
            if self.cfg.MODEL.WEIGHTS:
                return self.custom_load(self.cfg.MODEL.WEIGHTS, resume=False)
        return {}

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        """
        It now calls :func:`detectron2.solver.build_lr_scheduler`.
        Overwrite it if you'd like a different scheduler.
        """
        return StepLR(optimizer, step_size=10000, gamma=1)
        # return build_lr_scheduler(cfg, optimizer)

    @classmethod
    def build_optimizer(cls, cfg, model):
        weight_decay_norm = cfg.SOLVER.WEIGHT_DECAY_NORM
        weight_decay_embed = cfg.SOLVER.WEIGHT_DECAY_EMBED

        defaults = {}
        defaults["lr"] = cfg.SOLVER.BASE_LR
        defaults["weight_decay"] = cfg.SOLVER.WEIGHT_DECAY

        norm_module_types = (
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
            torch.nn.BatchNorm3d,
            torch.nn.SyncBatchNorm,
            # NaiveSyncBatchNorm inherits from BatchNorm2d
            torch.nn.GroupNorm,
            torch.nn.InstanceNorm1d,
            torch.nn.InstanceNorm2d,
            torch.nn.InstanceNorm3d,
            torch.nn.LayerNorm,
            torch.nn.LocalResponseNorm,
        )

        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        for module_name, module in model.named_modules():
            for module_param_name, value in module.named_parameters(recurse=False):
                if not value.requires_grad:
                    continue
                # Avoid duplicating parameters
                if value in memo:
                    continue
                memo.add(value)

                hyperparams = copy.copy(defaults)
                if "backbone" in module_name:
                    hyperparams["lr"] = hyperparams["lr"] * cfg.SOLVER.BACKBONE_MULTIPLIER
                if (
                    "relative_position_bias_table" in module_param_name
                    or "absolute_pos_embed" in module_param_name
                ):
                    # print(module_param_name)
                    hyperparams["weight_decay"] = 0.0
                if isinstance(module, norm_module_types):
                    hyperparams["weight_decay"] = weight_decay_norm
                if isinstance(module, torch.nn.Embedding):
                    hyperparams["weight_decay"] = weight_decay_embed
                params.append({"params": [value], **hyperparams})

        def maybe_add_full_model_gradient_clipping(optim):
            # detectron2 doesn't have full model gradient clipping now
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
            )
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer



def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    # for poly lr schedule
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.INPUT.LEARN_DATASET_NAME = args.learnsetname
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    # Setup logger for "mask_former" module
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="mask2former")
    return cfg


def register_datasets(args):
    learn_name = args.learnsetname

    image_dir_learn = args.learnset + "/train/images"
    annotation_dir_learn = args.learnset + "/train/annotations"
    image_dir_learn_val = args.learnset + "/test/images"
    annotation_dir_learn_val = args.learnset + "/test/annotations"

    if learn_name == "voc":
        # 注册数据集  voc 学习集
        register_sat(image_dir_learn, annotation_dir_learn, image_dir_learn_val, annotation_dir_learn_val)
    elif learn_name == "pet":
        # 注册数据集  pet 学习集
        register_pet(image_dir_learn, annotation_dir_learn, image_dir_learn_val, annotation_dir_learn_val)
    elif learn_name == "coco":
        # 注册数据集  coco 学习集
        register_coco(image_dir_learn, annotation_dir_learn, image_dir_learn_val, annotation_dir_learn_val)

    elif learn_name == "sat":
        # 注册数据集  sat 学习集
        register_sat_new(image_dir_learn, annotation_dir_learn, image_dir_learn_val, annotation_dir_learn_val)

    # 注册数据集  forget 学习集 ade
    image_dir_forget = args.forgetset + "/images/training"
    annotation_dir_forget = args.forgetset + "/annotations_detectron2/training"
    image_dir_forget_val = args.forgetset + "/images/validation"
    annotation_dir_forget_val = args.forgetset + "/annotations_detectron2/validation"
    register_forget_ade(image_dir_forget, annotation_dir_forget, image_dir_forget_val, annotation_dir_forget_val)

    # 注册数据集  remain 保持集 ade
    image_dir_remain = args.retainset + "/images/training"
    annotation_dir_remain = args.retainset + "/annotations_detectron2/training"
    image_dir_remain_val = args.retainset + "/images/validation"
    annotation_dir_remain_val = args.retainset + "/annotations_detectron2/validation"
    register_remain_ade(image_dir_remain, annotation_dir_remain, image_dir_remain_val, annotation_dir_remain_val)


def learn(args):
    register_datasets(args)
    cfg = setup(args)
    cfg.defrost()
    # cfg.OUTPUT_DIR = os.path.join(cfg.OUTPUT_DIR, time_str)
    cfg.DATASETS.TEST=("remain_dataset_val","learn_dataset_val","forget_dataset_val")

    # print("Command cfg:", cfg)
    if args.eval_only:
        trainer = Trainer_learn(cfg)
        trainer.custom_load(cfg.MODEL.WEIGHTS,resume=False)  # 直接加载指定权重
        # model = Trainer_learn.build_model(cfg)
        # DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        #     cfg.MODEL.WEIGHTS, resume=False
        # )
        # checkpointer = DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR)
        # checkpointer.resume_or_load(
        #     cfg.MODEL.WEIGHTS, resume=args.resume
        # )
        trainer.model.eval()
        res = Trainer_learn.test(cfg, trainer.model)
        # if cfg.TEST.AUG.ENABLED:
        #     res.update(Trainer_learn.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res
    max_iter = cfg.SOLVER.MAX_ITER
    new_max_iter = max_iter - (max_iter % 1000)
    cfg.MODEL.WEIGHTS = cfg.OUTPUT_DIR + "/" + f"forget_model_000{new_max_iter}.pth"
    cfg.freeze()
    trainer = Trainer_learn(cfg)
    # args.resume = True
    # trainer.resume_or_load(resume=args.resume)
    trainer.custom_load(cfg.MODEL.WEIGHTS)
    if cfg.MODEL.SEM_SEG_HEAD.LORA_RANK > 0:
        lora.mark_only_lora_as_trainable(trainer.model)
        # lora 初始化 一定注意初始化，第二次吧结果冲没了！！！！！！！！
        # reinitialize_lora_parameters(trainer.model)
        print("Use LoRA in Transformer FFN, loar_rank: ", cfg.MODEL.SEM_SEG_HEAD.LORA_RANK)
    else:
        print(
            "Do not use LoRA in Transformer FFN, train all parameters."
        )

    return trainer.train()

def forget(args):
    register_datasets(args)
    cfg = setup(args)
    cfg.defrost()
    # cfg.DATASETS.Train = ("remain_dataset_train", "learn_dataset_train")
    cfg.DATASETS.TEST = ("remain_dataset_val", "forget_dataset_val","learn_dataset_val",)

    if args.eval_only:
        trainer = Trainer_forget(cfg)
        trainer.custom_load(cfg.MODEL.WEIGHTS, resume=False)  # 直接加载指定权重

        trainer.model.eval()
        res = Trainer_forget.test(cfg, trainer.model)

        if comm.is_main_process():
            verify_results(cfg, res)
        return res
    # cfg.SOLVER.MAX_ITER=7500
    # cfg.MODEL.WEIGHTS = cfg.OUTPUT_DIR + "/" + "learn_model_0007000.pth"
    # cfg.MODEL.WEIGHTS = "/home/xmy/code/Mask2Former-main/exp/log/2024-12-27-22-15-16-voc-bad/learn_model_0007000.pth"
    cfg.freeze()

    trainer = Trainer_forget(cfg)

    # 从持续学习的outdir 学习
    # args.resume = True
    #
    # trainer.resume_or_load(resume=args.resume)
    trainer.custom_load(cfg.MODEL.WEIGHTS)
    if cfg.MODEL.SEM_SEG_HEAD.LORA_RANK > 0:
        lora.mark_only_lora_as_trainable(trainer.model)
        reinitialize_lora_parameters(trainer.model)
        print("Use LoRA in Transformer FFN, loar_rank: ", cfg.MODEL.SEM_SEG_HEAD.LORA_RANK)
    else:
        print(
            "Do not use LoRA in Transformer FFN, train all parameters."
        )
    return trainer.train()

def forget_twice(args):
    register_datasets(args)
    cfg = setup(args)
    cfg.defrost()
    # cfg.DATASETS.Train = ("remain_dataset_train", "learn_dataset_train")
    cfg.DATASETS.TEST = ("remain_dataset_val", "forget_dataset_val","learn_dataset_val",)

    if args.eval_only:
        trainer = Trainer_forget_twice(cfg)
        trainer.custom_load(cfg.MODEL.WEIGHTS, resume=False)  # 直接加载指定权重

        trainer.model.eval()
        res = Trainer_forget_twice.test(cfg, trainer.model)

        if comm.is_main_process():
            verify_results(cfg, res)
        return res
    max_iter = cfg.SOLVER.MAX_ITER
    new_max_iter = max_iter-(max_iter%1000)
    cfg.MODEL.WEIGHTS = cfg.OUTPUT_DIR + "/" + f"learn_model_000{new_max_iter}.pth"
    # cfg.MODEL.WEIGHTS = "/home/xmy/code/papercodegithub/semantic-segmentation/exp/log/2025-02-09-22-50-24/learn_model_0003000.pth"
    cfg.freeze()

    trainer = Trainer_forget_twice(cfg)

    # 从持续学习的outdir 学习
    # args.resume = True
    #
    # trainer.resume_or_load(resume=args.resume)
    trainer.custom_load(cfg.MODEL.WEIGHTS)
    if cfg.MODEL.SEM_SEG_HEAD.LORA_RANK > 0:
        lora.mark_only_lora_as_trainable(trainer.model)
        # reinitialize_lora_parameters(trainer.model)
        print("Use LoRA in Transformer FFN, loar_rank: ", cfg.MODEL.SEM_SEG_HEAD.LORA_RANK)
    else:
        print(
            "Do not use LoRA in Transformer FFN, train all parameters."
        )
    return trainer.train()


if __name__ == "__main__":

    parser = default_argument_parser()
    parser.add_argument("--learnsetname", type=str, default="voc")
    parser.add_argument("--learnset", type=str, default="/data1/xmy/Mask2Former-main/voclearn")
    parser.add_argument("--forgetset", type=str, default="/data1/xmy/forgetsmall")
    parser.add_argument("--retainset", type=str, default="/data1/xmy/remainsmall")
    args = parser.parse_args()

    # 日志文件夹更改
    for i,item in enumerate(args.opts):
        if item == "OUTPUT_DIR" :
            args.opts[i+1] = args.opts[i+1] + "/" + time_str

    # args.opts[5] = args.opts[5] +"/"+ time_str
    port = random.randint(1000, 20000)
    args.dist_url = 'tcp://127.0.0.1:' + str(port)
    # print("Command Line Args:", args)
    # print("pwd:", os.getcwd())

    print("遗忘开始了")
    launch(
        forget,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )


    print("持续学习开始了")
    launch(
        learn,
         args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )

    print("遗忘开始了")
    launch(
        forget_twice,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )



