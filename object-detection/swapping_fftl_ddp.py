# Copyright (c) 2022 IDEA. All Rights Reserved.
# ------------------------------------------------------------------------
import argparse
import datetime
import json
import random
import time
from pathlib import Path
import os, sys
import numpy as np
import loralib as lora
import torch
from torch.utils.data import DataLoader, DistributedSampler
from datetime import datetime
from datetime import timedelta

from util.get_param_dicts import get_param_dict
from util.logger import setup_logger
from util.slconfig import DictAction, SLConfig
from util.utils import ModelEma, BestMetricHolder
import util.misc as utils

import datasets
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate, train_one_epoch, test, train_one_epoch_learn, train_one_epoch_forget, \
    train_one_epoch_forget_twice
from datasets.coco import build_learn,build_forget,build_remain
from util.utils import reinitialize_lora_parameters
from torch.utils.data import ConcatDataset
from engine import train_one_epoch_learn_test
def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('-config_file', '-c', type=str, required=True)
    parser.add_argument('--options',
                        nargs='+',
                        action=DictAction,
                        help='override some settings in the used config, the key-value pair '
                             'in xxx=yyy format will be merged into config file.')

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    # parser.add_argument('--coco_path', type=str, default='/comp_robot/cv_public_dataset/COCO2017/')
    parser.add_argument('--ococo_panptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')
    parser.add_argument('--fix_size', action='store_true')

    # lora微调
    parser.add_argument('--lora_rank', type=int, default=0)
    parser.add_argument('--lora_pos', type=str, default='FFN')
    # 学习集合 遗忘集 路径
    parser.add_argument("--remain_path", type=str, default="")
    parser.add_argument("--forget_path", type=str, default="")
    parser.add_argument("--learn_path", type=str, default="")

    # training parameters
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--note', default='',
                        help='add some notes to the experiment')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    # 回复断点文件夹
    parser.add_argument('--resume_dir', default="")

    parser.add_argument('--pretrain_model_path', help='load from other checkpoint')
    parser.add_argument('--finetune_ignore', type=str, nargs='+')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--find_unused_params', action='store_true')

    parser.add_argument('--save_results', action='store_true')
    parser.add_argument('--save_log', action='store_true')

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='number of distributed processes')
    parser.add_argument("--local_rank", type=int, help='local rank for DistributedDataParallel')
    parser.add_argument('--amp', action='store_true',
                        help="Train with mixed precision")

    return parser


def build_model_main(args):
    # we use register to maintain models from catdet6 on.
    from models.registry import MODULE_BUILD_FUNCS
    assert args.modelname in MODULE_BUILD_FUNCS._module_dict
    build_func = MODULE_BUILD_FUNCS.get(args.modelname)
    model, criterion, postprocessors = build_func(args)
    return model, criterion, postprocessors

def record(logger,stats,epoch=None):
    stats_all = stats['coco_eval_bbox']
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
    logger.info(f"{'=' * 20} COCO Evaluation Metrics {'=' * 20}")
    if epoch is not None:
        logger.info(f"Epoch: {epoch}")
    for name, value in zip(metric_names, stats_all):
        logger.info(f"{name}: {value:.3f}")
    stats_single = stats['category_ap']
    stats_filter = {name:ap for name,ap in stats_single.items() if ap != -1}
    print(stats_filter)
    logger.info(stats_filter)

def main(args):
    utils.init_distributed_mode(args)
    # load cfg file and update the args
    print("Loading config file from {}".format(args.config_file))
    time.sleep(args.rank * 0.02)
    cfg = SLConfig.fromfile(args.config_file)
    if args.options is not None:
        cfg.merge_from_dict(args.options)
    if args.rank == 0:
        save_cfg_path = os.path.join(args.output_dir, "config_cfg.py")
        cfg.dump(save_cfg_path)
        save_json_path = os.path.join(args.output_dir, "config_args_raw.json")
        with open(save_json_path, 'w') as f:
            json.dump(vars(args), f, indent=2)
    cfg_dict = cfg._cfg_dict.to_dict()
    args_vars = vars(args)
    for k, v in cfg_dict.items():
        if k not in args_vars:
            setattr(args, k, v)
        else:
            raise ValueError("Key {} can used by args only".format(k))

    # update some new args temporally
    if not getattr(args, 'use_ema', None):
        args.use_ema = False
    if not getattr(args, 'debug', None):
        args.debug = False

    # setup logger
    current = datetime.now()
    time_str = current.strftime('%m-%d-%H-%M-%S')
    args.output_dir = args.output_dir+time_str
    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logger(output=os.path.join(args.output_dir, 'info.txt'), distributed_rank=args.rank, color=False,
                          name="detr")
    # logger.info("git:\n  {}\n".format(utils.get_sha()))
    # logger.info("Command: " + ' '.join(sys.argv))
    if args.rank == 0:
        save_json_path = os.path.join(args.output_dir, "config_args_all.json")
        with open(save_json_path, 'w') as f:
            json.dump(vars(args), f, indent=2)
        logger.info("Full config saved to {}".format(save_json_path))
    logger.info('world size: {}'.format(args.world_size))
    logger.info('rank: {}'.format(args.rank))
    logger.info('local_rank: {}'.format(args.local_rank))
    # logger.info("args: " + str(args) + '\n')

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    # print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # build model
    model, criterion, postprocessors = build_model_main(args)
    wo_class_error = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # ema是false
    if args.use_ema:
        ema_m = ModelEma(model, args.ema_decay)
    else:
        ema_m = None

    model_without_ddp = model
    if args.distributed:
        # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu],
        #                                                   find_unused_parameters=args.find_unused_params)
        # model = torch.nn.parallel.DataParallel(model, device_ids=[args.gpu])
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.gpu],
            find_unused_parameters=args.find_unused_params,
            broadcast_buffers=False
        )
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info('number of params:' + str(n_parameters))
    # logger.info(
    #     "params:\n" + json.dumps({n: p.numel() for n, p in model.named_parameters() if p.requires_grad}, indent=2))

    param_dicts = get_param_dict(args, model_without_ddp)

    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)
    # 构建数据集 和dataloader
    # dataset_train = build_dataset(image_set='train', args=args)
    # dataset_val = build_dataset(image_set='val', args=args)
    dataset_train_remain = build_remain(image_set="train", args=args)
    dataset_val_remain = build_remain(image_set="val", args=args)
    dataset_train_forget = build_forget(image_set="train", args=args)
    dataset_val_forget = build_forget(image_set="val", args=args)
    dataset_train_learn = build_learn(image_set="train", args=args)
    dataset_val_learn = build_learn(image_set="val", args=args)

    if args.distributed:
        # pass
        # sampler_train = DistributedSampler(dataset_train)
        # sampler_val = DistributedSampler(dataset_val, shuffle=False)
        sampler_train_remain = DistributedSampler(dataset_train_remain)
        sampler_val_remain = DistributedSampler(dataset_val_remain,shuffle=False)
        sampler_train_forget = DistributedSampler(dataset_train_forget)
        sampler_val_forget = DistributedSampler(dataset_val_forget,shuffle=False)
        sampler_train_learn = DistributedSampler(dataset_train_learn)
        sampler_val_learn = DistributedSampler(dataset_val_learn,shuffle=False)
    else:
        # sampler_train = torch.utils.data.RandomSampler(dataset_train)
        # sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        sampler_train_remain = torch.utils.data.RandomSampler(dataset_train_remain)
        sampler_val_remain = torch.utils.data.SequentialSampler(dataset_val_remain)
        sampler_train_forget = torch.utils.data.RandomSampler(dataset_train_forget)
        sampler_val_forget = torch.utils.data.SequentialSampler(dataset_val_forget)
        sampler_train_learn = torch.utils.data.RandomSampler(dataset_train_learn)
        sampler_val_learn = torch.utils.data.SequentialSampler(dataset_val_learn)

    # batch_sampler_train = torch.utils.data.BatchSampler(
    #     sampler_train, args.batch_size, drop_last=True)

    # data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
    #                                collate_fn=utils.collate_fn, num_workers=args.num_workers)
    # data_loader_val = DataLoader(dataset_val, 1, sampler=sampler_val,
    #                              drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)

    batch_sampler_train_remain = torch.utils.data.BatchSampler(
        sampler_train_remain, args.batch_size, drop_last=True)
    data_loader_train_remain = DataLoader(dataset_train_remain, batch_sampler=batch_sampler_train_remain,
                                          collate_fn=utils.collate_fn, num_workers=args.num_workers)
    data_loader_val_remain = DataLoader(dataset_val_remain, 1, sampler=sampler_val_remain,
                                        drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)
    batch_sampler_train_forget = torch.utils.data.BatchSampler(
        sampler_train_forget, args.batch_size, drop_last=True)
    data_loader_train_forget = DataLoader(dataset_train_forget, batch_sampler=batch_sampler_train_forget,
                                          collate_fn=utils.collate_fn, num_workers=args.num_workers)
    data_loader_val_forget = DataLoader(dataset_val_forget, 1, sampler=sampler_val_forget,
                                        drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)
    batch_sampler_train_learn = torch.utils.data.BatchSampler(
        sampler_train_learn, args.batch_size, drop_last=True)
    data_loader_train_learn = DataLoader(dataset_train_learn, batch_sampler=batch_sampler_train_learn,
                                         collate_fn=utils.collate_fn, num_workers=args.num_workers)
    data_loader_val_learn = DataLoader(dataset_val_learn, 1, sampler=sampler_val_learn,
                                       drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)
    if args.onecyclelr:
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr,
                                                           steps_per_epoch=len(data_loader_train_remain), epochs=args.epochs,
                                                           pct_start=0.2)
    elif args.multi_step_lr:
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_drop_list)
    else:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop,gamma=0.3)



    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.detr.load_state_dict(checkpoint['model'])

    output_dir = Path(args.output_dir)

    if args.resume_dir:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_frsom_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume_dir, map_location='cpu')
        load_info = model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        missing_keys = load_info.missing_keys
        unexpected_keys = load_info.unexpected_keys
        if len(missing_keys) > 0:
            print("Missing keys: {}".format(missing_keys))
            print("\n")

        # 添加
        if args.use_ema:
            if 'ema_model' in checkpoint:
                ema_m.module.load_state_dict(utils.clean_state_dict(checkpoint['ema_model']))
            else:
                del ema_m
                ema_m = ModelEma(model, args.ema_decay)


        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1

    if (not args.resume) and args.pretrain_model_path:
        print(args)
        checkpoint = torch.load(args.pretrain_model_path, map_location='cpu')['model']
        from collections import OrderedDict
        _ignorekeywordlist = args.finetune_ignore if args.finetune_ignore else []
        ignorelist = []

        def check_keep(keyname, ignorekeywordlist):
            for keyword in ignorekeywordlist:
                if keyword in keyname:
                    ignorelist.append(keyname)
                    return False
            return True

        logger.info("Ignore keys: {}".format(json.dumps(ignorelist, indent=2)))
        _tmp_st = OrderedDict(
            {k: v for k, v   in utils.clean_state_dict(checkpoint).items() if check_keep(k, _ignorekeywordlist)})

        _load_output = model_without_ddp.load_state_dict(_tmp_st, strict=False)
        # logger.info(str(_load_output))
    # lora 初始化 卡靠你慢 就是因为这个  学习效果不好就是因为没有加初始化
    reinitialize_lora_parameters(model_without_ddp)

    if args.lora_rank > 0:
        lora.mark_only_lora_as_trainable(model_without_ddp)
        print("Use LoRA in Transformer FFN, loar_rank: ", args.lora_rank)

    else:
        print(
            "Do not use LoRA in Transformer FFN, train all parameters."
        )
    # 就是这句话导致断言报错
    # base_ds = get_coco_api_from_dataset(dataset_val_learn)
    # for x,targets in data_loader_train_learn:
    #     print(targets[0]["labels"].item())

    if args.eval:
        os.environ['EVAL_FLAG'] = 'TRUE'
        logger.info("保持集准确率")
        base_ds_remain = get_coco_api_from_dataset(dataset_val_remain)
        test_stats, coco_evaluator = evaluate(model, criterion, postprocessors,
                                              data_loader_val_remain, base_ds_remain, device, args.output_dir,
                                              wo_class_error=wo_class_error, args=args)
        record(logger,test_stats)
        logger.info("学习集准确率")
        base_ds_learn = get_coco_api_from_dataset(dataset_val_learn)
        test_stats_learn, coco_evaluator_learn = evaluate(model, criterion, postprocessors,
                                              data_loader_val_learn, base_ds_learn, device, args.output_dir,
                                              wo_class_error=wo_class_error, args=args)
        record(logger,test_stats_learn)
        logger.info("遗忘集准确率")
        base_ds_forget = get_coco_api_from_dataset(dataset_val_forget)
        test_stats_forget, coco_evaluator_forget = evaluate(model, criterion, postprocessors,
                                              data_loader_val_forget, base_ds_forget, device, args.output_dir,
                                              wo_class_error=wo_class_error, args=args)
        record(logger,test_stats_forget)

        # 保存 测试结果 和 日志
        # if args.output_dir:
        #     utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")
        #
        # log_stats = {**{f'test_{k}': v for k, v in test_stats.items()}}
        # if args.output_dir and utils.is_main_process():
        #     with (output_dir / "log.txt").open("a") as f:
        #         f.write(json.dumps(log_stats) + "\n")

        return



    logger.info("Start training")
    start_time = time.time()
    best_map_holder = BestMetricHolder(use_ema=args.use_ema)



    logger.info("持续遗忘开始了")
    # for epoch in range(args.start_epoch, args.epochs):
    for epoch in range(args.start_epoch, args.epochs):
        epoch_start_time = time.time()
        if args.distributed:
            # pass
            sampler_train_remain.set_epoch(epoch)
            sampler_train_forget.set_epoch(epoch)
        train_stats = train_one_epoch_forget(
            model, criterion, data_loader_train_remain,data_loader_train_forget, optimizer, device, epoch,
            args.clip_max_norm, wo_class_error=wo_class_error, lr_scheduler=lr_scheduler, args=args,
            logger=(logger if args.save_log else None), ema_m=ema_m)
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']

        if not args.onecyclelr:
            lr_scheduler.step()
        if args.output_dir:
            # checkpoint_paths = [output_dir / 'checkpoint_forget.pth']
            checkpoint_paths = []
            # extra checkpoint before LR drop and every 100 epochs
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % args.save_checkpoint_interval == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}_forget.pth')
            for checkpoint_path in checkpoint_paths:
                weights = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }
                # if args.use_ema:
                #     weights.update({
                #         'ema_model': ema_m.module.state_dict(),
                #     })
                utils.save_on_master(weights, checkpoint_path)

        # eval
        logger.info("评估保持集")
        base_ds_remain = get_coco_api_from_dataset(dataset_val_remain)
        test_stats, coco_evaluator = evaluate(
            model, criterion, postprocessors, data_loader_val_remain, base_ds_remain, device, args.output_dir,
            wo_class_error=wo_class_error, args=args, logger=(logger if args.save_log else None)
        )
        record(logger,test_stats,epoch)
        logger.info("评估遗忘集")
        base_ds_forget = get_coco_api_from_dataset(dataset_val_forget)
        test_stats_forget, coco_evaluator_forget = evaluate(model, criterion, postprocessors,
                                                            data_loader_val_forget, base_ds_forget, device,
                                                            args.output_dir,
                                                            wo_class_error=wo_class_error, args=args,
                                                            logger=(logger if args.save_log else None))
        record(logger, test_stats_forget, epoch)
        logger.info("评估学习集")
        base_ds_learn = get_coco_api_from_dataset(dataset_val_learn)
        test_stats_learn, coco_evaluator_learn = evaluate(model, criterion, postprocessors,
                                                          data_loader_val_learn, base_ds_learn, device, args.output_dir,
                                                          wo_class_error=wo_class_error, args=args,
                                                          logger=(logger if args.save_log else None))

        record(logger, test_stats_learn, epoch)
        map_regular = test_stats['coco_eval_bbox'][0]

        # 保存最佳权重
        _isbest = best_map_holder.update(map_regular, epoch, is_ema=False)
        # if _isbest:
        #     checkpoint_path = output_dir / 'checkpoint_best_regular_best.pth'
        #     utils.save_on_master({
        #         'model': model_without_ddp.state_dict(),
        #         'optimizer': optimizer.state_dict(),
        #         'lr_scheduler': lr_scheduler.state_dict(),
        #         'epoch': epoch,
        #         'args': args,
        #     }, checkpoint_path)
        # log_stats = {
        #     **{f'train_{k}': v for k, v in train_stats.items()},
        #     **{f'test_{k}': v for k, v in test_stats.items()},
        # }
        #
        #
        # log_stats.update(best_map_holder.summary())
        #
        # ep_paras = {
        #     'epoch': epoch,
        #     'n_parameters': n_parameters
        # }
        # log_stats.update(ep_paras)
        # try:
        #     log_stats.update({'now_time': str(datetime.datetime.now())})
        # except:
        #     pass
        #
        # epoch_time = time.time() - epoch_start_time
        # epoch_time_str = str(timedelta(seconds=int(epoch_time)))
        # log_stats['epoch_time'] = epoch_time_str
        #
        # if args.output_dir and utils.is_main_process():
        #     with (output_dir / "log.txt").open("a") as f:
        #         f.write(json.dumps(log_stats) + "\n")
        #
        #     # for evaluation logs
        #     if coco_evaluator is not None:
        #         (output_dir / 'eval').mkdir(exist_ok=True)
        #         if "bbox" in coco_evaluator.coco_eval:
        #             filenames = ['latest.pth']
        #             if epoch % 50 == 0:
        #                 filenames.append(f'{epoch:03}.pth')
        #             for name in filenames:
        #                 torch.save(coco_evaluator.coco_eval["bbox"].eval,
        #                            output_dir / "eval" / name)


    # 训练(持续学习
    logger.info("持续学习开始了")
    for epoch in range(args.start_epoch, args.epochs):
        epoch_start_time = time.time()
        if args.distributed:
            # pass
            sampler_train_remain.set_epoch(epoch)
            sampler_train_learn.set_epoch(epoch)
        train_stats = train_one_epoch_learn_test(
            model, criterion, data_loader_train_remain,data_loader_train_learn, optimizer, device, epoch,
            args.clip_max_norm, wo_class_error=wo_class_error, lr_scheduler=lr_scheduler, args=args,
            logger=(logger if args.save_log else None), ema_m=ema_m)

        if args.distributed:
            torch.distributed.barrier()  # 确保训练完成

        if not args.onecyclelr:
            lr_scheduler.step()
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint_learn.pth']
            checkpoint_paths = []
            # extra checkpoint before LR drop and every 100 epochs
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % args.save_checkpoint_interval == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}_learn.pth')
            for checkpoint_path in checkpoint_paths:
                weights = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }
                utils.save_on_master(weights, checkpoint_path)

        # eval
        logger.info("评估保持集")
        base_ds_remain = get_coco_api_from_dataset(dataset_val_remain)
        test_stats, coco_evaluator = evaluate(
            model, criterion, postprocessors, data_loader_val_remain, base_ds_remain, device, args.output_dir,
            wo_class_error=wo_class_error, args=args, logger=(logger if args.save_log else None)
        )
        record(logger, test_stats, epoch)
        logger.info("评估学习集")
        base_ds_learn = get_coco_api_from_dataset(dataset_val_learn)
        test_stats_learn, coco_evaluator_learn = evaluate(model, criterion, postprocessors,
                                                          data_loader_val_learn, base_ds_learn, device, args.output_dir,
                                                          wo_class_error=wo_class_error, args=args,logger=(logger if args.save_log else None))
        record(logger, test_stats_learn, epoch)
        logger.info("评估遗忘集")
        base_ds_forget = get_coco_api_from_dataset(dataset_val_forget)
        test_stats_forget, coco_evaluator_forget = evaluate(model, criterion, postprocessors,
                                                            data_loader_val_forget, base_ds_forget, device,
                                                            args.output_dir,
                                                            wo_class_error=wo_class_error, args=args,logger=(logger if args.save_log else None))
        record(logger, test_stats_forget, epoch)
        map_regular = test_stats['coco_eval_bbox'][0]

        # with (output_dir / "log.txt").open("a") as f:
        #     f.write(json.dumps(log_stats) + "\n")

        # 保存最佳权重
        # _isbest = best_map_holder.update(map_regular, epoch, is_ema=False)
        # if _isbest:
        #     checkpoint_path = output_dir / 'checkpoint_best_regular_learn.pth'
        #     utils.save_on_master({
        #         'model': model_without_ddp.state_dict(),
        #         'optimizer': optimizer.state_dict(),
        #         'lr_scheduler': lr_scheduler.state_dict(),
        #         'epoch': epoch,
        #         'args': args,
        #     }, checkpoint_path)



    # 训练时间
    total_time = time.time() - start_time
    total_time_str = str(timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    # remove the copied files.
    copyfilelist = vars(args).get('copyfilelist')
    if copyfilelist and args.local_rank == 0:
        from datasets.data_util import remove
        for filename in copyfilelist:
            print("Removing: {}".format(filename))
            remove(filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
