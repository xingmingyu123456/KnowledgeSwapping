import os, argparse, sklearn
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import wandb
import sys
import random
sys.path.append('/root/autodl-tmp/contious-diffusion')
from image_iter import CLDatasetWrapper, CustomSubset,MergedDataset
from config import get_config
from util.utils import (
    separate_irse_bn_paras,
    separate_resnet_bn_paras,
    separate_mobilefacenet_bn_paras,
)
from util.utils import (
    get_val_data,
    perform_val,
    get_time,
    buffer_val,
    AverageMeter,
    train_accuracy,
)
from util.utils import (
    split_dataset,
    count_trainable_parameters,
    reinitialize_lora_parameters,
)
from util.args import get_args

import time
from vit_pytorch_face import ViT_face, ViT_face_low, ViT_face_up
from vit_pytorch_face import ViTs_face

# from IPython import embed
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
import loralib as lora
from engine_cl import train_one_epoch, train_one_epoch_combine,eval_data, train_one_epoch_regularzation,train_one_epoch_learn
from torch.utils.data import Subset
from baselines.LIRFtrain import train_one_epoch_LIRF, eval_data_LIRF
from baselines.Lwftrain import train_one_epoch_Lwf
from baselines.DERtrain import train_one_epoch_DER
from baselines.FDRtrain import train_one_epoch_FDR
from baselines.SCRUBtrain import train_one_superepoch_SCRUB
from IPython import embed

import copy
from util.cal_norm import get_norm_of_lora
from torch.utils.data import DataLoader
import math
def split_learn_dataset(dataset, class_order_list):
    # 获取类别的数量
    num_classes = len(dataset.classes)

    # 每个分割部分的类别数量
    num_classes_per_split = num_classes // 5

    # 用于存储分割数据集的列表
    split_datasets = []

    for i in range(5):
        # 确定当前分割部分的类别索引范围
        split_start = i * num_classes_per_split
        split_end = (i + 1) * num_classes_per_split if i < 4 else num_classes

        split_class_indices = class_order_list[split_start:split_end]

        # 创建该分割部分的数据集
        split_samples = [
            (sample, label + 100)  # 将标签加100
            for sample, label in dataset.samples
            if label in split_class_indices
        ]
        split_dataset = datasets.ImageFolder(root=dataset.root, transform=transforms.ToTensor())
        split_dataset.samples = split_samples
        split_dataset.targets = [label for _, label in split_samples]
        split_dataset.classes = [dataset.classes[idx] for idx in split_class_indices]
        split_dataset.class_to_idx = {
            class_name: j + 100 for j, class_name in enumerate(split_dataset.classes)
        }

        # 将分割数据集添加到列表中
        split_datasets.append(split_dataset)

    return split_datasets


# def merge_datasets(dataset1, dataset2):
#     #变成dataset
#     if isinstance(dataset1, CustomSubset):
#         dataset1 = dataset1.dataset
#     if isinstance(dataset2, CustomSubset):
#         dataset2 = dataset2.dataset
#
#     # Combine the samples and targets from both datasets
#     merged_samples = dataset1.samples + dataset2.samples
#     merged_targets = dataset1.targets + dataset2.targets
#
#     # Since the labels are mutually exclusive, we can directly merge classes
#     merged_classes = dataset1.classes + dataset2.classes
#
#     # Create a new class_to_idx mapping
#     merged_class_to_idx = {class_name: i for i, class_name in enumerate(merged_classes)}
#
#     # Update the dataset to reflect the merged samples and classes
#     merged_dataset = datasets.ImageFolder(root=dataset1.root, transform=transforms.ToTensor())
#     merged_dataset.samples = merged_samples
#     merged_dataset.targets = merged_targets
#     merged_dataset.classes = merged_classes
#     merged_dataset.class_to_idx = merged_class_to_idx
#
#     return merged_dataset

def extract_subset_samples_and_targets(subset):
    # 提取与子集对应的 samples 和 targets
    samples = [subset.dataset.samples[i] for i in subset.indices]
    targets = [subset.dataset.targets[i] for i in subset.indices]
    return samples, targets

def merge_datasets(dataset1, dataset2):
    # 如果是 CustomSubset，从原始数据集中提取子集对应的 samples 和 targets
    if isinstance(dataset1, CustomSubset):
        samples1, targets1 = extract_subset_samples_and_targets(dataset1)
    else:
        samples1, targets1 = dataset1.samples, dataset1.targets

    if isinstance(dataset2, CustomSubset):
        samples2, targets2 = extract_subset_samples_and_targets(dataset2)
    else:
        samples2, targets2 = dataset2.samples, dataset2.targets

    # 合并 samples 和 targets
    merged_samples = samples1 + samples2
    merged_targets = targets1 + targets2

    # 合并 classes
    merged_classes = dataset1.classes + dataset2.classes
    merged_class_to_idx = {class_name: i for i, class_name in enumerate(merged_classes)}

    # 创建合并后的数据集
    merged_dataset = datasets.ImageFolder(root=dataset2.root, transform=transforms.ToTensor())
    merged_dataset.samples = merged_samples
    merged_dataset.targets = merged_targets
    merged_dataset.classes = merged_classes
    merged_dataset.class_to_idx = merged_class_to_idx

    return merged_dataset

if __name__ == "__main__":
    args = get_args()

    # ======= hyperparameters & data loaders =======#
    cfg = get_config(args)

    SEED = cfg["SEED"]  # random seed for reproduce results
    torch.manual_seed(SEED)

    DATA_ROOT = cfg[
        "DATA_ROOT"
    ]  # the parent root where your train/val/test data are stored
    EVAL_PATH = cfg["EVAL_PATH"]
    WORK_PATH = cfg[
        "WORK_PATH"
    ]  # the root to buffer your checkpoints and to log your train/val status
    BACKBONE_RESUME_ROOT = cfg[
        "BACKBONE_RESUME_ROOT"
    ]  # the root to resume training from a saved checkpoint

    BACKBONE_NAME = cfg["BACKBONE_NAME"]
    HEAD_NAME = cfg[
        "HEAD_NAME"
    ]  # support:  ['Softmax', 'ArcFace', 'CosFace', 'SFaceLoss']

    INPUT_SIZE = cfg["INPUT_SIZE"]
    EMBEDDING_SIZE = cfg["EMBEDDING_SIZE"]  # feature dimension
    BATCH_SIZE = cfg["BATCH_SIZE"]
    NUM_EPOCH = cfg["NUM_EPOCH"]

    DEVICE = cfg["DEVICE"]
    MULTI_GPU = cfg["MULTI_GPU"]  # flag to use multiple GPUs
    GPU_ID = cfg["GPU_ID"]  # specify your GPU ids
    print("GPU_ID", GPU_ID)
    WORKERS = cfg["WORKERS"]
    print("=" * 60)
    print("Overall Configurations:")
    print(cfg)
    with open(os.path.join(WORK_PATH, "config.txt"), "w") as f:
        f.write(str(cfg))
    print("=" * 60)

    wandb.init(
        project="face recognition",
        group=args.wandb_group,
        mode="offline" if args.wandb_offline else "online",
    )
    wandb.config.update(args)
    # writer = SummaryWriter(WORK_PATH) # writer for buffering intermedium results
    torch.backends.cudnn.benchmark = True

    # with open(os.path.join(DATA_ROOT, 'property'), 'r') as f:
    #     NUM_CLASS, h, w = [int(i) for i in f.read().split(',')]
    if args.data_mode == "casia100":
        NUM_CLASS = 100

    h, w = 112, 112
    assert h == INPUT_SIZE[0] and w == INPUT_SIZE[1]

    # create order list
    order_list = list(range(NUM_CLASS))
    # shuffle order list
    random.seed(SEED)
    random.shuffle(order_list)
    print("order_list", order_list)

    BACKBONE_DICT = {
        "VIT": ViT_face(
            loss_type=HEAD_NAME,
            GPU_ID=GPU_ID,
            num_class=200,
            image_size=112,
            patch_size=8,
            dim=512,
            depth=args.vit_depth,
            heads=8,
            mlp_dim=2048,
            dropout=0.1,
            emb_dropout=0.1,
            lora_rank=args.lora_rank,
            lora_pos=args.lora_pos,
        ),
        "VITs": ViTs_face(
            loss_type=HEAD_NAME,
            GPU_ID=GPU_ID,
            num_class=NUM_CLASS,
            image_size=112,
            patch_size=8,
            ac_patch_size=12,
            pad=4,
            dim=512,
            depth=args.vit_depth,
            heads=8,
            mlp_dim=2048,
            dropout=0.1,
            emb_dropout=0.1,
            lora_rank=args.lora_rank,
        ),
    }
    BACKBONE = BACKBONE_DICT[BACKBONE_NAME]
    print("=" * 60)
    print(BACKBONE)
    print("{} Backbone Generated".format(BACKBONE_NAME))
    print("=" * 60)

    # optionally resume from a checkpoint(得把分类头去掉)
    if BACKBONE_RESUME_ROOT and not args.retrain:
        print("=" * 60)
        print(BACKBONE_RESUME_ROOT)
        if os.path.isfile(BACKBONE_RESUME_ROOT):
            print("Loading Backbone Checkpoint '{}'".format(BACKBONE_RESUME_ROOT))
            checkpoint = torch.load(BACKBONE_RESUME_ROOT)

            # 移除 checkpoint 中的 loss.weight，避免形状不匹配的问题
            if 'loss.weight' in checkpoint:
                print("\033[33mRemoving 'loss.weight' from checkpoint to avoid size mismatch.\033[0m")
                del checkpoint['loss.weight']

            if 'loss.bias' in checkpoint:  # 如果有 bias 也要删除
                del checkpoint['loss.bias']

            # 现在加载剩余的权重
            missing_keys, unexpected_keys = BACKBONE.load_state_dict(checkpoint, strict=False)

            if len(missing_keys) > 0:
                print("Missing keys: {}".format(missing_keys))
                print("\n")
                for missing_key in missing_keys:
                    if "loss.weight" in missing_key:
                        print("\033[33mTransferring 'loss.weight' knowledge from 100 to 200 classes.\033[0m")
                        # 初始化当前模型的 loss.weight
                        new_loss_weight = BACKBONE.loss.weight.data

                        # 将前100个类别的权重复制到新的 loss.weight 中
                        pretrained_loss_weight = torch.load(BACKBONE_RESUME_ROOT).get('loss.weight', None)
                        if pretrained_loss_weight is not None:
                            new_loss_weight[:100, :] = pretrained_loss_weight

                        # 对其余的100个类使用随机初始化
                        torch.nn.init.xavier_uniform_(new_loss_weight[100:, :])

                        # 重新将修改后的权重赋值给当前模型
                        BACKBONE.loss.weight.data = new_loss_weight

                    elif "lora" not in missing_key:
                        print("\033[31mWrong resume.\033[0m")
                        exit()

            if len(unexpected_keys) > 0:
                print("Unexpected keys: {}".format(unexpected_keys))
                print("\n")
        else:
            print(
                "No Checkpoint Found at '{}' . Please Have a Check or Continue to Train from Scratch".format(
                    BACKBONE_RESUME_ROOT
                )
            )
        print("=" * 60)

    if args.one_stage:
        if args.lora_rank > 0:
            lora.mark_only_lora_as_trainable(BACKBONE)
            print("Use LoRA in Transformer FFN, loar_rank: ", args.lora_rank)
            # for n,p in BACKBONE.named_parameters():
            #     if 'loss.weight' in n: # open the gradient
            #         p.requires_grad = True
        else:
            print(
                "Do not use LoRA in Transformer FFN, train all parameters."
            )  # 19,157,504

    # get the number of learnable parameters
    learnable_parameters = count_trainable_parameters(BACKBONE)
    print("learnable_parameters", learnable_parameters)  # 19,157,504
    print("ratio of learnable_parameters", learnable_parameters / 19157504)
    wandb.log(
        {
            "learnable_parameters": learnable_parameters,
            "ratio of learnable_parameters": learnable_parameters / 19157504,
            "lora_rank": args.lora_rank,
        }
    )

    if MULTI_GPU:
        # multi-GPU setting
        BACKBONE = nn.DataParallel(BACKBONE, device_ids=GPU_ID)
        BACKBONE = BACKBONE.to(DEVICE)
    else:
        # single-GPU setting
        BACKBONE = BACKBONE.to(DEVICE)

    BACKBONE.train()  # set to training mode

    model_without_ddp = BACKBONE.module if MULTI_GPU else BACKBONE

    regularization_terms = {}  # store the regularization terms


    # 持续学习数据集划分
    order_learn_list = list(range(100))
    # shuffle order list
    random.seed(SEED)
    random.shuffle(order_learn_list)
    print("order_list", order_learn_list)
    data_transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )
    # 持续学习数据集路径
    train_learn_dataset = datasets.ImageFolder(
        root=os.path.join("/root/autodl-tmp/contious-diffusion/data/face", "train"), transform=data_transform
    )
    test_learn_dataset = datasets.ImageFolder(
        root=os.path.join("/root/autodl-tmp/contious-diffusion/data/face", "test"), transform=data_transform
    )
    learn_split_datasets_train = split_learn_dataset(
        dataset=train_learn_dataset,
        class_order_list=order_list,
    )
    learn_split_datasets_test = split_learn_dataset(
        dataset=test_learn_dataset,
        class_order_list=order_list,
    )


    for task_i in range(args.num_tasks):
        print("\n")
        print(
            "\033[34m=========================task:{}==============================\033[0m".format(
                task_i
            )
        )  # blue
        print("\n")

        # load pretrained model when task_i > 0
        if task_i > 0 and args.one_stage:
            print("load pretrained model in task {}".format(task_i - 1))
            BACKBONE.load_state_dict(
                torch.load(
                    os.path.join(
                        WORK_PATH,
                        "task-level",
                        "Backbone_task_{}.pth".format(task_i - 1),
                    )
                )
            )
            # reinitialize LoRA model 初始化
            reinitialize_lora_parameters(model_without_ddp)
        # split datasets
        # 1. calculate st1, en1, st2, en2
        st1 = 0
        en1 = args.num_of_first_cls - task_i * args.per_forget_cls
        st2 = en1
        en2 = en1 + args.per_forget_cls
        if task_i > 0:  # not the first task
            old_st = en2
            old_en = NUM_CLASS - 1
        # 2. split datasets
        train_dataset = datasets.ImageFolder(
            root=os.path.join(DATA_ROOT, "train"), transform=data_transform
        )
        test_dataset = datasets.ImageFolder(
            root=os.path.join(DATA_ROOT, "test"), transform=data_transform
        )
        remain_dataset_train, forget_dataset_train = split_dataset(
            dataset=train_dataset,
            class_order_list=order_list,
            split1_start=st1,
            split1_end=en1,
            split2_start=st2,
            split2_end=en2,
        )
        remain_dataset_test, forget_dataset_test = split_dataset(
            dataset=test_dataset,
            class_order_list=order_list,
            split1_start=st1,
            split1_end=en1,
            split2_start=st2,
            split2_end=en2,
        )

        # get sub datasets
        len_forget_dataset_train = len(forget_dataset_train)
        len_remain_dataset_train = len(remain_dataset_train)
        subset_size_forget = int(len_forget_dataset_train * 0.1)
        subset_size_remain = int(len_remain_dataset_train * 0.1)

        subset_indices_forget = torch.randperm(len_forget_dataset_train)[
            :subset_size_forget
        ]
        subset_indices_remain = torch.randperm(len_remain_dataset_train)[
            :subset_size_remain
        ]

        forget_dataset_train_sub = CustomSubset(
            forget_dataset_train, subset_indices_forget
        )
        remain_dataset_train_sub = CustomSubset(
            remain_dataset_train, subset_indices_remain
        )

        # 测试集缩小
        # len_forget_dataset_test = len(forget_dataset_test)
        # len_remain_dataset_test = len(remain_dataset_test)
        # subset_size_forget_test = int(len_forget_dataset_test * 0.2)
        # subset_size_remain_test = int(len_remain_dataset_test * 0.2)
        #
        # subset_indices_forget_test = torch.randperm(len_forget_dataset_test)[
        #                              :subset_size_forget_test
        #                              ]
        # subset_indices_remain_test = torch.randperm(len_remain_dataset_test)[
        #                              :subset_size_remain_test
        #                              ]
        #
        # forget_dataset_test_sub = CustomSubset(
        #     forget_dataset_test, subset_indices_forget_test
        # )
        # remain_dataset_test_sub = CustomSubset(
        #     remain_dataset_test, subset_indices_remain_test
        # )
        # 适用于连续交换 一次没用
        for i in range(task_i):
            remain_dataset_train_sub = MergedDataset(remain_dataset_train_sub, learn_split_datasets_train[i])
            remain_dataset_test_sub = MergedDataset(remain_dataset_test_sub, learn_split_datasets_train[i])
        # for i in range(task_i):
        #     remain_dataset_train_sub=merge_datasets(remain_dataset_train_sub, learn_split_datasets_train[i])
        #     remain_dataset_test=merge_datasets(remain_dataset_test, learn_split_datasets_test[i])

        if task_i == 0:
            # create importance dataset and dataloader
            len_importance_dataset_train = len(train_dataset)
            subset_size_importance = int(len_importance_dataset_train * 0.1)
            subset_indices_importance = torch.randperm(len_importance_dataset_train)[
                :subset_size_importance
            ]
            importance_dataset_train = Subset(train_dataset, subset_indices_importance)
            importance_dataloader_train = torch.utils.data.DataLoader(
                importance_dataset_train,
                batch_size=BATCH_SIZE,
                shuffle=True,
                num_workers=WORKERS,
                drop_last=False,
            )

        train_loader_forget = torch.utils.data.DataLoader(
            forget_dataset_train_sub,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=WORKERS,
            drop_last=False,
        )
        train_loader_remain = torch.utils.data.DataLoader(
            remain_dataset_train_sub,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=WORKERS,
            drop_last=False,
        )
        testloader_forget = torch.utils.data.DataLoader(
            forget_dataset_test,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=WORKERS,
            drop_last=False,
        )
        testloader_remain = torch.utils.data.DataLoader(
            remain_dataset_test,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=WORKERS,
            drop_last=False,
        )
        # 持续学习数据集loeader
        train_loader_learn = torch.utils.data.DataLoader(
            learn_split_datasets_train[task_i],
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=WORKERS,
            drop_last=False,
        )
        test_loader_learn = torch.utils.data.DataLoader(
            learn_split_datasets_test[task_i],
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=WORKERS,
            drop_last=False,
        )

        print("len(train_loader_forget)", len(train_loader_forget.dataset))
        print("len(train_loader_remain)", len(train_loader_remain.dataset))
        print("len(testloader_forget)", len(testloader_forget.dataset))
        print("len(testloader_remain)", len(testloader_remain.dataset))
        print("len(train_loader_learn)", len(train_loader_learn.dataset))
        print("len(test_loader_learn)", len(test_loader_learn.dataset))

        if task_i > 0:
            _, old_dataset_test = split_dataset(
                dataset=test_dataset,
                class_order_list=order_list,
                split1_start=0,
                split1_end=old_st,
                split2_start=old_st,
                split2_end=old_en,
            )
            # for i in range(0,task_i):
            #     old_dataset_test = merge_datasets(old_dataset_test, learn_split_datasets_test[i])
            testloader_old = torch.utils.data.DataLoader(
                old_dataset_test,
                batch_size=BATCH_SIZE,
                shuffle=False,
                num_workers=WORKERS,
                drop_last=False,
            )
            print("\n")
            print("len(testloader_old)", len(testloader_old))



        highest_H_mean = 0.0

        # embed()
        # ======= model & loss & optimizer =======#

        # LOSS = LossFaceCE(type=HEAD_NAME,dim=512,num_class=NUM_CLASS, GPU_ID=GPU_ID)
        LOSS = nn.CrossEntropyLoss()
        # embed()
        OPTIMIZER = create_optimizer(
            args, BACKBONE
        )  # create again to reinitialize optimizer
        print("=" * 60)
        print(OPTIMIZER)
        print("Optimizer Generated")
        print("=" * 60)
        lr_scheduler, _ = create_scheduler(
            args, OPTIMIZER
        )  # create again to reinitialize lr_scheduler

        # ======= train & validation & save checkpoint =======#

        batch = 0  # batch index

        if args.one_stage:
            losses_forget = AverageMeter()
            top1_forget = AverageMeter()
            losses_remain = AverageMeter()
            top1_remain = AverageMeter()
            losses_total = AverageMeter()
            losses_structure = AverageMeter()
            losses_learn = AverageMeter()
            top1_learn = AverageMeter()
        # 训练前评估
        if not args.LIRF:
            # eval before training
            print("Perform Evaluation on forget train set and remain train set...")
            forget_acc_train_before = eval_data(
                BACKBONE,
                train_loader_forget,
                DEVICE,
                "forget-train-{}".format(task_i),
                batch,
            )
            remain_acc_train_before = eval_data(
                BACKBONE,
                train_loader_remain,
                DEVICE,
                "remain-train-{}".format(task_i),
                batch,
            )
            print("forget_acc_train_before-{}".format(task_i), forget_acc_train_before)
            print("remain_acc_train_before-{}".format(task_i), remain_acc_train_before)
            print("\n")
            print("Perform Evaluation on forget test set and remain test set...")
            forget_acc_before = eval_data(
                BACKBONE, testloader_forget, DEVICE, "forget-{}".format(task_i), batch
            )
            remain_acc_before = eval_data(
                BACKBONE, testloader_remain, DEVICE, "remain-{}".format(task_i), batch
            )
            wandb.log(
                {
                    "forget_acc_before_{}".format(task_i): forget_acc_before,
                    "remain_acc_before_{}".format(task_i): remain_acc_before,
                }
            )
            if task_i > 0:
                # eval old test set
                old_acc_before = eval_data(
                    BACKBONE, testloader_old, DEVICE, "old-{}".format(task_i), batch
                )
                wandb.log({"old_acc_before_{}".format(task_i): old_acc_before})

        parms_without_ddp = {
            n: p for n, p in model_without_ddp.named_parameters() if p.requires_grad
        }  # for convenience
        # 持续遗忘
        if args.one_stage:
            cl_beta = args.cl_beta_list[task_i]
            BACKBONE.train()  # set to training mode
            print("start one stage forget remain training...")
            epoch = 0  # force it to be 0 to avoid affecting the epoch calculation of the next task
            for epoch in range(NUM_EPOCH):  # start training process
                lr_scheduler.step(epoch)
                (
                    batch,
                    highest_H_mean,
                    losses_forget,
                    losses_remain,
                    top1_forget,
                    top1_remain,
                    losses_total,
                    losses_structure,
                ) = train_one_epoch_combine(
                    model=BACKBONE,
                    dataloader_forget=train_loader_forget,
                    dataloader_remain=train_loader_remain,
                    dataloader_learn=train_loader_learn,
                    testloader_forget=testloader_forget,
                    testloader_remain=testloader_remain,
                    testloader_learn=test_loader_learn,
                    device=DEVICE,
                    criterion=LOSS,
                    optimizer=OPTIMIZER,
                    epoch=epoch,
                    batch=batch,
                    losses_forget=losses_forget,
                    top1_forget=top1_forget,
                    losses_remain=losses_remain,
                    top1_remain=top1_remain,
                    losses_total=losses_total,
                    losses_structure=losses_structure,
                    beta=cl_beta,
                    BND=args.BND,
                    forget_acc_before=forget_acc_before,
                    highest_H_mean=highest_H_mean,
                    cfg=cfg,
                    alpha=args.alpha,
                    task_i=task_i,
                )
                # print(batch)
                # calculate norm list

            norm_list = get_norm_of_lora(
                model_without_ddp, type="L2", group_num=args.vit_depth
            )
            wandb.log({"norm_list-{}".format(task_i): norm_list})





        # test for old classes after training task_i
        # save the model after one task training
        # 评估老数据集 保存盲从
        if args.one_stage:
            BACKBONE.eval()
            os.makedirs(os.path.join(WORK_PATH, "task-level"), exist_ok=True)
            torch.save(
                BACKBONE.state_dict(),
                os.path.join(
                    WORK_PATH, "task-level", "Backbone_task_{}.pth".format(task_i)
                ),
            )
            BACKBONE.train()
        if task_i > 0:
            if not args.LIRF:
                old_acc = eval_data(
                    BACKBONE, testloader_old, DEVICE, "old-{}".format(task_i), batch
                )
            wandb.log({"old_acc_after_{}".format(task_i): old_acc})


    wandb.run.name = (
        "remain-"
        + str(args.num_of_first_cls)
        + "-forget-"
        + str(args.per_forget_cls)
        + "-lora_rank-"
        + str(args.lora_rank)
        + "beta"
        + str(args.beta)
        + "lr"
        + str(args.lr)
    )
    if args.ewc:
        wandb.run.name = "ewc" + str(args.ewc_lambda) + wandb.run.name
    elif args.MAS:
        wandb.run.name = "mas" + str(args.mas_lambda) + wandb.run.name
    elif args.l2:
        wandb.run.name = "l2" + str(args.l2_lambda) + wandb.run.name
    elif args.retrain:
        wandb.run.name = "retrain-" + wandb.run.name
    elif args.LIRF:
        wandb.run.name = "LIRF" + wandb.run.name
    elif args.SCRUB:
        wandb.run.name = "SCRUB" + str(args.sgda_smoothing) + wandb.run.name
    elif args.Lwf:
        wandb.run.name = "Lwf" + wandb.run.name
    elif args.Der:
        wandb.run.name = (
            "DER" + str(args.DER_plus) + str(args.DER_lambda) + wandb.run.name
        )
    elif args.FDR:
        wandb.run.name = "FDR" + str(args.FDR_lambda) + wandb.run.name
