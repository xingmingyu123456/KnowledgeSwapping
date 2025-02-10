import os, argparse, sklearn
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import wandb
import sys
import random
import numpy as np
import numpy as np
np.bool = bool  # 临时兼容
import os
import ml_collections
from datetime import datetime
from torch.utils.data import Dataset
from PIL import Image
# sys.path.append('/home/xmy/code/classify-bird')
current_dir = os.path.dirname(os.path.abspath(__file__))  # 获取当前脚本的绝对路径
sys.path.append(os.path.join(current_dir, '..'))
from image_iter import CLDatasetWrapper, CustomSubset,MergedDataset
from config import get_config
from util.utils import (
    AverageMeter,
)
import torch
import  numpy
from util.utils import (
    split_dataset,
    count_trainable_parameters,
    reinitialize_lora_parameters,
)
from util.args import get_args
from vit_pytorch_face import ViT_bird
# from IPython import embed
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
import loralib as lora
from engine_cl import train_one_epoch, eval_data, train_one_epoch_regularzation,train_one_epoch_learn
from torch.utils.data import Subset
from util.cal_norm import get_norm_of_lora
from torch.utils.data import DataLoader
from customlogger import CustomLogger
import math
from test_selective import get_norm_of_lora

def split_learn_dataset(dataset, class_order_list,data_transform):
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
        split_dataset = datasets.ImageFolder(root=dataset.root, transform=data_transform)
        split_dataset.samples = split_samples
        split_dataset.targets = [label for _, label in split_samples]
        split_dataset.classes = [dataset.classes[idx] for idx in split_class_indices]
        split_dataset.class_to_idx = {
            class_name: j + 100 for j, class_name in enumerate(split_dataset.classes)
        }

        # 将分割数据集添加到列表中
        split_datasets.append(split_dataset)

    return split_datasets

print('hello')
def get_b16_config():
    """Returns the ViT-B/16 configuration."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (16, 16)})
    config.hidden_size = 768
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 3072
    config.transformer.num_heads = 12
    config.transformer.num_layers = 12
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.classifier = 'token'
    config.representation_size = None
    return config

class CustomImageDataset_remain(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        # self.classes = sorted(os.listdir(root_dir), key=lambda x: int(x[1:]))  # 假设文件夹名是数字
        self.classes = sorted(os.listdir(root_dir))  # 假设每类图片在各自文件夹下
        # self.classes = os.listdir(root_dir)  # 假设每类图片在各自文件夹下
        self.img_paths = []
        self.labels = []

        for sub in self.classes:
            label = int(sub[:-10])
            class_dir = os.path.join(root_dir, sub)
            for img_name in os.listdir(class_dir):
                self.img_paths.append(os.path.join(class_dir, img_name))
                self.labels.append(label)

        # for label in range(len(self.classes)):
        #     class_dir = os.path.join(root_dir, self.classes[label])
        #     for img_name in os.listdir(class_dir):
        #         self.img_paths.append(os.path.join(class_dir, img_name))
        #         self.labels.append(label)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        #
        return img, label

class CustomImageDataset_foregt(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        # self.classes = sorted(os.listdir(root_dir), key=lambda x: int(x[1:]))  # 假设文件夹名是数字
        self.classes = sorted(os.listdir(root_dir))  # 假设每类图片在各自文件夹下
        # self.classes = os.listdir(root_dir)  # 假设每类图片在各自文件夹下
        self.img_paths = []
        self.labels = []

        for sub in self.classes:
            label = int(sub[:-10])
            class_dir = os.path.join(root_dir, sub)
            for img_name in os.listdir(class_dir):
                self.img_paths.append(os.path.join(class_dir, img_name))
                self.labels.append(label)
        # for label in range(len(self.classes)):
        #     class_dir = os.path.join(root_dir, self.classes[label])
        #     for img_name in os.listdir(class_dir):
        #         self.img_paths.append(os.path.join(class_dir, img_name))
        #         self.labels.append(label+80)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        #
        return img, label
class CustomImageDataset_learn(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        # self.classes = sorted(os.listdir(root_dir), key=lambda x: int(x[1:]))  # 假设文件夹名是数字
        self.classes = sorted(os.listdir(root_dir))  # 假设每类图片在各自文件夹下
        self.img_paths = []
        self.labels = []

        for label in range(len(self.classes)):
            class_dir = os.path.join(root_dir, self.classes[label])
            for img_name in os.listdir(class_dir):
                self.img_paths.append(os.path.join(class_dir, img_name))
                self.labels.append(label+100)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        #
        return img, label

if __name__ == "__main__":
    logger = CustomLogger()
    logger.info('This is an info message')
    args = get_args()
    # ======= hyperparameters & data loaders =======#
    cfg = get_config(args)

    SEED = cfg["SEED"]  # random seed for reproduce results
    torch.manual_seed(SEED)
    DATA_ROOT = cfg[
        "DATA_ROOT"
    ]  # the parent root where your train/val/test data are stored
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
    logger.info("Overall Configurations:")
    logger.info(cfg)
    with open(os.path.join(WORK_PATH, "config.txt"), "w") as f:
        f.write(str(cfg))
    print("=" * 60)
    # 指定系统证书目录
    os.environ['SSL_CERT_DIR'] = '/etc/ssl/certs'
    # 证书捆绑文件路径
    os.environ['REQUESTS_CA_BUNDLE'] = '/etc/ssl/certs/ca-certificates.crt'
    os.environ["WANDB_API_KEY"] = '33342c2d14bf7985e96d5f496f3c095af0aca6f7'  # 将引号内的*替换成自己在wandb上的key
    os.environ["WANDB_MODE"] = "offline"
    os.environ['WANDB_VERIFY_SSL'] = 'false'

    # 之前下面有个不离线的选项给覆盖了！！
    wandb.init(
        settings=wandb.Settings(init_timeout=120),
        project="face recognition",
        # group=args.wandb_group,
        entity="15033203761-hefei-university-of-technology-org",
    )
    wandb.config.update(args)
    # writer = SummaryWriter(WORK_PATH) # writer for buffering intermedium results
    torch.backends.cudnn.benchmark = True

    # with open(os.path.join(DATA_ROOT, 'property'), 'r') as f:
    #     NUM_CLASS, h, w = [int(i) for i in f.read().split(',')]
    if args.data_mode == "casia100":
        NUM_CLASS = 100

    h, w = 224, 224
    assert h == INPUT_SIZE[0] and w == INPUT_SIZE[1]

    # create order list
    order_list = list(range(NUM_CLASS))
    # shuffle order list
    random.seed(SEED)
    random.shuffle(order_list)
    print("order_list", order_list)


    config = get_b16_config()
    BACKBONE = ViT_bird(config, 224 , zero_head=True, num_classes=200)
    # BACKBONE.load_state_dict(torch.load(BACKBONE_RESUME_ROOT))
    # BACKBONE.load_from(np.load(BACKBONE_RESUME_ROOT))
    # optionally resume from a checkpoint(得把分类头去掉)
    if BACKBONE_RESUME_ROOT and not args.retrain:
        print("=" * 60)
        print(BACKBONE_RESUME_ROOT)
        if os.path.isfile(BACKBONE_RESUME_ROOT):
            print("Loading Backbone Checkpoint '{}'".format(BACKBONE_RESUME_ROOT))
            checkpoint = torch.load(BACKBONE_RESUME_ROOT)

            # 移除 checkpoint 中的 loss.weight，避免形状不匹配的问题
            if 'head.weight' in checkpoint:
                print("\033[33mRemoving 'head.weight' from checkpoint to avoid size mismatch.\033[0m")
                del checkpoint['head.weight']

            if 'head.bias' in checkpoint:  # 如果有 bias 也要删除
                del checkpoint['head.bias']

            # 现在加载剩余的权重
            missing_keys, unexpected_keys = BACKBONE.load_state_dict(checkpoint, strict=False)

            if len(missing_keys) > 0:
                print("Missing keys: {}".format(missing_keys))
                print("\n")
                for missing_key in missing_keys:
                    if "head.weight" in missing_key:
                        print("\033[33mTransferring 'head.weight' knowledge from 100 to 200 classes.\033[0m")
                        # 初始化当前模型的 head.weight
                        new_loss_weight = BACKBONE.head.weight.data

                        # 将前100个类别的权重复制到新的 head.weight 中
                        pretrained_loss_weight = torch.load(BACKBONE_RESUME_ROOT).get('head.weight', None)
                        if pretrained_loss_weight is not None:
                            new_loss_weight[:100, :] = pretrained_loss_weight

                        # 对其余的100个类使用随机初始化
                        torch.nn.init.xavier_uniform_(new_loss_weight[100:, :])

                        # 重新将修改后的权重赋值给当前模型
                        BACKBONE.head.weight.data = new_loss_weight
                    elif "head.bias" in missing_key:
                        print("\033[33mTransferring 'head.bias' knowledge from 100 to 200 classes.\033[0m")
                        # 初始化当前模型的 head.bias
                        new_bias = BACKBONE.head.bias.data

                        # 将前100个类别的偏置复制到新的 head.bias 中
                        pretrained_bias = torch.load(BACKBONE_RESUME_ROOT).get('head.bias', None)
                        if pretrained_bias is not None:
                            new_bias[:100] = pretrained_bias

                        # 对其余的100个类使用零初始化
                        new_bias[100:] = 0.0  # 零初始化

                        # 重新将修改后的偏置赋值给当前模型
                        BACKBONE.head.bias.data = new_bias
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

    # 仅仅让lora刻意训练

    if args.lora_rank > 0:
        lora.mark_only_lora_as_trainable(BACKBONE)
        reinitialize_lora_parameters(BACKBONE)
        norm_list = get_norm_of_lora(BACKBONE, type="L2", group_num=12,group_type="block",group_pos="FFN")
        print(norm_list)
        print("Use LoRA in Transformer FFN, loar_rank: ", args.lora_rank)
    else:
        print(
            "Do not use LoRA in Transformer FFN, train all parameters."
        )  # 19,157,504

    # get the number of learnable parameters
    learnable_parameters = count_trainable_parameters(BACKBONE)
    print("learnable_parameters", learnable_parameters)  # 19,157,504
    print("ratio of learnable_parameters", learnable_parameters / 19157504)
    logger.info("learnable_parameters:{}".format( learnable_parameters))  # 19,157,504
    logger.info("ratio of learnable_parameters {}".format(learnable_parameters / 19157504))
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

    print("\n")
    print(
        "\033[34m=========================知识交换开始了==============================\033[0m"
    )
    print("\n")
    logger.info(
        "\033[34m=========================知识交换开始了==============================\033[0m"
    )

    # 持续学习数据集划分
    data_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # 归一化
        ]
    )
    # 持续学习数据集路径
    train_learn_dataset = CustomImageDataset_learn(
        root_dir=os.path.join(args.learnpath, "train"), transform=data_transform
    )
    test_learn_dataset = CustomImageDataset_learn(
        root_dir=os.path.join(args.learnpath, "test"), transform=data_transform
    )

    remain_train_dataset = CustomImageDataset_remain(root_dir=os.path.join(args.retainpath, "train"),transform=data_transform)
    remain_test_dataset = CustomImageDataset_remain(root_dir=os.path.join(args.retainpath, "test"),transform=data_transform)
    forget_train_dataset = CustomImageDataset_foregt(root_dir=os.path.join(args.forgetpath, "train"), transform=data_transform)
    forget_test_dataset = CustomImageDataset_foregt(root_dir=os.path.join(args.forgetpath, "test"), transform=data_transform)

    len_forget_dataset_train = len(forget_train_dataset)
    len_remain_dataset_train = len(remain_train_dataset)
    subset_size_forget = int(len_forget_dataset_train * 0.1)
    subset_size_remain = int(len_remain_dataset_train * 0.1)

    subset_indices_forget = torch.randperm(len_forget_dataset_train)[
        :subset_size_forget
    ]
    subset_indices_remain = torch.randperm(len_remain_dataset_train)[
        :subset_size_remain
    ]

    forget_dataset_train_sub = CustomSubset(
        forget_train_dataset, subset_indices_forget
    )
    remain_dataset_train_sub = CustomSubset(
        remain_train_dataset, subset_indices_remain
    )


    # 加载dataloader
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
        forget_test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=WORKERS,
        drop_last=False,
    )
    testloader_remain = torch.utils.data.DataLoader(
        remain_test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=WORKERS,
        drop_last=False,
    )
    # 持续学习数据集loeader
    train_loader_learn = torch.utils.data.DataLoader(
        train_learn_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=WORKERS,
        drop_last=False,
    )
    test_loader_learn = torch.utils.data.DataLoader(
        test_learn_dataset,
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


    losses_forget = AverageMeter()
    top1_forget = AverageMeter()
    losses_remain = AverageMeter()
    top1_remain = AverageMeter()
    losses_total = AverageMeter()
    losses_structure = AverageMeter()
    losses_learn = AverageMeter()
    top1_learn = AverageMeter()
    # 训练前评估

    # eval before training
    print("Perform Evaluation on forget train set and remain train set...")
    logger.info("Perform Evaluation on forget train set and remain train set...")
    forget_acc_train_before = eval_data(
        BACKBONE,
        train_loader_forget,
        DEVICE,
        "forget-train-{}".format(0),
        batch,
        logger,
    )
    remain_acc_train_before = eval_data(
        BACKBONE,
        train_loader_remain,
        DEVICE,
        "remain-train-{}".format(0),
        batch,
        logger,
    )
    learn_acc_train_before = eval_data(
        BACKBONE,
        train_loader_learn,
        DEVICE,
        "learn-train-{}".format(0),
        batch,
        logger,
    )
    print("forget_acc_train_before-{}".format(0), forget_acc_train_before)
    print("remain_acc_train_before-{}".format(0), remain_acc_train_before)
    print("learn_acc_train_before-{}".format(0), learn_acc_train_before)
    print("\n")
    print("Perform Evaluation on forget test set and remain test set...")
    logger.info("forget_acc_train_before-{}".format( forget_acc_train_before))
    logger.info("remain_acc_train_before-{}".format( remain_acc_train_before))
    logger.info("learn_acc_train_before-{}".format( learn_acc_train_before))
    logger.info("\n")
    logger.info("Perform Evaluation on forget test set and remain test set...")
    forget_acc_before = eval_data(
        BACKBONE, testloader_forget, DEVICE, "forget-{}".format(0), batch,logger,
    )
    remain_acc_before = eval_data(
        BACKBONE, testloader_remain, DEVICE, "remain-{}".format(0), batch,logger,
    )
    learn_acc_before = eval_data(
        BACKBONE, test_loader_learn, DEVICE, "learn-{}".format(0), batch,logger,
    )
    wandb.log(
        {
            "forget_acc_before_{}".format(0): forget_acc_before,
            "remain_acc_before_{}".format(0): remain_acc_before,
            "learn_acc_before_{}".format(0): learn_acc_before,
        }
    )


    parms_without_ddp = {
        n: p for n, p in model_without_ddp.named_parameters() if p.requires_grad
    }  # for convenience
    # 持续遗忘

    beta = args.beta
    BACKBONE.train()  # set to training mode
    print("遗忘开始了")
    logger.info("遗忘开始了")
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
        ) = train_one_epoch(
            model=BACKBONE,
            dataloader_forget=train_loader_forget,
            dataloader_remain=train_loader_remain,
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
            beta=beta,
            BND=args.BND,
            forget_acc_before=forget_acc_before,
            highest_H_mean=highest_H_mean,
            cfg=cfg,
            alpha=args.alpha,
            task_i=0,
            logger = logger,
        )
        # print(batch)
        # calculate norm list
        if (epoch+1)%(NUM_EPOCH) == 0:
            torch.save(
                BACKBONE.state_dict(),
                os.path.join(
                    WORK_PATH,f"forget_{epoch+1}.pth"
                ),
            )
            print("权重已保存")

    norm_list = get_norm_of_lora(
        model_without_ddp, type="L2", group_num=args.vit_depth
    )
    wandb.log({"norm_list-{}".format(0): norm_list})




    # 持续学习

    beta = args.beta
    BACKBONE.train()  # set to training mode
    print("持续学习开始了...")
    logger.info("持续学习开始了...")
    epoch_learn = 0  # force it to be 0 to avoid affecting the epoch calculation of the next task
    for epoch_learn in range(NUM_EPOCH):  # start training process
        lr_scheduler.step(epoch_learn)
        (
            batch,
            highest_H_mean,
            losses_forget,
            losses_remain,
            top1_forget,
            top1_remain,
            losses_total,
            losses_structure,
        ) = train_one_epoch_learn(
            model=BACKBONE,
            dataloader_learn=train_loader_learn,
            dataloader_remain=train_loader_remain,
            testloader_learn=test_loader_learn,
            testloader_remain=testloader_remain,
            testloader_forget=testloader_forget,
            device=DEVICE,
            criterion=LOSS,
            optimizer=OPTIMIZER,
            epoch=epoch_learn,
            batch=batch,
            losses_learn=losses_learn,
            top1_learn=top1_learn,
            losses_remain=losses_remain,
            top1_remain=top1_remain,
            losses_total=losses_total,
            losses_structure=losses_structure,
            beta=beta,
            BND=args.BND,
            learn_acc_before=forget_acc_before,
            highest_H_mean=highest_H_mean,
            cfg=cfg,
            alpha=args.alpha,
            task_i=0,
            logger=logger,
        )
        # print(batch)
        # calculate norm list
        if (epoch_learn+1) % NUM_EPOCH == 0:
            torch.save(
                BACKBONE.state_dict(),
                os.path.join(
                    WORK_PATH, f"learn_{epoch_learn+1}.pth"
                ),
            )
            print("权重已保存")

    norm_list = get_norm_of_lora(
        model_without_ddp, type="L2", group_num=args.vit_depth
    )
    wandb.log({"norm_list-{}".format(0): norm_list})


    # test for old classes after training task_i
    # save the model after one task training
    # 评估老数据集 保存权重
    # bug 这样保存最后会更新参数 导致测试结果变差
    # if args.one_stage:
    #     BACKBONE.eval()
    #     current_time=datetime.now()
    #     time_str = current_time.strftime("%Y-%m-%d-%H-%M-%S")
    #     os.makedirs(os.path.join(WORK_PATH, "task-level"), exist_ok=True)
    #     torch.save(
    #         BACKBONE.state_dict(),
    #         f"./forget_check/forget_final.pth"
    #         # os.path.join(
    #         #     WORK_PATH, "task-level", "backbone{}.pth".format(time_str)
    #         # ),
    #     )
    #     # BACKBONE.train()



    # wandb 文件夹命名
    wandb.run.name = (
       "lora_rank-"
        + str(args.lora_rank)
        + "beta"
        + str(args.beta)
        + "lr"
        + str(args.lr)
    )

