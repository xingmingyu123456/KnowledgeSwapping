import os
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from vit_pytorch_face import ViT_bird
import ml_collections
import torch
import random
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

#初始数据
# 初始模型
SEED=1337
# shuffle order list
torch.manual_seed(SEED)
order_list = list(range(100))
# shuffle order list
random.seed(SEED)
random.shuffle(order_list)
print("order_list", order_list)
DATA_ROOT=""
data_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # 归一化
        ]
    )
test_learn_dataset = datasets.ImageFolder(
        root=os.path.join("/root/autodl-tmp/bird_dataset_split", "test"), transform=data_transform
    )
learn_split_datasets_test = split_learn_dataset(
    dataset=test_learn_dataset,
    class_order_list=order_list,
)
dataset = learn_split_datasets_test[0]

config = get_b16_config()
BACKBONE = ViT_bird(config, 224 , zero_head=True, num_classes=200)
BACKBONE_RESUME_ROOT="/root/autodl-tmp/ViT-pytorch-main/output/imagenet100_small_checkpoint_2024-09-27-00-31-30.bin"
checkpoint = torch.load(BACKBONE_RESUME_ROOT)

BACKBONE.load_state_dict(checkpoint, strict=False)
pred,_=BACKBONE(img,1)



