import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from ddpm import TransformedImgLblDataset
import torch
# clean





# 全局配置
BASE_PATH = os.path.expandvars("/mnt/yihang_backup/final_82")  # 数据集根路径
NUM_CLASSES = 13  # 数据集类别数
BACKGROUND_CLASS = 0  # 背景类索引
MEAN = [0.3361, 0.2113, 0.1803]  # 数据集均值
STD = [0.2766, 0.2158, 0.1825]   # 数据集标准差

# 获取类别权重
def get_weights(class_weights):
    """
    根据类别分布获取类别权重
    Args:
        class_weights (str): 权重类型 ('uniform' 或 'weighted')
    Returns:
        torch.Tensor: 类别权重向量
    """
    if class_weights == 'uniform':
        return torch.ones(NUM_CLASSES)
    elif class_weights == 'weighted':
        # 自定义权重，根据类别分布调整
        return torch.tensor([1,1,1,1,1,1,1,1,1,1,1,11,1])
    else:
        raise ValueError(f"Unknown class_weights: {class_weights}")

# Cholec 数据集定义
class CholecDataset(Dataset):
    def __init__(self, root, split="train"):
        """
        初始化 Cholec 数据集
        Args:
            root (str): 数据集根路径
            split (str): 数据集划分 ('train', 'val', 'test')
        """
        self.root = root
        self.split = split
        self.image_dir = os.path.join(root, split, "image")
        self.label_dir = os.path.join(root, split, "label")
        self.image_files = sorted(os.listdir(self.image_dir))
        self.label_files = sorted(os.listdir(self.label_dir))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        """
        获取单个样本
        Args:
            idx (int): 样本索引
        Returns:
            tuple: 图像 (PIL.Image) 和标签 (PIL.Image)
        """
        # 加载图像和标签
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        label_path = os.path.join(self.label_dir, self.label_files[idx])
        image = Image.open(image_path).convert("RGB")
        label = Image.open(label_path).convert("L")  # 标签为灰度图
        return image, label

# 数据集加载接口
def training_dataset(transforms_dict_train=None):
    """
    返回训练数据集
    Args:
        transforms_dict_train (dict): 数据增强配置字典
    Returns:
        TransformedImgLblDataset: 包装后的训练数据集
    """
    print(f"Loading Cholec training dataset from {BASE_PATH}")
    base_dataset = CholecDataset(root=BASE_PATH, split="train")
    dataset = TransformedImgLblDataset(
        source_dataset=base_dataset,
        transforms_dict=transforms_dict_train,
        num_classes=NUM_CLASSES,
        label_mapping_func=None  # 可根据需要添加标签映射函数
    )
    return dataset

def validation_dataset(transforms_dict_val=None, max_size=None):
    """
    返回验证数据集
    Args:
        transforms_dict_val (dict): 数据增强配置字典
        max_size (int, optional): 限制数据集大小
    Returns:
        TransformedImgLblDataset: 包装后的验证数据集
    """
    print(f"Loading Cholec validation dataset from {BASE_PATH}")
    base_dataset = CholecDataset(root=BASE_PATH, split="test")
    dataset = TransformedImgLblDataset(
        source_dataset=base_dataset,
        transforms_dict=transforms_dict_val,
        num_classes=NUM_CLASSES,
        label_mapping_func=None
    )

    if max_size:
        # 随机选择子集
        dataset, _ = torch.utils.data.random_split(
            dataset,
            [max_size, len(dataset) - max_size],
            generator=torch.Generator().manual_seed(1)  # 确保可重复性
        )
    # if max_size:
    #     dataset = torch.utils.data.Subset(dataset, indices=range(max_size))

    return dataset

def test_dataset(transforms_dict_test=None, max_size=None):
    return validation_dataset(transforms_dict_test, max_size)

# 工具函数
def get_num_classes():
    """
    返回类别数
    Returns:
        int: 类别数
    """
    return NUM_CLASSES

def get_ignore_class():
    """
    返回需要忽略的类别索引
    Returns:
        int: 忽略类别的索引
    """
    return None
def train_ids_to_class_names():
    """
    返回类别 ID 到类别名称的映射字典
    Returns:
        dict: 类别映射字典
    """
    return {i: f"class_{i}" for i in range(NUM_CLASSES)}
