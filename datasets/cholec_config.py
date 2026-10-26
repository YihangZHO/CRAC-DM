import numpy as np
import torch
from collections import namedtuple

# 定义类别属性
CholecClass = namedtuple('CholecClass', ['name', 'id', 'train_id', 'category', 'has_instances', 'ignore_in_eval', 'color'])


classes = [
    CholecClass('Black Background', 0, 0, 'background', False, False, (127, 127, 127)),  # BGR remains the same
    CholecClass('Abdominal Wall', 1, 1, 'tissue', False, False, (210, 140, 140)),
    CholecClass('Liver', 2, 2, 'tissue', False, False, (255, 114, 114)),
    CholecClass('Gastrointestinal Tract', 3, 3, 'tissue', False, False, (231, 70, 156)),
    CholecClass('Fat', 4, 4, 'tissue', False, False, (186, 183, 75)),
    CholecClass('Grasper', 5, 5, 'tool', False, False, (170, 255, 0)),
    CholecClass('Connective Tissue', 6, 6, 'tissue', False, False, (255, 85, 0)),
    CholecClass('Blood', 7, 7, 'tissue', False, False, (255, 0, 0)),
    CholecClass('Cystic Duct', 8, 8, 'tissue', False, False, (255, 255, 0)),
    CholecClass('L-hook Electrocautery', 9, 9, 'tool', False, False, (169, 255, 184)),
    CholecClass('Gallblader', 10, 10, 'tissue', False, False, (255, 160, 165)),
    CholecClass('Hepatic Vein', 11, 11, 'tissue', False, False, (0, 50, 128)),
    CholecClass('Liver Ligament', 12, 12, 'tissue', False, False, (111, 74, 0))
]

# 建立映射
TRAIN_ID_TO_COLOR = torch.tensor([cls.color for cls in classes], dtype=torch.uint8)
COLOR_TO_TRAIN_ID = {cls.color: cls.train_id for cls in classes}

# 标签编码函数
def encode_target(target):
    """
    将输入 RGB 标签图转化为 train_id 类别标签。
    Args:
        target (np.ndarray): RGB 标签图 (H, W, 3)
    Returns:
        np.ndarray: train_id 类别标签 (H, W)
    """
    target_encoded = np.zeros(target.shape[:2], dtype=np.uint8)
    for color, train_id in COLOR_TO_TRAIN_ID.items():
        mask = np.all(target == np.array(color, dtype=np.uint8), axis=-1)
        target_encoded[mask] = train_id
    return target_encoded

def decode_target_to_color(target: torch.Tensor) -> torch.Tensor:
    """
    将类别标签转换为 RGB 颜色图。

    Args:
        target (torch.Tensor): 类别标签，支持维度 (H, W)、(B, H, W) 或 (B, 1, H, W)。

    Returns:
        torch.Tensor: 对应的 RGB 颜色图，维度为 (3, H, W) 或 (B, 3, H, W)。
    """
    device = target.device  # 获取 target 的设备
    train_id_to_color_device = TRAIN_ID_TO_COLOR.to(device)  # 确保映射表也在同一设备

    if target.dim() == 4:  # (B, 1, H, W)
        target = target.squeeze(1)  # 转换为 (B, H, W)
    return train_id_to_color_device[target]  # 自动广播，返回 (B, H, W, 3)


def map_train_id_to_id(train_id_tensor):
    """
    Map train IDs to original IDs for the Cholec dataset.

    Args:
        train_id_tensor (torch.Tensor): A tensor of train IDs.

    Returns:
        torch.Tensor: A tensor with corresponding original IDs.
    """
    # Define the mapping between train IDs and original IDs
    train_id_to_id_map = {
        0: 0,   # Black Background
        1: 1,   # Abdominal Wall
        2: 2,   # Liver
        3: 3,   # Gastrointestinal Tract
        4: 4,   # Fat
        5: 5,   # Grasper
        6: 6,   # Connective Tissue
        7: 7,   # Blood
        8: 8,   # Cystic Duct
        9: 9,   # L-hook Electrocautery
        10: 10, # Gallbladder
        11: 11, # Hepatic Vein
        12: 12  # Liver Ligament
    }

    # Apply the mapping
    mapped_tensor = train_id_tensor.clone()
    for train_id, original_id in train_id_to_id_map.items():
        mapped_tensor[train_id_tensor == train_id] = original_id

    return mapped_tensor



# 测试函数
if __name__ == "__main__":
    # 示例标签图
    label_rgb = np.array([
        [[127, 127, 127], [140, 140, 210]],
        [[114, 114, 255], [156, 70, 231]]
    ], dtype=np.uint8)

    # 编码为 train_id 类别标签
    label_encoded = encode_target(label_rgb)
    print("Encoded Label:")
    print(label_encoded)

    # 解码回颜色
    label_tensor = torch.tensor(label_encoded, dtype=torch.uint8).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    label_decoded = decode_target_to_color(label_tensor)
    print("Decoded Label (Color):")
    print(label_decoded.shape)
