import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class LicensePlateOCRDataset(Dataset):
    """车牌OCR数据集类"""
    
    def __init__(self, labels_file, image_dir, transform=None, img_height=32, img_width=128):
        """
        初始化数据集
        
        Args:
            labels_file: 标签文件路径
            image_dir: 图像目录路径
            transform: 图像变换
            img_height: 图像高度
            img_width: 图像宽度
        """
        self.image_dir = image_dir
        self.transform = transform
        self.img_height = img_height
        self.img_width = img_width
        
        # 读取标签文件
        self.samples = []
        with open(labels_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split(',')
                    if len(parts) >= 2:
                        image_path = parts[0]
                        label = parts[1]
                        self.samples.append((image_path, label))
        
        # 构建字符到索引的映射（为CTC空白符预留索引0）
        self.characters = self._build_character_set()
        
        # 创建字符映射（索引0保留给CTC空白符）
        self.char_to_idx = {}
        self.idx_to_char = {}
        
        # 空白符索引为0
        self.char_to_idx['<blank>'] = 0
        self.idx_to_char[0] = '<blank>'
        
        # 字符索引从1开始
        for idx, char in enumerate(self.characters):
            self.char_to_idx[char] = idx + 1
            self.idx_to_char[idx + 1] = char
        
        print(f"字符集大小: {len(self.characters)}")
        print(f"字符集内容: {''.join(sorted(self.characters))}")
        print(f"总类别数（含空白符）: {len(self.characters) + 1}")
        
        print(f"数据集加载完成，共 {len(self.samples)} 个样本")
        print(f"字符集大小: {len(self.characters)}")
        print(f"字符集: {''.join(sorted(self.characters))}")
    
    def _build_character_set(self):
        """构建字符集"""
        characters = set()
        for _, label in self.samples:
            characters.update(label)
        return sorted(characters)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        
        # 修复路径处理：统一路径处理逻辑
        # 如果image_path是相对路径，直接拼接
        if os.path.isabs(image_path):
            full_image_path = image_path
        elif image_path.startswith('data/ocr_data/'):
            # 去掉前缀，构建相对路径
            relative_path = image_path.replace('data/ocr_data/', '')
            full_image_path = os.path.join('../../data/ocr_data', relative_path)
        else:
            # 直接使用image_dir和文件名
            filename = os.path.basename(image_path)
            full_image_path = os.path.join(self.image_dir, filename)
        
        # 读取图像
        try:
            if not os.path.exists(full_image_path):
                raise FileNotFoundError(f"图像文件不存在: {full_image_path}")
                
            image = cv2.imread(full_image_path)
            if image is None:
                raise ValueError(f"无法读取图像: {full_image_path}")
            
            # 转换为灰度图
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # 调整图像大小
            image = cv2.resize(image, (self.img_width, self.img_height))
            
            # 归一化
            image = image.astype(np.float32) / 255.0
            
            # 转换为PyTorch张量
            image = torch.from_numpy(image).unsqueeze(0)  # 添加通道维度
            
            # 转换标签为索引序列
            label_indices = []
            for char in label:
                if char in self.char_to_idx:
                    label_indices.append(self.char_to_idx[char])
                else:
                    print(f"警告: 字符 '{char}' 不在字符集中，跳过")
                    continue
            
            if not label_indices:
                print(f"警告: 标签 '{label}' 中没有有效字符，使用空白符")
                label_indices = [0]  # 使用空白符
                
            # 确保标签长度不超过最大长度（31个时间步）
            max_length = 31
            if len(label_indices) > max_length:
                label_indices = label_indices[:max_length]
            else:
                # 填充到固定长度
                label_indices.extend([0] * (max_length - len(label_indices)))
                
            label_tensor = torch.tensor(label_indices, dtype=torch.long)
            
            return image, label_tensor, label
            
        except Exception as e:
            print(f"处理图像 {full_image_path} 时出错: {e}")
            # 返回一个空样本
            empty_image = torch.zeros((1, self.img_height, self.img_width))
            empty_label = torch.tensor([0], dtype=torch.long)
            return empty_image, empty_label, ""

class DataPreprocessor:
    """数据预处理器"""
    
    def __init__(self, data_root="../../data/ocr_data"):
        self.data_root = data_root
        
    def prepare_datasets(self):
        """准备训练、验证和测试数据集"""
        
        # 定义图像变换
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomRotation(degrees=5),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
        ])
        
        # 训练集
        train_labels = os.path.join(self.data_root, "train/train_labels.txt")
        train_image_dir = os.path.join(self.data_root, "train/images")
        train_dataset = LicensePlateOCRDataset(train_labels, train_image_dir, transform)
        
        # 验证集
        val_labels = os.path.join(self.data_root, "val/val_labels.txt")
        val_image_dir = os.path.join(self.data_root, "val/images")
        val_dataset = LicensePlateOCRDataset(val_labels, val_image_dir)
        
        # 测试集
        test_labels = os.path.join(self.data_root, "test/test_labels.txt")
        test_image_dir = os.path.join(self.data_root, "test/images")
        test_dataset = LicensePlateOCRDataset(test_labels, test_image_dir)
        
        print(f"训练集大小: {len(train_dataset)}")
        print(f"验证集大小: {len(val_dataset)}")
        print(f"测试集大小: {len(test_dataset)}")
        
        return train_dataset, val_dataset, test_dataset
    
    def create_data_loaders(self, batch_size=32):
        """创建数据加载器"""
        train_dataset, val_dataset, test_dataset = self.prepare_datasets()
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        
        return train_loader, val_loader, test_loader

if __name__ == "__main__":
    # 测试数据预处理器
    preprocessor = DataPreprocessor()
    train_loader, val_loader, test_loader = preprocessor.create_data_loaders(batch_size=8)
    
    # 测试一个批次
    for images, labels, original_labels in train_loader:
        print(f"批次图像形状: {images.shape}")
        print(f"批次标签形状: {labels.shape}")
        print(f"原始标签: {original_labels[:5]}")
        break