"""
数据预处理模块 - 将CCPD标注格式转换为YOLOv8格式
"""

import json
import os
import shutil
from sklearn.model_selection import train_test_split

class CCPDDataPreprocessor:
    def __init__(self, annotation_path, images_dir, output_dir):
        self.annotation_path = annotation_path
        self.images_dir = images_dir
        self.output_dir = output_dir
        
    def load_annotations(self):
        """加载CCPD标注数据"""
        with open(self.annotation_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def convert_to_yolo_format(self, annotation_data):
        """将CCPD标注转换为YOLO格式"""
        yolo_samples = []
        
        for sample in annotation_data['samples']:
            try:
                img_path = sample['media']['media_path']
                img_name = os.path.basename(img_path)
                
                # 获取图像尺寸（注意：您的数据中media_shape是[width, height]）
                img_w = sample['media']['media_shape'][0]  # 宽度
                img_h = sample['media']['media_shape'][1]  # 高度
                
                # 处理每个标注（可能有多个车牌）
                for annotation in sample['annotations']:
                    # 检查是否有bbox字段（您的数据格式）
                    if 'bbox' in annotation:
                        bbox = annotation['bbox']  # [x, y, w, h]
                        plate_number = annotation.get('content', '')
                        
                        # 转换为YOLO格式: class x_center y_center width height
                        x_center = (bbox[0] + bbox[2]/2) / img_w
                        y_center = (bbox[1] + bbox[3]/2) / img_h
                        width = bbox[2] / img_w
                        height = bbox[3] / img_h
                        
                        yolo_samples.append({
                            'image_name': img_name,
                            'image_path': img_path,
                            'yolo_bbox': [0, x_center, y_center, width, height],  # class=0 表示车牌
                            'plate_number': plate_number,
                            'image_size': (img_w, img_h)
                        })
            except Exception as e:
                print(f"处理样本时出错: {sample.get('media', {}).get('media_path', 'unknown')}, 错误: {e}")
                continue
        
        return yolo_samples
    
    def create_dataset_structure(self):
        """创建YOLO数据集目录结构"""
        dirs = [
            'images/train',
            'images/val',
            'labels/train', 
            'labels/val'
        ]
        
        for dir_path in dirs:
            os.makedirs(os.path.join(self.output_dir, dir_path), exist_ok=True)
    
    def split_dataset(self, yolo_samples, test_size=0.2, random_state=42):
        """分割训练集和验证集"""
        train_samples, val_samples = train_test_split(
            yolo_samples, test_size=test_size, random_state=random_state
        )
        return train_samples, val_samples
    
    def copy_images_and_create_labels(self, samples, split_type):
        """复制图像并创建标签文件"""
        images_dir = os.path.join(self.output_dir, 'images', split_type)
        labels_dir = os.path.join(self.output_dir, 'labels', split_type)
        
        # 按图像分组样本（一张图可能有多个车牌）
        image_groups = {}
        for sample in samples:
            img_name = sample['image_name']
            if img_name not in image_groups:
                image_groups[img_name] = []
            image_groups[img_name].append(sample)
        
        for img_name, img_samples in image_groups.items():
            # 复制图像 - 处理可能的子目录结构
            # 尝试多种可能的图像路径
            src_img_paths = [
                os.path.join(self.images_dir, img_name),  # 直接文件名
                os.path.join(self.images_dir, img_samples[0]['image_path']),  # 完整路径
            ]
            
            src_img_path = None
            for path in src_img_paths:
                if os.path.exists(path):
                    src_img_path = path
                    break
            
            if src_img_path is None:
                print(f"警告: 找不到图像文件: {img_name}")
                continue
                
            dst_img_path = os.path.join(images_dir, img_name)
            
            if os.path.exists(src_img_path):
                shutil.copy2(src_img_path, dst_img_path)
            
            # 创建标签文件
            label_file = os.path.splitext(img_name)[0] + '.txt'
            label_path = os.path.join(labels_dir, label_file)
            
            with open(label_path, 'w', encoding='utf-8') as f:
                for sample in img_samples:
                    bbox_str = ' '.join(str(x) for x in sample['yolo_bbox'])
                    f.write(bbox_str + '\n')
    
    def create_data_yaml(self):
        """创建data.yaml配置文件"""
        data_yaml = {
            'path': self.output_dir,
            'train': 'images/train',
            'val': 'images/val',
            'nc': 1,  # 类别数量
            'names': ['license_plate']  # 类别名称
        }
        
        yaml_path = os.path.join(self.output_dir, 'data.yaml')
        with open(yaml_path, 'w', encoding='utf-8') as f:
            f.write(f"path: {data_yaml['path']}\n")
            f.write(f"train: {data_yaml['train']}\n")
            f.write(f"val: {data_yaml['val']}\n")
            f.write(f"nc: {data_yaml['nc']}\n")
            f.write(f"names: {data_yaml['names']}\n")
    
    def process(self):
        """执行完整的数据预处理流程"""
        print("开始数据预处理...")
        
        # 加载标注数据
        annotation_data = self.load_annotations()
        print(f"加载了 {len(annotation_data['samples'])} 个样本")
        
        # 转换为YOLO格式
        yolo_samples = self.convert_to_yolo_format(annotation_data)
        print(f"成功转换 {len(yolo_samples)} 个YOLO样本")
        
        # 创建目录结构
        self.create_dataset_structure()
        
        # 分割数据集
        train_samples, val_samples = self.split_dataset(yolo_samples)
        print(f"训练集: {len(train_samples)} 个样本")
        print(f"验证集: {len(val_samples)} 个样本")
        
        # 处理训练集和验证集
        self.copy_images_and_create_labels(train_samples, 'train')
        self.copy_images_and_create_labels(val_samples, 'val')
        
        # 创建配置文件
        self.create_data_yaml()
        
        print("数据预处理完成!")
        return len(train_samples), len(val_samples)

if __name__ == "__main__":
    # 配置路径
    annotation_path = "../../data/labels/ccpd_base_samples.json"
    images_dir = "../../data/images"
    output_dir = "../../datasets/plates"
    
    preprocessor = CCPDDataPreprocessor(annotation_path, images_dir, output_dir)
    preprocessor.process()