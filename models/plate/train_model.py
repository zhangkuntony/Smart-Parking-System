"""
YOLOv8模型训练模块
"""

import os
import yaml
from ultralytics import YOLO
import torch

class YOLOv8Trainer:
    def __init__(self, data_yaml_path, model_size='n', pretrained=True):
        self.data_yaml_path = data_yaml_path
        self.model_size = model_size
        self.pretrained = pretrained
        self.model = None
        
    def setup_environment(self):
        """设置训练环境"""
        # 检查GPU可用性
        if torch.cuda.is_available():
            print(f"使用GPU: {torch.cuda.get_device_name(0)}")
            device = 'cuda'
        else:
            print("使用CPU进行训练")
            device = 'cpu'
        
        return device
    
    def load_model(self):
        """加载预训练模型"""
        model_name = f'yolov8{self.model_size}.pt'
        
        if self.pretrained:
            print(f"加载预训练模型: {model_name}")
            self.model = YOLO(model_name)
        else:
            print("从零开始训练模型")
            self.model = YOLO(f'yolov8{self.model_size}.yaml')
        
        return self.model
    
    def configure_training(self):
        """配置训练参数 - 使用最新的YOLOv8 API"""
        # 使用YOLOv8推荐的基本参数，避免已弃用的参数
        # 检查GPU可用性
        if torch.cuda.is_available():
            device = 0  # 使用GPU
            batch_size = 16
            epochs = 100  # GPU上训练更多轮次
            print("使用GPU进行训练")
        else:
            device = 'cpu'
            batch_size = 8  # CPU上减小批量大小
            epochs = 3  # CPU上训练较少轮次
            print("使用CPU进行训练，建议使用GPU以获得更好性能")
        
        training_config = {
            'data': self.data_yaml_path,
            'epochs': epochs,  # 根据设备调整训练轮次
            'imgsz': 640,
            'batch': batch_size,
            'workers': 4,
            'patience': 20,  # 增加早停耐心
            'save': True,
            'device': device,
            'optimizer': 'auto',
            'lr0': 0.01,  # 初始学习率
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'warmup_epochs': 3.0,
            # 简化数据增强参数
            'hsv_h': 0.015,
            'hsv_s': 0.7,
            'hsv_v': 0.4,
            'fliplr': 0.5,  # 只保留左右翻转
        }
        
        return training_config
    
    def train(self, output_dir='../../runs/detect/train'):
        """执行模型训练"""
        print("开始模型训练...")
        
        # 设置环境
        device = self.setup_environment()
        
        # 加载模型
        model = self.load_model()
        
        # 配置训练参数
        training_config = self.configure_training()
        
        # 开始训练
        print(f"训练配置: {training_config}")
        
        results = model.train(
            **training_config,
            project=output_dir,
            name=f'yolov8{self.model_size}_plate_detection',
            exist_ok=True
        )
        
        print("训练完成!")
        return results
    
    def save_best_model(self, source_path, destination_path):
        """保存最佳模型到指定位置"""
        best_model_path = os.path.join(source_path, 'weights', 'best.pt')
        
        if os.path.exists(best_model_path):
            os.makedirs(os.path.dirname(destination_path), exist_ok=True)
            
            # 复制模型文件
            import shutil
            shutil.copy2(best_model_path, destination_path)
            print(f"最佳模型已保存到: {destination_path}")
            return True
        else:
            print("未找到最佳模型文件")
            return False

def main():
    """主训练函数"""
    # 配置路径
    data_yaml_path = "../../datasets/plates/data.yaml"
    output_model_path = "../plate_detection_model.pt"
    
    # 检查数据文件是否存在
    if not os.path.exists(data_yaml_path):
        print("数据文件不存在，请先运行数据预处理")
        return
    
    # 创建训练器
    trainer = YOLOv8Trainer(data_yaml_path, model_size='n', pretrained=True)
    
    # 开始训练
    results = trainer.train()
    
    # 保存最佳模型
    run_dir = "../../runs/detect/train/yolov8n_plate_detection"
    trainer.save_best_model(run_dir, output_model_path)
    
    # 打印训练结果摘要
    if results:
        print("\n训练结果摘要:")
        print(f"- 最佳mAP50: {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
        print(f"- 最佳mAP50-95: {results.results_dict.get('metrics/mAP50-95(B)', 'N/A')}")
        print(f"- 训练损失: {results.results_dict.get('train/box_loss', 'N/A')}")

if __name__ == "__main__":
    main()