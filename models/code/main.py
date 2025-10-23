"""
车牌识别系统主程序
"""

import os
import argparse
from data_preprocessor import CCPDDataPreprocessor
from train_model import YOLOv8Trainer
from inference import LicensePlateDetector

def setup_directories():
    """创建必要的目录结构"""
    directories = [
        "../../datasets/plates",
        "../../runs/detect",
        "../../results"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print("目录结构创建完成")

def preprocess_data():
    """数据预处理"""
    print("=== 数据预处理阶段 ===")
    
    annotation_path = "../../data/labels/ccpd_base_samples.json"
    images_dir = "../../data/images"
    output_dir = "../../datasets/plates"
    
    preprocessor = CCPDDataPreprocessor(annotation_path, images_dir, output_dir)
    train_count, val_count = preprocessor.process()
    
    print(f"数据预处理完成: 训练集 {train_count} 样本, 验证集 {val_count} 样本")
    return True

def train_model():
    """模型训练"""
    print("\n=== 模型训练阶段 ===")
    
    data_yaml_path = "../../datasets/plates/data.yaml"
    output_model_path = "../../plate_detection_model.pt"
    
    # 检查数据文件是否存在
    if not os.path.exists(data_yaml_path):
        print("数据文件不存在，请先运行数据预处理")
        return False
    
    # 创建训练器
    trainer = YOLOv8Trainer(data_yaml_path, model_size='n', pretrained=True)
    
    # 开始训练
    results = trainer.train()
    print("训练结束！")
    print("训练结果：", results)
    
    # 保存最佳模型
    run_dir = "../../runs/detect/train/yolov8n_plate_detection"
    trainer.save_best_model(run_dir, output_model_path)
    
    print("模型训练完成!")
    return True

def run_inference():
    """运行推理"""
    print("\n=== 推理测试阶段 ===")
    
    model_path = "../../plate_detection_model.pt"
    test_image_path = "../../data/images/"
    output_dir = "../../results"
    
    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        print("模型文件不存在，请先训练模型")
        return False
    
    # 创建检测器
    detector = LicensePlateDetector(model_path)
    
    # 批量处理测试图像
    summary = detector.process_batch_images(test_image_path, output_dir)
    
    # 打印结果摘要
    print("\n检测结果摘要:")
    print(f"总图像数: {summary['total_images']}")
    print(f"成功处理: {summary['successful_processing']}")
    print(f"检测到车牌数: {summary['total_plates_detected']}")
    
    return True

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='车牌识别系统')
    parser.add_argument('--mode', choices=['all', 'preprocess', 'train', 'inference'], 
                       default='all', help='运行模式')
    parser.add_argument('--image', type=str, help='单张图像推理路径')
    
    args = parser.parse_args()
    
    # 设置目录结构
    setup_directories()
    
    if args.mode == 'all' or args.mode == 'preprocess':
        if not preprocess_data():
            return
    
    if args.mode == 'all' or args.mode == 'train':
        if not train_model():
            return
    
    if args.mode == 'all' or args.mode == 'inference':
        if args.image:
            # 单张图像推理
            detector = LicensePlateDetector("../../plate_detection_model.pt")
            result = detector.process_single_image(args.image, "../../results")
            print(f"检测结果: {result}")
        else:
            # 批量推理
            if not run_inference():
                return
    
    print("\n=== 车牌识别系统运行完成 ===")

if __name__ == "__main__":
    main()