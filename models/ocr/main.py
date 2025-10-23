#!/usr/bin/env python3
"""
车牌OCR识别系统主程序
基于CRNN模型的车牌号码识别
"""

import os
import sys
import argparse
from pathlib import Path

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_preprocessor import DataPreprocessor
from train_model import OCRTrainer
from inference import LicensePlateOCR

def preprocess_data():
    """数据预处理"""
    print("=== 数据预处理 ===")
    
    preprocessor = DataPreprocessor()
    
    try:
        # 准备数据集
        train_loader, val_loader, test_loader = preprocessor.create_data_loaders(batch_size=8)
        
        # 测试数据加载
        for images, labels, original_labels in train_loader:
            print(f"训练集批次图像形状: {images.shape}")
            print(f"训练集批次标签形状: {labels.shape}")
            print(f"样本标签示例: {original_labels[:5]}")
            break
            
        print("数据预处理完成!")
        return True
        
    except Exception as e:
        print(f"数据预处理失败: {e}")
        return False

def train_model(epochs=3, batch_size=16, learning_rate=0.001):
    """训练模型"""
    print("=== 模型训练 ===")
    
    trainer = OCRTrainer()
    
    try:
        recognizer = trainer.train(
            epochs=epochs, 
            batch_size=batch_size, 
            learning_rate=learning_rate
        )
        print("模型训练完成!")
        return True
        
    except Exception as e:
        print(f"模型训练失败: {e}")
        return False

def run_inference(image_path=None, image_dir=None, output_dir="../../results/ocr_results"):
    """运行推理"""
    print("=== 推理识别 ===")
    
    ocr = LicensePlateOCR()
    
    try:
        if image_path:
            # 单张图像识别
            result = ocr.recognize_single_image(image_path)
            if result:
                print(f"识别结果: {result}")
                return True
            else:
                print("识别失败")
                return False
                
        elif image_dir:
            # 批量识别
            results = ocr.recognize_batch_images(image_dir, output_dir)
            print(f"批量识别完成，共处理 {len(results)} 张图像")
            return True
            
        else:
            print("请指定图像路径或图像目录")
            return False
            
    except Exception as e:
        print(f"推理识别失败: {e}")
        return False

def test_accuracy():
    """测试模型准确率"""
    print("=== 模型测试 ===")
    
    ocr = LicensePlateOCR()
    
    try:
        test_labels = "../../data/ocr_data/test/test_labels.txt"
        test_image_dir = "../../data/ocr_data/test/images"
        
        accuracy = ocr.test_accuracy(test_labels, test_image_dir)
        
        if accuracy is not None:
            print("模型测试完成!")
            return True
        else:
            print("模型测试失败")
            return False
            
    except Exception as e:
        print(f"模型测试失败: {e}")
        return False

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='车牌OCR识别系统')
    parser.add_argument('--mode', type=str, required=True,
                       choices=['preprocess', 'train', 'inference', 'test', 'all'],
                       help='运行模式: preprocess(数据预处理), train(模型训练), inference(推理识别), test(模型测试), all(完整流程)')
    parser.add_argument('--image', type=str, help='单张图像路径（用于inference模式）')
    parser.add_argument('--image_dir', type=str, help='图像目录路径（用于inference模式）')
    parser.add_argument('--output_dir', type=str, default='../../results/ocr_results',
                       help='输出目录路径')
    parser.add_argument('--epochs', type=int, default=30, help='训练轮次')
    parser.add_argument('--batch_size', type=int, default=16, help='批量大小')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='学习率')
    
    args = parser.parse_args()
    
    print("=== 车牌OCR识别系统 ===")
    print(f"运行模式: {args.mode}")
    print(f"工作目录: {os.getcwd()}")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    success = True
    
    if args.mode == 'preprocess' or args.mode == 'all':
        success &= preprocess_data()
    
    if args.mode == 'train' or args.mode == 'all':
        success &= train_model(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate
        )
    
    if args.mode == 'inference' or (args.mode == 'all' and (args.image or args.image_dir)):
        success &= run_inference(
            image_path=args.image,
            image_dir=args.image_dir,
            output_dir=args.output_dir
        )
    
    if args.mode == 'test' or args.mode == 'all':
        success &= test_accuracy()
    
    if success:
        print("\n=== 任务完成 ===")
        print("所有操作执行成功!")
    else:
        print("\n=== 任务失败 ===")
        print("部分操作执行失败，请检查错误信息")
        sys.exit(1)

if __name__ == "__main__":
    main()