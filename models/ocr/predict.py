#!/usr/bin/env python3
"""
OCR模型预测脚本
"""

import os
import sys
import argparse

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from inference import LicensePlateOCR

def predict_single_image(image_path, model_path="../plate_ocr_model.pt"):
    """预测单张图像"""
    print(f"=== 预测单张图像: {image_path} ===")
    
    # 检查图像文件是否存在
    if not os.path.exists(image_path):
        print(f"错误: 图像文件不存在: {image_path}")
        return None
    
    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        print(f"错误: 模型文件不存在: {model_path}")
        return None
    
    try:
        # 创建OCR识别器
        ocr = LicensePlateOCR(model_path)
        
        # 进行预测
        result = ocr.recognize_single_image(image_path)
        
        if result:
            print(f"识别结果: {result}")
            return result
        else:
            print("识别失败")
            return None
            
    except Exception as e:
        print(f"预测失败: {e}")
        return None

def predict_batch_images(image_dir, output_dir="../../results/ocr_results", model_path="../plate_ocr_model.pt"):
    """批量预测图像"""
    print(f"=== 批量预测图像: {image_dir} ===")
    
    # 检查图像目录是否存在
    if not os.path.exists(image_dir):
        print(f"错误: 图像目录不存在: {image_dir}")
        return None
    
    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        print(f"错误: 模型文件不存在: {model_path}")
        return None
    
    try:
        # 创建OCR识别器
        ocr = LicensePlateOCR(model_path)
        
        # 进行批量预测
        results = ocr.recognize_batch_images(image_dir, output_dir)
        
        print(f"批量识别完成，共处理 {len(results)} 张图像")
        
        # 显示结果摘要
        successful = [r for r in results if r['success']]
        failed = [r for r in results if not r['success']]
        
        print(f"成功识别: {len(successful)} 张")
        print(f"识别失败: {len(failed)} 张")
        
        if successful:
            print("\n成功识别的结果:")
            for result in successful[:10]:  # 显示前10个结果
                print(f"  {result['image_file']}: {result['plate_text']}")
        
        if failed:
            print(f"\n识别失败的图像:")
            for result in failed[:5]:  # 显示前5个失败结果
                print(f"  {result['image_file']}")
        
        return results
        
    except Exception as e:
        print(f"批量预测失败: {e}")
        return None

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='OCR模型预测')
    parser.add_argument('--mode', type=str, required=True,
                       choices=['single', 'batch'],
                       help='预测模式: single(单张图像), batch(批量预测)')
    parser.add_argument('--image', type=str, help='单张图像路径')
    parser.add_argument('--image_dir', type=str, help='图像目录路径')
    parser.add_argument('--output_dir', type=str, default='../../results/ocr_results',
                       help='输出目录路径')
    parser.add_argument('--model', type=str, default='../plate_ocr_model.pt',
                       help='模型文件路径')
    
    args = parser.parse_args()
    
    print("=== OCR模型预测 ===")
    print(f"预测模式: {args.mode}")
    print(f"模型文件: {args.model}")
    
    if args.mode == 'single':
        if not args.image:
            print("错误: 单张图像预测需要指定 --image 参数")
            return
        
        result = predict_single_image(args.image, args.model)
        if result:
            print(f"\n[SUCCESS] 预测成功: {result}")
        else:
            print(f"\n[FAILED] 预测失败")
    
    elif args.mode == 'batch':
        if not args.image_dir:
            print("错误: 批量预测需要指定 --image_dir 参数")
            return
        
        results = predict_batch_images(args.image_dir, args.output_dir, args.model)
        if results:
            print(f"\n[SUCCESS] 批量预测完成")
        else:
            print(f"\n[FAILED] 批量预测失败")

if __name__ == "__main__":
    main()
