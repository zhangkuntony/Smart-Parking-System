#!/usr/bin/env python3
"""
车牌识别系统 - 集成车牌检测和OCR识别
功能：
1. 检测汽车图像中的车牌位置
2. 识别车牌号码
3. 输出识别结果和可视化图像
"""

import os
import cv2
from pathlib import Path
import json
from datetime import datetime

# 导入车牌检测模块
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from plate.inference import LicensePlateDetector
from ocr.inference import LicensePlateOCR

class LicensePlateRecognitionSystem:
    """车牌识别系统"""
    
    def __init__(self, 
                 detection_model_path="plate_detection_model.pt",
                 ocr_model_path="plate_ocr_model.pt",
                 conf_threshold=0.5):
        """
        初始化车牌识别系统
        
        Args:
            detection_model_path: 车牌检测模型路径
            ocr_model_path: OCR识别模型路径
            conf_threshold: 检测置信度阈值
        """
        # 检查模型文件是否存在
        if not os.path.exists(detection_model_path):
            raise FileNotFoundError(f"车牌检测模型文件不存在: {detection_model_path}")
        if not os.path.exists(ocr_model_path):
            raise FileNotFoundError(f"OCR识别模型文件不存在: {ocr_model_path}")
        
        # 初始化检测器和识别器
        self.detector = LicensePlateDetector(detection_model_path)
        self.ocr = LicensePlateOCR(ocr_model_path)
        self.conf_threshold = conf_threshold
        
        print("车牌识别系统初始化完成")
        print(f"车牌检测模型: {detection_model_path}")
        print(f"OCR识别模型: {ocr_model_path}")
        print(f"检测置信度阈值: {conf_threshold}")
    
    def recognize_single_image(self, image_path, output_dir=None, save_results=True):
        """
        识别单张图像中的车牌
        
        Args:
            image_path: 输入图像路径
            output_dir: 输出目录
            save_results: 是否保存结果
            
        Returns:
            dict: 识别结果
        """
        try:
            # 检查图像文件是否存在
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"图像文件不存在: {image_path}")
            
            print(f"\n开始处理图像: {os.path.basename(image_path)}")
            
            # 1. 车牌检测
            print("步骤1: 车牌检测...")
            detections, original_image = self.detector.detect_plates(
                image_path, self.conf_threshold
            )
            
            result = {
                'image_path': image_path,
                'detections': detections,
                'plate_count': len(detections),
                'success': True,
                'timestamp': datetime.now().isoformat()
            }
            
            # 2. 车牌识别
            if detections:
                print(f"检测到 {len(detections)} 个车牌，开始识别...")
                
                plate_results = []
                for i, detection in enumerate(detections):
                    print(f"  识别第 {i+1} 个车牌...")
                    
                    # 裁剪车牌区域
                    plate_image = self.detector.crop_plate_region(
                        original_image, detection['bbox']
                    )
                    
                    # 临时保存车牌图像用于OCR识别
                    temp_plate_path = f"temp_plate_{i}.jpg"
                    cv2.imwrite(temp_plate_path, plate_image)
                    
                    # OCR识别
                    plate_text = self.ocr.recognize_single_image(temp_plate_path)
                    
                    # 删除临时文件
                    os.remove(temp_plate_path)
                    
                    plate_result = {
                        'plate_id': i + 1,
                        'bbox': detection['bbox'],
                        'confidence': detection['confidence'],
                        'plate_text': plate_text,
                        'recognition_success': plate_text is not None
                    }
                    
                    plate_results.append(plate_result)
                    
                    print(f"    车牌 {i+1} 识别结果: {plate_text}")
                
                result['plate_results'] = plate_results
                result['recognized_count'] = len([r for r in plate_results if r['recognition_success']])
            else:
                print("未检测到车牌")
                result['plate_results'] = []
                result['recognized_count'] = 0
            
            # 3. 保存结果和可视化
            if save_results and output_dir:
                self._save_results(result, original_image, output_dir)
            
            print("处理完成!")
            return result
            
        except Exception as e:
            print(f"处理图像时出错: {e}")
            return {
                'image_path': image_path,
                'error': str(e),
                'success': False,
                'timestamp': datetime.now().isoformat()
            }
    
    def recognize_batch_images(self, image_dir, output_dir=None):
        """
        批量识别图像中的车牌
        
        Args:
            image_dir: 图像目录路径
            output_dir: 输出目录
            
        Returns:
            dict: 批量识别结果汇总
        """
        if not os.path.exists(image_dir):
            raise FileNotFoundError(f"图像目录不存在: {image_dir}")
        
        # 获取所有图像文件
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = []
        
        for file in os.listdir(image_dir):
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_files.append(os.path.join(image_dir, file))
        
        if not image_files:
            print(f"在目录 {image_dir} 中未找到图像文件")
            return {'total_images': 0, 'results': []}
        
        print(f"开始批量处理 {len(image_files)} 张图像...")
        
        results = []
        for i, image_path in enumerate(image_files):
            print(f"\n[{i+1}/{len(image_files)}] 处理: {os.path.basename(image_path)}")
            
            # 为每张图像创建单独的输出子目录
            image_output_dir = None
            if output_dir:
                image_name = Path(image_path).stem
                image_output_dir = os.path.join(output_dir, image_name)
            
            result = self.recognize_single_image(
                image_path, image_output_dir, save_results=True
            )
            results.append(result)
        
        # 生成汇总报告
        summary = self._generate_summary(results)
        
        # 保存汇总报告
        if output_dir:
            self._save_summary_report(summary, output_dir)
        
        return summary
    
    def _save_results(self, result, original_image, output_dir):
        """保存识别结果"""
        os.makedirs(output_dir, exist_ok=True)
        
        image_name = Path(result['image_path']).stem
        
        # 1. 保存可视化图像
        if result['detections']:
            vis_image = self.detector.visualize_detection(original_image, result['detections'])
            
            # 在可视化图像上添加识别结果
            for plate_result in result.get('plate_results', []):
                if plate_result['recognition_success']:
                    x1, y1, x2, y2 = plate_result['bbox']
                    plate_text = plate_result['plate_text']
                    
                    # 添加识别结果文本
                    text = f"{plate_text}"
                    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                    
                    cv2.rectangle(vis_image, (x1, y2), 
                                 (x1 + text_size[0] + 10, y2 + text_size[1] + 10), 
                                 (0, 255, 0), -1)
                    cv2.putText(vis_image, text, (x1 + 5, y2 + text_size[1] + 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            
            vis_path = os.path.join(output_dir, f"{image_name}_detection.jpg")
            cv2.imwrite(vis_path, vis_image)
            result['visualization_path'] = vis_path
        
        # 2. 保存识别结果JSON
        json_path = os.path.join(output_dir, f"{image_name}_result.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            # 移除图像对象，只保存路径信息
            result_copy = result.copy()
            if 'visualization_path' in result_copy:
                result_copy['visualization_path'] = str(result_copy['visualization_path'])
            json.dump(result_copy, f, indent=2, ensure_ascii=False)
        
        # 3. 保存车牌区域图像
        if result.get('plate_results'):
            plates_dir = os.path.join(output_dir, "plates")
            os.makedirs(plates_dir, exist_ok=True)
            
            for plate_result in result['plate_results']:
                plate_image = self.detector.crop_plate_region(
                    original_image, plate_result['bbox']
                )
                
                plate_filename = f"plate_{plate_result['plate_id']}_{image_name}.jpg"
                plate_path = os.path.join(plates_dir, plate_filename)
                cv2.imwrite(plate_path, plate_image)
                
                plate_result['plate_image_path'] = plate_path
    
    def _generate_summary(self, results):
        """生成批量处理汇总报告"""
        total_images = len(results)
        successful_images = len([r for r in results if r['success']])
        total_plates_detected = sum([r.get('plate_count', 0) for r in results if r['success']])
        total_plates_recognized = sum([r.get('recognized_count', 0) for r in results if r['success']])
        
        summary = {
            'total_images': total_images,
            'successful_images': successful_images,
            'failed_images': total_images - successful_images,
            'total_plates_detected': total_plates_detected,
            'total_plates_recognized': total_plates_recognized,
            'detection_rate': total_plates_detected / successful_images if successful_images > 0 else 0,
            'recognition_rate': total_plates_recognized / total_plates_detected if total_plates_detected > 0 else 0,
            'overall_success_rate': successful_images / total_images if total_images > 0 else 0,
            'results': results
        }
        
        return summary
    
    def _save_summary_report(self, summary, output_dir):
        """保存汇总报告"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存JSON格式汇总
        summary_path = os.path.join(output_dir, "recognition_summary.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        # 保存文本格式汇总
        txt_path = os.path.join(output_dir, "recognition_summary.txt")
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("=== 车牌识别系统批量处理汇总报告 ===\n\n")
            f.write(f"处理时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"总图像数: {summary['total_images']}\n")
            f.write(f"成功处理: {summary['successful_images']}\n")
            f.write(f"处理失败: {summary['failed_images']}\n")
            f.write(f"处理成功率: {summary['overall_success_rate']:.2%}\n\n")
            
            f.write(f"检测到车牌总数: {summary['total_plates_detected']}\n")
            f.write(f"成功识别车牌数: {summary['total_plates_recognized']}\n")
            f.write(f"车牌检测率: {summary['detection_rate']:.2%}\n")
            f.write(f"车牌识别率: {summary['recognition_rate']:.2%}\n\n")
            
            f.write("=== 详细结果 ===\n")
            for i, result in enumerate(summary['results']):
                f.write(f"\n图像 {i+1}: {os.path.basename(result['image_path'])} - ")
                if result['success']:
                    f.write(f"检测到 {result['plate_count']} 个车牌，识别 {result['recognized_count']} 个\n")
                    if result.get('plate_results'):
                        for plate in result['plate_results']:
                            status = "成功" if plate['recognition_success'] else "失败"
                            f.write(f"  车牌 {plate['plate_id']}: {plate['plate_text']} ({status})\n")
                else:
                    f.write(f"处理失败: {result.get('error', '未知错误')}\n")

def main():
    """主函数 - 使用示例"""
    import argparse
    
    parser = argparse.ArgumentParser(description='车牌识别系统')
    parser.add_argument('--mode', type=str, required=True,
                       choices=['single', 'batch'],
                       help='运行模式: single(单张图像), batch(批量识别)')
    parser.add_argument('--image', type=str, help='单张图像路径')
    parser.add_argument('--image_dir', type=str, help='图像目录路径')
    parser.add_argument('--output_dir', type=str, default='../results/recognition_results',
                       help='输出目录路径')
    parser.add_argument('--detection_model', type=str, default='./plate_detection_model.pt',
                       help='车牌检测模型路径')
    parser.add_argument('--ocr_model', type=str, default='./plate_ocr_model.pt',
                       help='OCR识别模型路径')
    parser.add_argument('--conf_threshold', type=float, default=0.5,
                       help='检测置信度阈值')
    
    args = parser.parse_args()
    
    # 创建识别系统
    try:
        recognition_system = LicensePlateRecognitionSystem(
            detection_model_path=args.detection_model,
            ocr_model_path=args.ocr_model,
            conf_threshold=args.conf_threshold
        )
    except Exception as e:
        print(f"初始化识别系统失败: {e}")
        return
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.mode == 'single':
        if not args.image:
            print("请使用 --image 参数指定图像路径")
            return
        
        result = recognition_system.recognize_single_image(
            args.image, args.output_dir
        )
        
        # 打印结果
        print("\n=== 识别结果 ===")
        print(f"图像: {os.path.basename(result['image_path'])}")
        print(f"检测到车牌数: {result['plate_count']}")
        print(f"成功识别数: {result.get('recognized_count', 0)}")
        
        if result.get('plate_results'):
            for plate in result['plate_results']:
                status = "成功" if plate['recognition_success'] else "失败"
                print(f"车牌 {plate['plate_id']}: {plate['plate_text']} ({status})")
    
    elif args.mode == 'batch':
        if not args.image_dir:
            print("请使用 --image_dir 参数指定图像目录")
            return
        
        summary = recognition_system.recognize_batch_images(
            args.image_dir, args.output_dir
        )
        
        # 打印汇总
        print("\n=== 批量处理汇总 ===")
        print(f"总图像数: {summary['total_images']}")
        print(f"成功处理: {summary['successful_images']}")
        print(f"检测到车牌总数: {summary['total_plates_detected']}")
        print(f"成功识别车牌数: {summary['total_plates_recognized']}")
        print(f"检测率: {summary['detection_rate']:.2%}")
        print(f"识别率: {summary['recognition_rate']:.2%}")

if __name__ == "__main__":
    main()