import os
import cv2
import torch
import numpy as np
from PIL import Image
import argparse
import sys

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from crnn_model import LicensePlateRecognizer

class LicensePlateOCR:
    """车牌OCR推理类"""
    
    def __init__(self, model_path="../plate_ocr_model.pt", device=None):
        """
        初始化OCR推理器
        
        Args:
            model_path: 模型文件路径
            device: 设备 (cpu/cuda)
        """
        self.model_path = model_path
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 加载模型
        self.recognizer = LicensePlateRecognizer(model_path, self.device)
        
        print(f"OCR模型加载完成，使用设备: {self.device}")
    
    def preprocess_image(self, image_path):
        """预处理图像"""
        try:
            # 读取图像
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"无法读取图像: {image_path}")
            
            # 转换为灰度图
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # 调整图像大小
            image = cv2.resize(image, (128, 32))
            
            # 归一化
            image = image.astype(np.float32) / 255.0
            
            # 转换为PyTorch张量
            image_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)  # [1, 1, 32, 128]
            
            return image_tensor
            
        except Exception as e:
            print(f"图像预处理失败: {e}")
            return None
    
    def recognize_single_image(self, image_path):
        """识别单张图像"""
        try:
            # 预处理图像
            image_tensor = self.preprocess_image(image_path)
            if image_tensor is None:
                return None
            
            # 识别
            result = self.recognizer.recognize_single_image(image_tensor)
            
            return result
        except Exception as e:
            print(f"识别图像 {image_path} 时出错: {e}")
            return None
    
    def recognize_batch_images(self, image_dir, output_dir=None):
        """批量识别图像"""
        if not os.path.exists(image_dir):
            print(f"图像目录不存在: {image_dir}")
            return []
        
        # 获取所有图像文件
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = []
        for ext in image_extensions:
            image_files.extend([f for f in os.listdir(image_dir) if f.lower().endswith(ext)])
        
        if not image_files:
            print(f"在目录 {image_dir} 中未找到图像文件")
            return []
        
        results = []
        print(f"开始批量识别 {len(image_files)} 张图像...")
        
        for i, image_file in enumerate(image_files):
            image_path = os.path.join(image_dir, image_file)
            
            try:
                # 识别
                plate_text = self.recognize_single_image(image_path)
                
                result = {
                    'image_file': image_file,
                    'plate_text': plate_text,
                    'success': plate_text is not None
                }
                
                results.append(result)
                
                # 打印进度
                if (i + 1) % 10 == 0 or (i + 1) == len(image_files):
                    print(f"已处理 {i+1}/{len(image_files)} 张图像")
                
                # 保存结果到文件
                if output_dir:
                    self._save_single_result(result, output_dir)
                    
            except Exception as e:
                print(f"处理图像 {image_file} 时出错: {e}")
                results.append({
                    'image_file': image_file,
                    'plate_text': None,
                    'success': False,
                    'error': str(e)
                })
        
        # 保存批量结果
        if output_dir:
            self._save_batch_results(results, output_dir)
        
        return results
    
    def _save_single_result(self, result, output_dir):
        """保存单个结果"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 创建结果文件
        result_file = os.path.join(output_dir, 'ocr_results.txt')
        
        with open(result_file, 'a', encoding='utf-8') as f:
            status = "成功" if result['success'] else "失败"
            f.write(f"{result['image_file']},{result['plate_text']},{status}\n")
    
    def _save_batch_results(self, results, output_dir):
        """保存批量结果"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 创建汇总文件
        summary_file = os.path.join(output_dir, 'ocr_summary.txt')
        
        successful = [r for r in results if r['success']]
        failed = [r for r in results if not r['success']]
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("=== 车牌OCR识别结果汇总 ===\n")
            f.write(f"总图像数: {len(results)}\n")
            f.write(f"成功识别: {len(successful)}\n")
            f.write(f"识别失败: {len(failed)}\n")
            f.write(f"识别准确率: {len(successful)/len(results)*100:.2f}%\n\n")
            
            f.write("=== 成功识别结果 ===\n")
            for result in successful:
                f.write(f"{result['image_file']}: {result['plate_text']}\n")
            
            f.write("\n=== 识别失败图像 ===\n")
            for result in failed:
                f.write(f"{result['image_file']}\n")
        
        print(f"结果汇总已保存到: {summary_file}")
    
    def test_accuracy(self, test_labels_file, test_image_dir):
        """测试模型准确率"""
        if not os.path.exists(test_labels_file):
            print(f"测试标签文件不存在: {test_labels_file}")
            return None
        
        # 读取测试标签
        test_samples = []
        with open(test_labels_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split(',')
                    if len(parts) >= 2:
                        image_path = parts[0]
                        label = parts[1]
                        test_samples.append((image_path, label))
        
        print(f"开始测试准确率，共 {len(test_samples)} 个样本...")
        
        correct = 0
        total = 0
        
        for i, (image_path, true_label) in enumerate(test_samples):
            # 修复路径处理：标签文件中的路径已经是完整路径
            # 格式：data/ocr_data/datasets/test/images/000000.jpg
            # 我们需要从项目根目录开始构建路径
            
            # 提取相对路径（去掉开头的 data/ocr_data/）
            if image_path.startswith('data/ocr_data/'):
                relative_path = image_path.replace('data/ocr_data/', '')
            else:
                relative_path = image_path
            
            # 构建完整图像路径（从项目根目录开始）
            full_image_path = os.path.join('../../data/ocr_data', relative_path)
            
            if not os.path.exists(full_image_path):
                print(f"图像文件不存在: {full_image_path}")
                continue
            
            try:
                # 识别
                predicted_text = self.recognize_single_image(full_image_path)
                
                if predicted_text == true_label:
                    correct += 1
                
                total += 1
                
                # 打印进度
                if (i + 1) % 100 == 0 or (i + 1) == len(test_samples):
                    accuracy = correct / total * 100 if total > 0 else 0
                    print(f"已测试 {i+1}/{len(test_samples)}，当前准确率: {accuracy:.2f}%")
                
            except Exception as e:
                print(f"测试图像 {image_path} 时出错: {e}")
        
        accuracy = correct / total * 100 if total > 0 else 0
        
        print(f"\n=== 测试结果 ===")
        print(f"总样本数: {total}")
        print(f"正确识别: {correct}")
        print(f"识别准确率: {accuracy:.2f}%")
        
        return accuracy

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='车牌OCR识别')
    parser.add_argument('--mode', type=str, default='single', 
                       choices=['single', 'batch', 'test'],
                       help='运行模式: single(单张图像), batch(批量识别), test(测试准确率)')
    parser.add_argument('--image', type=str, help='单张图像路径')
    parser.add_argument('--image_dir', type=str, help='图像目录路径')
    parser.add_argument('--output_dir', type=str, default='../../results/ocr_results',
                       help='输出目录路径')
    parser.add_argument('--model', type=str, default='../plate_ocr_model.pt',
                       help='模型文件路径')
    
    args = parser.parse_args()
    
    # 创建OCR识别器
    ocr = LicensePlateOCR(args.model)
    
    if args.mode == 'single':
        if not args.image:
            print("请使用 --image 参数指定图像路径")
            return
        
        result = ocr.recognize_single_image(args.image)
        if result:
            print(f"识别结果: {result}")
        else:
            print("识别失败")
    
    elif args.mode == 'batch':
        if not args.image_dir:
            print("请使用 --image_dir 参数指定图像目录")
            return
        
        results = ocr.recognize_batch_images(args.image_dir, args.output_dir)
        print(f"批量识别完成，共处理 {len(results)} 张图像")
    
    elif args.mode == 'test':
        test_labels = "../../data/ocr_data/test/test_labels.txt"
        test_image_dir = "../../data/ocr_data/test/images"
        
        accuracy = ocr.test_accuracy(test_labels, test_image_dir)

if __name__ == "__main__":
    main()