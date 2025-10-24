"""
车牌识别推理模块
"""

import cv2
from ultralytics import YOLO
import os

class LicensePlateDetector:
    def __init__(self, model_path):
        """初始化检测器"""
        self.model = YOLO(model_path)
        self.class_names = ['license_plate']
        
    def detect_plates(self, image_path, conf_threshold=0.5, iou_threshold=0.5):
        """检测图像中的车牌"""
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图像: {image_path}")
        
        # 使用YOLO进行检测
        results = self.model.predict(
            source=image,
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=False
        )
        
        # 解析检测结果
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # 获取边界框坐标
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    cls = int(box.cls[0].cpu().numpy())
                    
                    detections.append({
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': float(conf),
                        'class_name': self.class_names[cls],
                        'class_id': cls
                    })
        
        return detections, image
    
    def crop_plate_region(self, image, bbox):
        """裁剪车牌区域"""
        x1, y1, x2, y2 = bbox
        plate_image = image[y1:y2, x1:x2]
        return plate_image
    
    def visualize_detection(self, image, detections, output_path=None):
        """可视化检测结果"""
        result_image = image.copy()
        
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            conf = detection['confidence']
            class_name = detection['class_name']
            
            # 绘制边界框
            cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 绘制标签
            label = f"{class_name}: {conf:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            
            cv2.rectangle(result_image, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), (0, 255, 0), -1)
            cv2.putText(result_image, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        if output_path:
            cv2.imwrite(output_path, result_image)
        
        return result_image
    
    def process_single_image(self, image_path, output_dir=None, conf_threshold=0.5):
        """处理单张图像"""
        try:
            # 检测车牌
            detections, image = self.detect_plates(image_path, conf_threshold)
            
            # 准备结果
            result = {
                'image_path': image_path,
                'detections': detections,
                'plate_count': len(detections),
                'success': True
            }
            
            # 如果有检测结果，裁剪车牌区域
            if detections:
                plate_images = []
                for i, detection in enumerate(detections):
                    plate_image = self.crop_plate_region(image, detection['bbox'])
                    plate_images.append(plate_image)
                    
                    # 保存裁剪的车牌图像
                    if output_dir:
                        os.makedirs(output_dir, exist_ok=True)
                        plate_filename = f"plate_{i}_{os.path.basename(image_path)}"
                        plate_path = os.path.join(output_dir, plate_filename)
                        cv2.imwrite(plate_path, plate_image)
                        detection['plate_image_path'] = plate_path
                
                result['plate_images'] = plate_images
            
            # 保存可视化结果
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                vis_filename = f"detection_{os.path.basename(image_path)}"
                vis_path = os.path.join(output_dir, vis_filename)
                self.visualize_detection(image, detections, vis_path)
                result['visualization_path'] = vis_path
            
            return result
            
        except Exception as e:
            return {
                'image_path': image_path,
                'error': str(e),
                'success': False
            }
    
    def process_batch_images(self, image_dir, output_dir=None, conf_threshold=0.5):
        """批量处理图像"""
        results = []
        
        # 获取所有图像文件
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        
        for file in os.listdir(image_dir):
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_files.append(os.path.join(image_dir, file))
        
        print(f"找到 {len(image_files)} 张图像")
        
        # 处理每张图像
        for image_path in image_files:
            print(f"处理: {os.path.basename(image_path)}")
            result = self.process_single_image(image_path, output_dir, conf_threshold)
            results.append(result)
        
        # 统计结果
        total_detections = sum(len(result.get('detections', [])) for result in results)
        successful_processing = sum(1 for result in results if result['success'])
        
        summary = {
            'total_images': len(results),
            'successful_processing': successful_processing,
            'total_plates_detected': total_detections,
            'results': results
        }
        
        return summary

def main():
    """主推理函数"""
    # 配置路径
    model_path = "../plate_detection_model.pt"
    test_image_path = "../../data/images/"  # 测试图像目录
    output_dir = "../../results"
    
    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        print("模型文件不存在，请先训练模型")
        return
    
    # 创建检测器
    detector = LicensePlateDetector(model_path)
    
    # 批量处理测试图像
    summary = detector.process_batch_images(test_image_path, output_dir)
    
    # 打印结果摘要
    print("\n检测结果摘要:")
    print(f"总图像数: {summary['total_images']}")
    print(f"成功处理: {summary['successful_processing']}")
    print(f"检测到车牌数: {summary['total_plates_detected']}")
    
    # 保存详细结果
    results_file = os.path.join(output_dir, "detection_results.json")
    import json
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"详细结果已保存到: {results_file}")

if __name__ == "__main__":
    main()