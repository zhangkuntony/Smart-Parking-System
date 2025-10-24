import os
import sys
from typing import Dict, Any

# 添加模型路径到系统路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'models'))

# 导入车牌识别模块
try:
    from plate.inference import LicensePlateDetector
    from ocr.inference import LicensePlateOCR
except ImportError as e:
    print(f"导入模块失败: {e}")
    print("请确保模型文件存在且路径正确")


class LicensePlateRecognizer:
    """车牌识别类"""
    
    def __init__(self):
        """初始化车牌识别系统"""
        self.detector = None
        self.ocr = None
        self.initialized = False
        
        # 模型路径
        self.detection_model_path = os.path.join(
            os.path.dirname(__file__), 'models', 'plate_detection_model.pt'
        )
        self.ocr_model_path = os.path.join(
            os.path.dirname(__file__), 'models', 'plate_ocr_model.pt'
        )
        
        self._initialize_models()
    
    def _initialize_models(self):
        """初始化模型"""
        try:
            # 检查模型文件是否存在
            if not os.path.exists(self.detection_model_path):
                raise FileNotFoundError(f"车牌检测模型文件不存在: {self.detection_model_path}")
            
            if not os.path.exists(self.ocr_model_path):
                raise FileNotFoundError(f"OCR识别模型文件不存在: {self.ocr_model_path}")
            
            # 初始化检测器和识别器
            self.detector = LicensePlateDetector(self.detection_model_path)
            self.ocr = LicensePlateOCR(self.ocr_model_path)
            
            self.initialized = True
            print("车牌识别系统初始化完成")
            
        except Exception as e:
            print(f"模型初始化失败: {e}")
            self.initialized = False
    
    def recognize_plate(self, image_path: str) -> Dict[str, Any]:
        """
        识别车牌
        
        Args:
            image_path: 图像文件路径
            
        Returns:
            dict: 识别结果，包含车牌号码和识别状态
        """
        if not self.initialized:
            return {
                'success': False,
                'error': '模型未初始化',
                'plate_number': None,
                'message': '模型未初始化'
            }
        
        try:
            # 检查图像文件是否存在
            if not os.path.exists(image_path):
                return {
                    'success': False,
                    'error': f'图像文件不存在: {image_path}',
                    'plate_number': None,
                    'message': f'图像文件不存在: {image_path}'
                }
            
            # 1. 车牌检测
            detections, original_image = self.detector.detect_plates(image_path, conf_threshold=0.5)
            
            result = {
                'image_path': image_path,
                'detections': detections,
                'plate_count': len(detections),
                'success': True,
                'plate_number': None,
                'message': ''
            }
            
            # 2. 车牌识别
            if detections:
                plate_results = []
                for i, detection in enumerate(detections):
                    # 裁剪车牌区域
                    plate_image = self.detector.crop_plate_region(original_image, detection['bbox'])
                    
                    # 临时保存车牌图像用于OCR识别
                    temp_plate_path = f"temp_plate_{i}.jpg"
                    import cv2
                    cv2.imwrite(temp_plate_path, plate_image)
                    
                    # OCR识别
                    plate_text = self.ocr.recognize_single_image(temp_plate_path)
                    
                    # 删除临时文件
                    if os.path.exists(temp_plate_path):
                        os.remove(temp_plate_path)
                    
                    plate_result = {
                        'plate_id': i + 1,
                        'bbox': detection['bbox'],
                        'confidence': detection['confidence'],
                        'plate_text': plate_text,
                        'recognition_success': plate_text is not None
                    }
                    
                    plate_results.append(plate_result)
                    
                    # 使用第一个成功识别的车牌号码
                    if plate_text and result['plate_number'] is None:
                        result['plate_number'] = plate_text
                
                result['plate_results'] = plate_results
                result['recognized_count'] = len([r for r in plate_results if r['recognition_success']])
                
                # 生成识别成功字符串
                if result['recognized_count'] > 0:
                    plate_texts = [r['plate_text'] for r in plate_results if r['recognition_success']]
                    result['message'] = f"识别成功！检测到 {len(detections)} 个车牌，成功识别 {result['recognized_count']} 个：{', '.join(plate_texts)}"
                else:
                    result['message'] = "检测到车牌但识别失败"
                    result['plate_number'] = None
            else:
                result['plate_results'] = []
                result['recognized_count'] = 0
                result['message'] = "未检测到车牌"
                result['plate_number'] = None
            
            return result
            
        except Exception as e:
            return {
                'image_path': image_path,
                'error': str(e),
                'success': False,
                'plate_number': None,
                'message': f"识别失败：{str(e)}"
            }


def extract_plate_region(image, box):
    """根据边界框提取车牌区域"""
    x1, y1, x2, y2 = box
    plate_region = image[y1:y2, x1:x2]
    return plate_region