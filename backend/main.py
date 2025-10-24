from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import mysql.connector
from datetime import date, datetime
from fastapi import UploadFile, File
import os
import sys
import tempfile

# 添加模型路径到系统路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'models'))

# 导入车牌识别模块
try:
    from plate.inference import LicensePlateDetector
    from ocr.inference import LicensePlateOCR
except ImportError as e:
    print(f"导入模块失败: {e}")
    print("请确保模型文件存在且路径正确")
import os
import sys
import tempfile

# 添加模型路径到系统路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'models'))

# 导入车牌识别模块
try:
    from plate.inference import LicensePlateDetector
    from ocr.inference import LicensePlateOCR
except ImportError as e:
    print(f"导入模块失败: {e}")
    print("请确保模型文件存在且路径正确")

# FastAPI应用初始化
app = FastAPI()

# 允许跨域请求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 数据库配置
DB_CONFIG = {
    "host": "127.0.0.1",
    "user": "parking",
    "password": "123456",
    "database": "smart_parking"
}

# 车牌识别类
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
    
    def recognize_plate(self, image_path: str) -> dict:
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

# 全局车牌识别实例
plate_recognizer = LicensePlateRecognizer()

# 车牌识别类
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
    
    def recognize_plate(self, image_path: str) -> dict:
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

# 全局车牌识别实例
plate_recognizer = LicensePlateRecognizer()

# 数据模型
class LicensePlate(BaseModel):
    plate_number: str
    owner_name: Optional[str] = None
    valid_from: date
    valid_to: date
    is_active: Optional[bool] = True

class LicensePlateResponse(LicensePlate):
    id: int
    created_at: datetime
    updated_at: datetime

# 数据库连接依赖
def get_db_connection():
    conn = mysql.connector.connect(**DB_CONFIG)
    try:
        yield conn
    finally:
        conn.close()

# API路由
@app.get("/plates/", response_model=List[LicensePlateResponse])
def list_plates(conn=Depends(get_db_connection)):
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM license_plates")
    plates = cursor.fetchall()
    cursor.close()
    return plates

@app.post("/plates/", response_model=LicensePlateResponse)
def create_plate(plate: LicensePlate, conn=Depends(get_db_connection)):
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO license_plates (plate_number, owner_name, valid_from, valid_to, is_active)
        VALUES (%s, %s, %s, %s, %s)
        """,
        (plate.plate_number, plate.owner_name, plate.valid_from, plate.valid_to, plate.is_active)
    )
    conn.commit()
    plate_id = cursor.lastrowid
    cursor.close()
    return {**plate.dict(), "id": plate_id, "created_at": "now", "updated_at": "now"}

@app.get("/plates/{plate_id}", response_model=LicensePlateResponse)
def get_plate(plate_id: int, conn=Depends(get_db_connection)):
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM license_plates WHERE id = %s", (plate_id,))
    plate = cursor.fetchone()
    cursor.close()
    if not plate:
        raise HTTPException(status_code=404, detail="Plate not found")
    return plate

@app.put("/plates/{plate_id}", response_model=LicensePlateResponse)
def update_plate(plate_id: int, plate: LicensePlate, conn=Depends(get_db_connection)):
    cursor = conn.cursor()
    cursor.execute(
        """
        UPDATE license_plates
        SET plate_number = %s, owner_name = %s, valid_from = %s, valid_to = %s, is_active = %s
        WHERE id = %s
        """,
        (plate.plate_number, plate.owner_name, plate.valid_from, plate.valid_to, plate.is_active, plate_id)
    )
    conn.commit()
    cursor.close()
    return {**plate.dict(), "id": plate_id, "updated_at": "now"}

@app.delete("/plates/{plate_id}")
def delete_plate(plate_id: int, conn=Depends(get_db_connection)):
    cursor = conn.cursor()
    cursor.execute("DELETE FROM license_plates WHERE id = %s", (plate_id,))
    conn.commit()
    cursor.close()
    return {"message": "Plate deleted successfully"}

# 车牌检测API
@app.post("/detect/")
async def detect_plate(image: UploadFile = File(...), conn=Depends(get_db_connection)):
    # 验证文件类型
    if not image.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="请上传图片文件")
    
    # 检查模型是否初始化
    if not plate_recognizer.initialized:
        raise HTTPException(status_code=500, detail="车牌识别模型未初始化")
    
    # 创建临时文件
    temp_file = None
    try:
        # 保存上传的文件到临时文件
        file_extension = os.path.splitext(image.filename)[1] if image.filename else '.jpg'
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=file_extension)
        
        # 读取上传的文件内容并写入临时文件
        content = await image.read()
        temp_file.write(content)
        temp_file.close()
        
        # 调用车牌识别
        recognition_result = plate_recognizer.recognize_plate(temp_file.name)
        
        # 获取识别结果
        plate_number = recognition_result.get('plate_number')
        message = recognition_result.get('message', '')
        
        # 如果识别失败，返回错误信息
        if not recognition_result['success']:
            return {
                "plate_number": None,
                "message": recognition_result.get('error', '识别失败'),
                "is_valid": False,
                "valid_to": None,
                "owner_name": None
            }
        
        # 如果识别到车牌号码，查询数据库验证有效性
        if plate_number:
            cursor = conn.cursor(dictionary=True)
            cursor.execute(
                "SELECT * FROM license_plates WHERE plate_number = %s AND is_active = 1",
                (plate_number,)
            )
            plate_info = cursor.fetchone()
            cursor.close()
            
            if plate_info:
                # 检查有效期
                today = date.today()
                is_valid = plate_info['valid_to'] >= today
                
                return {
                    "plate_number": plate_number,
                    "message": message,
                    "is_valid": is_valid,
                    "valid_to": plate_info['valid_to'].isoformat(),
                    "owner_name": plate_info['owner_name']
                }
            else:
                return {
                    "plate_number": plate_number,
                    "message": f"{message}，但车牌不在系统中",
                    "is_valid": False,
                    "valid_to": None,
                    "owner_name": None
                }
        else:
            # 没有识别到车牌号码
            return {
                "plate_number": None,
                "message": message,
                "is_valid": False,
                "valid_to": None,
                "owner_name": None
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"检测过程中发生错误: {str(e)}")
    
    finally:
        # 清理临时文件
        if temp_file and os.path.exists(temp_file.name):
            os.unlink(temp_file.name)

def extract_plate_region(image, box):
    """根据边界框提取车牌区域"""
    x1, y1, x2, y2 = box
    plate_region = image[y1:y2, x1:x2]
    return plate_region

# 启动应用
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)