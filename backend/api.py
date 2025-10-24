from fastapi import FastAPI, HTTPException, Depends, UploadFile, File
from pydantic import BaseModel
from typing import Optional, List
import mysql.connector
from datetime import date, datetime
import os
import tempfile

from plate_model import LicensePlateRecognizer

# 数据库配置
DB_CONFIG = {
    "host": "127.0.0.1",
    "user": "parking",
    "password": "123456",
    "database": "smart_parking"
}

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

# 全局车牌识别实例
plate_recognizer = LicensePlateRecognizer()

def setup_routes(app: FastAPI):
    """设置API路由"""
    
    # 车牌管理API
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
        
        # 重新从数据库查询完整的记录，确保时间字段格式正确
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM license_plates WHERE id = %s", (plate_id,))
        new_plate = cursor.fetchone()
        cursor.close()
        
        if not new_plate:
            raise HTTPException(status_code=500, detail="Failed to retrieve created plate")
        
        return new_plate

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
        
        # 重新从数据库查询完整的记录，确保时间字段格式正确
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM license_plates WHERE id = %s", (plate_id,))
        updated_plate = cursor.fetchone()
        cursor.close()
        
        if not updated_plate:
            raise HTTPException(status_code=404, detail="Plate not found")
        
        return updated_plate

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