-- 创建数据库
CREATE DATABASE IF NOT EXISTS smart_parking;
USE smart_parking;

-- 创建车牌信息表
CREATE TABLE IF NOT EXISTS license_plates (
    id INT AUTO_INCREMENT PRIMARY KEY,
    plate_number VARCHAR(20) NOT NULL UNIQUE,
    owner_name VARCHAR(10),
    valid_from DATE NOT NULL,
    valid_to DATE NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

-- 创建用户表（可选，用于管理系统用户）
CREATE TABLE IF NOT EXISTS users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(50) NOT NULL UNIQUE,
    password_hash VARCHAR(255) NOT NULL,
    is_admin BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 插入示例数据
INSERT INTO license_plates (plate_number, owner_name, valid_from, valid_to)
VALUES 
    ('京A12345', '张三', '2025-01-01', '2026-01-01'),
    ('沪B67890', '李四', '2025-02-01', '2026-02-01'),
    ('粤C13579', '王五', '2025-03-01', '2026-03-01');