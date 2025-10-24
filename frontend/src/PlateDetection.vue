<template>
  <div>
    <el-card class="upload-card">
      <template #header>
        <div class="card-header">
          <span>上传车牌图片</span>
        </div>
      </template>
      
      <!-- 上传区域 - 上传图片后隐藏 -->
      <div v-if="!imageFile">
        <!-- 拖拽上传区域 -->
        <el-upload
          class="drag-upload"
          drag
          action=""
          :auto-upload="false"
          :on-change="handleFileUpload"
          :show-file-list="false"
          :before-upload="beforeUpload"
        >
          <div class="drag-area">
            <el-icon class="el-icon--upload">
              <upload-filled />
            </el-icon>
            <div class="el-upload__text">
              将文件拖到此处，或<em>点击上传</em>
            </div>
            <div class="el-upload__tip" slot="tip">
              支持 jpg、png 格式图片，大小不超过 5MB
            </div>
          </div>
        </el-upload>

        <!-- 按钮区域 - 放在同一行 -->
        <div class="button-group">
          <el-upload
            class="button-upload"
            action=""
            :auto-upload="false"
            :on-change="handleFileUpload"
            :show-file-list="false"
          >
            <el-button type="primary" size="large">
              <el-icon><upload /></el-icon>
              选择图片
            </el-button>
          </el-upload>
          
          <el-button 
            type="success" 
            size="large"
            :disabled="true"
          >
            <el-icon><search /></el-icon>
            检测车牌
          </el-button>
        </div>
      </div>

      <!-- 图片预览区域 -->
      <div v-if="imagePreview" class="preview-container">
        <h3>图片预览</h3>
        <img :src="imagePreview" alt="预览图" class="preview-image" />
        
        <!-- 操作按钮区域 - 上传图片后显示 -->
        <div class="action-buttons">
          <el-button 
            type="warning" 
            size="large"
            @click="clearImage"
          >
            <el-icon><delete /></el-icon>
            删除图片
          </el-button>
          
          <el-button 
            type="success" 
            size="large"
            @click="uploadImage" 
            :loading="loading"
          >
            <el-icon><search /></el-icon>
            检测车牌
          </el-button>
        </div>
      </div>
    </el-card>

    <el-card class="result-card" v-if="detectionResult">
      <template #header>
        <div class="card-header">
          <span>检测结果</span>
        </div>
      </template>
      <el-descriptions border>
        <el-descriptions-item label="车牌号码">{{ detectionResult.plate_number }}</el-descriptions-item>
        <el-descriptions-item label="有效期至">{{ detectionResult.valid_to }}</el-descriptions-item>
        <el-descriptions-item label="是否有效">
          <el-tag :type="detectionResult.is_valid ? 'success' : 'danger'">
            {{ detectionResult.is_valid ? '有效' : '无效' }}
          </el-tag>
        </el-descriptions-item>
      </el-descriptions>
    </el-card>
  </div>
</template>

<script>
import { UploadFilled, Upload, Search, Delete } from '@element-plus/icons-vue'

export default {
  name: 'PlateDetection',
  components: {
    UploadFilled,
    Upload,
    Search,
    Delete
  },
  data() {
    return {
      imageFile: null,
      imagePreview: null,
      detectionResult: null,
      loading: false
    }
  },
  methods: {
    beforeUpload(file) {
      const isImage = file.type === 'image/jpeg' || file.type === 'image/png'
      const isLt5M = file.size / 1024 / 1024 < 5

      if (!isImage) {
        this.$message.error('只能上传 JPG/PNG 格式的图片!')
        return false
      }
      if (!isLt5M) {
        this.$message.error('图片大小不能超过 5MB!')
        return false
      }
      return true
    },
    
    handleFileUpload(file) {
      if (file && this.beforeUpload(file.raw)) {
        this.imageFile = file.raw
        // 生成预览图
        const reader = new FileReader()
        reader.onload = (e) => {
          this.imagePreview = e.target.result
        }
        reader.readAsDataURL(file.raw)
      }
    },
    
    async uploadImage() {
      if (!this.imageFile) {
        this.$message.warning('请先选择图片')
        return
      }

      this.loading = true
      const formData = new FormData()
      formData.append('image', this.imageFile)

      try {
        const response = await fetch('http://localhost:8000/detect/', {
          method: 'POST',
          body: formData
        })
        
        if (!response.ok) {
          throw new Error('检测失败')
        }
        
        this.detectionResult = await response.json()
        this.$message.success('车牌检测成功!')
      } catch (error) {
        console.error('Error:', error)
        this.$message.error('车牌检测失败，请重试')
      } finally {
        this.loading = false
      }
    },
    
    clearImage() {
      this.imageFile = null
      this.imagePreview = null
      this.detectionResult = null
      this.$message.info('图片已删除，可以重新上传')
    }
  }
}
</script>

<style scoped>
.upload-card, .result-card {
  max-width: 800px;
  margin: 20px auto;
}

.card-header {
  font-size: 18px;
  font-weight: bold;
}

/* 拖拽上传样式 */
.drag-upload {
  margin-bottom: 20px;
}

.drag-area {
  padding: 40px 20px;
  text-align: center;
}

.el-icon--upload {
  font-size: 67px;
  color: #c0c4cc;
  margin-bottom: 16px;
  line-height: 50px;
}

.el-upload__text {
  font-size: 14px;
  color: #606266;
  margin-bottom: 8px;
}

.el-upload__tip {
  font-size: 12px;
  color: #909399;
}

/* 按钮组样式 */
.button-group {
  display: flex;
  justify-content: center;
  gap: 20px;
  margin-top: 20px;
}

.button-upload {
  display: inline-block;
}

/* 预览区域样式 */
.preview-container {
  margin: 20px 0;
  text-align: center;
}

.preview-container h3 {
  margin-bottom: 15px;
  color: #606266;
}

.preview-image {
  max-width: 100%;
  max-height: 300px;
  border-radius: 8px;
  box-shadow: 0 2px 12px 0 rgba(0, 0, 0, 0.1);
}

/* 操作按钮区域样式 */
.action-buttons {
  display: flex;
  justify-content: center;
  gap: 20px;
  margin-top: 20px;
}

/* 响应式设计 */
@media (max-width: 768px) {
  .button-group {
    flex-direction: column;
    gap: 10px;
  }
  
  .button-group .el-button {
    width: 100%;
  }
  
  .action-buttons {
    flex-direction: column;
    gap: 10px;
  }
  
  .action-buttons .el-button {
    width: 100%;
  }
}
</style>