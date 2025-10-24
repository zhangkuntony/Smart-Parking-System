<template>
  <div id="app">
    <el-container>
      <el-header>
        <h1>车牌检测系统</h1>
      </el-header>
      <el-main>
        <el-card class="upload-card">
          <template #header>
            <div class="card-header">
              <span>上传车牌图片</span>
            </div>
          </template>
          <el-upload
            class="upload-demo"
            action=""
            :auto-upload="false"
            :on-change="handleFileUpload"
            :show-file-list="false"
          >
            <el-button type="primary">选择图片</el-button>
            <div v-if="imagePreview" class="preview-container">
              <img :src="imagePreview" alt="预览图" class="preview-image" />
            </div>
          </el-upload>
          <el-button 
            type="success" 
            @click="uploadImage" 
            :disabled="!imageFile"
            style="margin-top: 20px;"
          >
            检测车牌
          </el-button>
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
      </el-main>
    </el-container>
  </div>
</template>

<script>
export default {
  name: 'App',
  data() {
    return {
      imageFile: null,
      imagePreview: null,
      detectionResult: null
    }
  },
  methods: {
    handleFileUpload(file) {
      if (file) {
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
        alert('请先选择图片')
        return
      }

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
      } catch (error) {
        console.error('Error:', error)
        alert('车牌检测失败，请重试')
      }
    }
  }
}
</script>

<style>
#app {
  font-family: Avenir, Helvetica, Arial, sans-serif;
  text-align: center;
  color: #2c3e50;
}

.el-header {
  background-color: #409EFF;
  color: white;
  line-height: 60px;
}

.upload-card, .result-card {
  max-width: 800px;
  margin: 20px auto;
}

.card-header {
  font-size: 18px;
  font-weight: bold;
}

.preview-container {
  margin: 20px auto;
}

.preview-image {
  max-width: 100%;
  max-height: 300px;
  border-radius: 4px;
}
</style>