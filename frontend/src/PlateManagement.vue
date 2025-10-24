<template>
  <div class="management-container">
    <el-card class="management-card" shadow="hover">
      <template #header>
        <div class="card-header">
          <div class="header-content">
            <h2 class="title">
              <el-icon><management /></el-icon>
              车牌管理
            </h2>
            <div class="header-actions">
              <el-button type="success" @click="showAddDialog" size="large">
                <el-icon><plus /></el-icon>
                添加车辆
              </el-button>
              <el-button type="primary" @click="refreshData" size="large">
                <el-icon><refresh /></el-icon>
                刷新数据
              </el-button>
            </div>
          </div>
        </div>
      </template>
      
      <!-- 搜索和筛选区域 -->
      <div class="filter-section">
        <div class="filter-row">
          <div class="filter-left">
            <el-input
              v-model="searchKeyword"
              placeholder="搜索车牌号码"
              size="large"
              style="width: 280px;"
              @input="handleSearch"
              clearable
            >
              <template #prefix>
                <el-icon><search /></el-icon>
              </template>
            </el-input>
            
            <el-select 
              v-model="filterStatus" 
              placeholder="筛选状态" 
              size="large"
              style="width: 180px; margin-left: 16px;"
              @change="handleFilter"
            >
              <el-option label="全部状态" value=""></el-option>
              <el-option label="有效" value="valid"></el-option>
              <el-option label="无效" value="invalid"></el-option>
            </el-select>
          </div>
          
          <div class="filter-right">
            <span class="total-count">共 {{ totalPlates }} 条记录</span>
          </div>
        </div>
      </div>

      <!-- 车牌数据表格 -->
      <div class="table-container">
        <el-table 
          :data="filteredPlates" 
          :border="true"
          :stripe="true"
          :highlight-current-row="true"
          empty-text="暂无数据"
          class="data-table"
        >
          <el-table-column type="index" label="序号" width="80" align="center"></el-table-column>
          <el-table-column prop="plate_number" label="车牌号码" min-width="120" align="center">
            <template #default="scope">
              <span class="plate-number">{{ scope.row.plate_number }}</span>
            </template>
          </el-table-column>
          <el-table-column prop="owner_name" label="车主姓名" min-width="120" align="center">
            <template #default="scope">
              <span>{{ scope.row.owner_name || '-' }}</span>
            </template>
          </el-table-column>
          <el-table-column prop="valid_to" label="有效期至" min-width="120" align="center"></el-table-column>
          <el-table-column prop="is_valid" label="状态" min-width="100" align="center">
            <template #default="scope">
              <el-tag 
                :type="scope.row.is_valid ? 'success' : 'danger'"
                size="large"
                effect="light"
              >
                {{ scope.row.is_valid ? '有效' : '无效' }}
              </el-tag>
            </template>
          </el-table-column>
          <el-table-column prop="detection_time" label="创建时间" min-width="160" align="center"></el-table-column>
          <el-table-column label="操作" min-width="220" align="center" fixed="right">
            <template #default="scope">
              <div class="action-buttons">
                <el-button 
                  type="primary" 
                  size="small" 
                  @click="viewDetails(scope.row)"
                  link
                >
                  <el-icon><view /></el-icon>
                  查看详情
                </el-button>
                <el-button 
                  type="warning" 
                  size="small" 
                  @click="showEditDialog(scope.row)"
                  link
                >
                  <el-icon><edit /></el-icon>
                  修改
                </el-button>
                <el-button 
                  type="danger" 
                  size="small" 
                  @click="deletePlate(scope.row)"
                  link
                >
                  <el-icon><delete /></el-icon>
                  删除
                </el-button>
              </div>
            </template>
          </el-table-column>
        </el-table>
      </div>

      <!-- 分页 -->
      <div class="pagination-section">
        <el-pagination
          v-model:current-page="currentPage"
          v-model:page-size="pageSize"
          :page-sizes="[10, 20, 50, 100]"
          :total="totalPlates"
          :layout="'total, sizes, prev, pager, next, jumper'"
          @size-change="handleSizeChange"
          @current-change="handleCurrentChange"
        />
      </div>
    </el-card>

    <!-- 详情对话框 -->
    <el-dialog 
      v-model="detailDialogVisible" 
      title="车牌详情" 
      width="600px"
      center
    >
      <el-descriptions v-if="currentPlate" :column="2" border>
        <el-descriptions-item label="车牌号码" :span="2">
          <span class="detail-value">{{ currentPlate.plate_number }}</span>
        </el-descriptions-item>
        <el-descriptions-item label="车主姓名" :span="2">
          <span class="detail-value">{{ currentPlate.owner_name || '未填写' }}</span>
        </el-descriptions-item>
        <el-descriptions-item label="有效期至">
          <span class="detail-value">{{ currentPlate.valid_to }}</span>
        </el-descriptions-item>
        <el-descriptions-item label="状态">
          <el-tag :type="currentPlate.is_valid ? 'success' : 'danger'" size="large">
            {{ currentPlate.is_valid ? '有效' : '无效' }}
          </el-tag>
        </el-descriptions-item>
        <el-descriptions-item label="创建时间">
          <span class="detail-value">{{ currentPlate.detection_time }}</span>
        </el-descriptions-item>
      </el-descriptions>
      <template #footer>
        <el-button @click="detailDialogVisible = false" size="large">关闭</el-button>
      </template>
    </el-dialog>

    <!-- 新增车牌对话框 -->
    <el-dialog 
      v-model="addDialogVisible" 
      title="新增车牌" 
      width="500px"
      center
      @close="resetAddForm"
    >
      <el-form 
        ref="addFormRef" 
        :model="addForm" 
        :rules="addFormRules" 
        label-width="100px"
        label-position="left"
      >
        <el-form-item label="车牌号码" prop="plate_number">
          <el-input 
            v-model="addForm.plate_number" 
            placeholder="请输入车牌号码"
            size="large"
            maxlength="10"
            show-word-limit
          />
        </el-form-item>
        
        <el-form-item label="车主姓名" prop="owner_name">
          <el-input 
            v-model="addForm.owner_name" 
            placeholder="请输入车主姓名"
            size="large"
            maxlength="20"
            show-word-limit
          />
        </el-form-item>
        
        <el-form-item label="有效期至" prop="valid_to">
          <el-date-picker
            v-model="addForm.valid_to"
            type="date"
            placeholder="请选择有效期"
            size="large"
            style="width: 100%"
            :disabled-date="disabledDate"
            value-format="YYYY-MM-DD"
          />
        </el-form-item>
        
        <el-form-item label="状态">
          <el-radio-group v-model="addForm.is_valid">
            <el-radio :label="true">有效</el-radio>
            <el-radio :label="false">无效</el-radio>
          </el-radio-group>
        </el-form-item>
      </el-form>
      
      <template #footer>
        <div class="dialog-footer">
          <el-button @click="addDialogVisible = false" size="large">取消</el-button>
          <el-button 
            type="primary" 
            @click="submitAddForm" 
            size="large"
            :loading="addLoading"
          >
            确认添加
          </el-button>
        </div>
      </template>
    </el-dialog>

    <!-- 修改车牌对话框 -->
    <el-dialog 
      v-model="editDialogVisible" 
      title="修改车牌信息" 
      width="500px"
      center
      @close="resetEditForm"
    >
      <el-form 
        ref="editFormRef" 
        :model="editForm" 
        label-width="100px"
        label-position="left"
      >
        <el-form-item label="车牌号码">
          <el-input 
            v-model="editForm.plate_number" 
            placeholder="车牌号码"
            size="large"
            disabled
          />
        </el-form-item>
        
        <el-form-item label="车主姓名">
          <el-input 
            v-model="editForm.owner_name" 
            placeholder="车主姓名"
            size="large"
            disabled
          />
        </el-form-item>
        
        <el-form-item label="有效期至" prop="valid_to">
          <el-date-picker
            v-model="editForm.valid_to"
            type="date"
            placeholder="请选择有效期"
            size="large"
            style="width: 100%"
            :disabled-date="disabledDate"
            value-format="YYYY-MM-DD"
          />
        </el-form-item>
        
        <el-form-item label="状态">
          <el-radio-group v-model="editForm.is_valid">
            <el-radio :label="true">有效</el-radio>
            <el-radio :label="false">无效</el-radio>
          </el-radio-group>
        </el-form-item>
      </el-form>
      
      <template #footer>
        <div class="dialog-footer">
          <el-button @click="editDialogVisible = false" size="large">取消</el-button>
          <el-button 
            type="primary" 
            @click="submitEditForm" 
            size="large"
            :loading="editLoading"
          >
            确认修改
          </el-button>
        </div>
      </template>
    </el-dialog>

    <!-- 修改车牌对话框 -->
    <el-dialog 
      v-model="editDialogVisible" 
      title="修改车牌信息" 
      width="500px"
      center
      @close="resetEditForm"
    >
      <el-form 
        ref="editFormRef" 
        :model="editForm" 
        label-width="100px"
        label-position="left"
      >
        <el-form-item label="车牌号码">
          <el-input 
            v-model="editForm.plate_number" 
            placeholder="车牌号码"
            size="large"
            disabled
          />
        </el-form-item>
        
        <el-form-item label="车主姓名">
          <el-input 
            v-model="editForm.owner_name" 
            placeholder="车主姓名"
            size="large"
            disabled
          />
        </el-form-item>
        
        <el-form-item label="有效期至" prop="valid_to">
          <el-date-picker
            v-model="editForm.valid_to"
            type="date"
            placeholder="请选择有效期"
            size="large"
            style="width: 100%"
            :disabled-date="disabledDate"
            value-format="YYYY-MM-DD"
          />
        </el-form-item>
        
        <el-form-item label="状态">
          <el-radio-group v-model="editForm.is_valid">
            <el-radio :label="true">有效</el-radio>
            <el-radio :label="false">无效</el-radio>
          </el-radio-group>
        </el-form-item>
      </el-form>
      
      <template #footer>
        <div class="dialog-footer">
          <el-button @click="editDialogVisible = false" size="large">取消</el-button>
          <el-button 
            type="primary" 
            @click="submitEditForm" 
            size="large"
            :loading="editLoading"
          >
            确认修改
          </el-button>
        </div>
      </template>
    </el-dialog>
  </div>
</template>

<script>
import { Management, Refresh, Search, View, Delete, Plus, Edit } from '@element-plus/icons-vue'

export default {
  name: 'PlateManagement',
  components: {
    Management,
    Refresh,
    Search,
    View,
    Delete,
    Plus,
    Edit
  },
  data() {
    return {
      plates: [],
      searchKeyword: '',
      filterStatus: '',
      currentPage: 1,
      pageSize: 10,
      totalPlates: 0,
      detailDialogVisible: false,
      currentPlate: null,
      addDialogVisible: false,
      addForm: {
        plate_number: '',
        owner_name: '',
        valid_to: '',
        is_valid: true
      },
      addFormRules: {
        plate_number: [
          { required: true, message: '请输入车牌号码', trigger: 'blur' },
          { min: 6, max: 10, message: '车牌号码长度应为6-10个字符', trigger: 'blur' }
        ],
        owner_name: [
          { required: true, message: '请输入车主姓名', trigger: 'blur' },
          { min: 2, max: 20, message: '车主姓名长度应为2-20个字符', trigger: 'blur' }
        ],
        valid_to: [
          { required: true, message: '请选择有效期', trigger: 'change' }
        ]
      },
      addLoading: false,
      editDialogVisible: false,
      editForm: {
        id: null,
        plate_number: '',
        owner_name: '',
        valid_to: '',
        is_valid: true
      },
      editLoading: false
    }
  },
  computed: {
    filteredPlates() {
      let filtered = this.plates
      
      // 根据搜索关键词过滤
      if (this.searchKeyword) {
        filtered = filtered.filter(plate => 
          plate.plate_number.toLowerCase().includes(this.searchKeyword.toLowerCase())
        )
      }
      
      // 根据状态过滤
      if (this.filterStatus) {
        filtered = filtered.filter(plate => 
          this.filterStatus === 'valid' ? plate.is_valid : !plate.is_valid
        )
      }
      
      // 分页
      this.totalPlates = filtered.length
      const start = (this.currentPage - 1) * this.pageSize
      const end = start + this.pageSize
      return filtered.slice(start, end)
    }
  },
  mounted() {
    this.loadPlates()
  },
  methods: {
    async loadPlates() {
      try {
        const response = await fetch('http://localhost:8000/plates/')
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`)
        }
        const data = await response.json()
        
        // 转换数据格式以匹配前端显示
        this.plates = data.map(plate => ({
          id: plate.id,
          plate_number: plate.plate_number,
          valid_to: plate.valid_to,
          is_valid: plate.is_active && new Date(plate.valid_to) >= new Date(),
          detection_time: this.formatDateTime(plate.created_at),
          image_path: '/images/default_plate.jpg',
          owner_name: plate.owner_name,
          valid_from: plate.valid_from
        }))
        
        this.totalPlates = this.plates.length
      } catch (error) {
        console.error('加载车牌数据失败:', error)
        this.$message.error('加载数据失败')
      }
    },
    formatDateTime(dateTimeString) {
      if (!dateTimeString) return ''
      
      const date = new Date(dateTimeString)
      if (isNaN(date.getTime())) return dateTimeString
      
      // 格式化为：YYYY-MM-DD HH:mm:ss
      const year = date.getFullYear()
      const month = String(date.getMonth() + 1).padStart(2, '0')
      const day = String(date.getDate()).padStart(2, '0')
      const hours = String(date.getHours()).padStart(2, '0')
      const minutes = String(date.getMinutes()).padStart(2, '0')
      const seconds = String(date.getSeconds()).padStart(2, '0')
      
      return `${year}-${month}-${day} ${hours}:${minutes}:${seconds}`
    },
    handleSearch() {
      this.currentPage = 1
    },
    handleFilter() {
      this.currentPage = 1
    },
    handleSizeChange(newSize) {
      this.pageSize = newSize
      this.currentPage = 1
    },
    handleCurrentChange(newPage) {
      this.currentPage = newPage
    },
    refreshData() {
      this.loadPlates()
      this.$message.success('数据已刷新')
    },
    viewDetails(plate) {
      this.currentPlate = plate
      this.detailDialogVisible = true
    },
    deletePlate(plate) {
      this.$confirm(`确定要删除车牌 ${plate.plate_number} 吗？`, '提示', {
        confirmButtonText: '确定',
        cancelButtonText: '取消',
        type: 'warning'
      }).then(async () => {
        try {
          // 调用后端API删除数据
          const response = await fetch(`http://localhost:8000/plates/${plate.id}`, { 
            method: 'DELETE' 
          })
          
          if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`)
          }
          
          // 删除成功后重新加载数据
          await this.loadPlates()
          this.$message.success('删除成功')
        } catch (error) {
          console.error('删除失败:', error)
          this.$message.error('删除失败')
        }
      }).catch(() => {
        this.$message.info('已取消删除')
      })
    },
    showAddDialog() {
      this.addDialogVisible = true
    },
    resetAddForm() {
      this.addForm = {
        plate_number: '',
        owner_name: '',
        valid_to: '',
        is_valid: true
      }
      this.$nextTick(() => {
        if (this.$refs.addFormRef) {
          this.$refs.addFormRef.clearValidate()
        }
      })
    },
    disabledDate(time) {
      // 禁止选择今天之前的日期
      return time.getTime() < Date.now() - 8.64e7
    },
    async submitAddForm() {
      if (!this.$refs.addFormRef) return
      
      try {
        const valid = await this.$refs.addFormRef.validate()
        if (!valid) return
        
        this.addLoading = true
        
        try {
          // 准备请求数据，匹配后端API的数据结构
          const requestData = {
            plate_number: this.addForm.plate_number,
            owner_name: this.addForm.owner_name || '', // 使用用户输入的业主姓名
            valid_to: this.addForm.valid_to,
            valid_from: new Date().toISOString().split('T')[0], // 使用今天作为开始日期
            is_active: this.addForm.is_valid
          }
          
          // 调用后端API添加数据
          const response = await fetch('http://localhost:8000/plates/', {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestData)
          })
          
          if (!response.ok) {
            const errorData = await response.json()
            throw new Error(errorData.detail || `HTTP error! status: ${response.status}`)
          }
          
          // 添加成功后重新加载数据
          await this.loadPlates()
          this.addDialogVisible = false
          this.$message.success('车牌添加成功')
          
          // 重置搜索和分页
          this.searchKeyword = ''
          this.filterStatus = ''
          this.currentPage = 1
          
        } catch (error) {
          console.error('添加失败:', error)
          this.$message.error(`添加失败: ${error.message}`)
        }
        
      } catch (error) {
        console.error('添加失败:', error)
        this.$message.error('添加失败，请重试')
      } finally {
        this.addLoading = false
      }
    },
    showEditDialog(plate) {
      this.editForm = {
        id: plate.id,
        plate_number: plate.plate_number,
        owner_name: plate.owner_name || '',
        valid_to: plate.valid_to,
        is_valid: plate.is_valid
      }
      this.editDialogVisible = true
    },
    resetEditForm() {
      this.editForm = {
        id: null,
        plate_number: '',
        owner_name: '',
        valid_to: '',
        is_valid: true
      }
      this.$nextTick(() => {
        if (this.$refs.editFormRef) {
          this.$refs.editFormRef.clearValidate()
        }
      })
    },
    async submitEditForm() {
      if (!this.$refs.editFormRef) return
      
      try {
        this.editLoading = true
        
        // 准备请求数据，匹配后端API的数据结构
        const requestData = {
          plate_number: this.editForm.plate_number,
          owner_name: this.editForm.owner_name || '',
          valid_to: this.editForm.valid_to,
          valid_from: new Date().toISOString().split('T')[0], // 使用今天作为开始日期
          is_active: this.editForm.is_valid
        }
        
        // 调用后端API修改数据
        const response = await fetch(`http://localhost:8000/plates/${this.editForm.id}`, {
          method: 'PUT',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify(requestData)
        })
        
        if (!response.ok) {
          const errorData = await response.json()
          throw new Error(errorData.detail || `HTTP error! status: ${response.status}`)
        }
        
        // 修改成功后重新加载数据
        await this.loadPlates()
        this.editDialogVisible = false
        this.$message.success('车牌信息修改成功')
        
      } catch (error) {
        console.error('修改失败:', error)
        this.$message.error(`修改失败: ${error.message}`)
      } finally {
        this.editLoading = false
      }
    }
  }
}
</script>

<style scoped>
.management-container {
  padding: 16px 12px;
  min-height: 100vh;
}

.management-card {
  min-height: 500px;
  max-height: 80vh;
  display: flex;
  flex-direction: column;
  border-radius: 12px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
  position: relative;
}

.management-card :deep(.el-card__body) {
  flex: 1;
  display: flex;
  flex-direction: column;
  padding: 20px;
}

.management-card :deep(.el-card__body) {
  flex: 1;
  display: flex;
  flex-direction: column;
  padding: 20px;
}

.card-header {
  padding: 0;
}

.header-content {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0 8px;
}

.title {
  display: flex;
  align-items: center;
  gap: 8px;
  margin: 0;
  font-size: 20px;
  font-weight: 600;
  color: #303133;
}

.header-actions {
  display: flex;
  gap: 12px;
}

.filter-section {
  margin-bottom: 16px;
  padding: 0 2px;
}

.content-wrapper {
  flex: 1;
  display: flex;
  flex-direction: column;
  min-height: 0;
  height: 100%;
}

.content-wrapper {
  flex: 1;
  display: flex;
  flex-direction: column;
  min-height: 0;
  height: 100%;
}

.filter-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  flex-wrap: wrap;
  gap: 16px;
}

.filter-left {
  display: flex;
  align-items: center;
  gap: 16px;
  flex-wrap: wrap;
}

.filter-right {
  display: flex;
  align-items: center;
}

.total-count {
  font-size: 14px;
  color: #606266;
  font-weight: 500;
}

.table-container {
  flex: 1;
  overflow: auto;
  margin-bottom: 0;
  min-height: 0;
}

.data-table {
  width: 100%;
  height: 100%;
  border-radius: 8px;
  overflow: hidden;
}

.data-table :deep(.el-table__header-wrapper) {
  background-color: #f5f7fa;
}

.data-table :deep(.el-table__header th) {
  background-color: #f5f7fa;
  color: #303133;
  font-weight: 600;
  font-size: 13px;
  padding: 8px 4px;
}

.data-table :deep(.el-table__body tr:hover > td) {
  background-color: #f5f7fa !important;
}

.data-table :deep(.el-table__body td) {
  padding: 8px 4px;
  font-size: 13px;
}

.data-table :deep(.el-table__body td) {
  padding: 8px 4px;
  font-size: 13px;
}

.data-table :deep(.el-table__body td) {
  padding: 8px 4px;
  font-size: 13px;
}

.plate-number {
  font-weight: 600;
  color: #409eff;
  font-size: 14px;
}

.action-buttons {
  display: flex;
  gap: 8px;
  justify-content: center;
}

.pagination-section {
  margin-top: auto;
  padding: 16px 0 0 0;
  border-top: 1px solid #ebeef5;
  flex-shrink: 0;
}

.pagination-section :deep(.el-pagination) {
  justify-content: center;
}

.detail-value {
  font-weight: 500;
  color: #303133;
}

/* 响应式设计 */
@media (max-width: 768px) {
  .management-container {
    padding: 12px;
  }
  
  .header-content {
    flex-direction: column;
    gap: 16px;
    align-items: stretch;
  }
  
  .filter-row {
    flex-direction: column;
    align-items: stretch;
  }
  
  .filter-left {
    justify-content: center;
  }
  
  .filter-right {
    justify-content: center;
  }
  
  .data-table {
    font-size: 12px;
  }
  
  .action-buttons {
    flex-direction: column;
    gap: 4px;
  }
}

@media (max-width: 480px) {
  .filter-left {
    flex-direction: column;
    align-items: stretch;
  }
  
  .filter-left .el-input,
  .filter-left .el-select {
    width: 100% !important;
    margin-left: 0 !important;
  }
}
</style>