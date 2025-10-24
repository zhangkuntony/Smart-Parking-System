<template>
  <div id="app">
    <el-container style="height: 100vh;">
      <el-header style="height: auto; padding: 0;">
        <div class="header-content">
          <h1>智慧停车管理系统</h1>
          <el-menu
            :default-active="activeTab"
            mode="horizontal"
            @select="handleTabSelect"
            background-color="#409EFF"
            text-color="#fff"
            active-text-color="#ffd04b"
            style="min-height: 60px;"
          >
            <el-menu-item index="detection">车牌检测</el-menu-item>
            <el-menu-item index="management">车牌管理</el-menu-item>
          </el-menu>
        </div>
      </el-header>
      <el-main style="padding: 20px; overflow: auto;">
        <plate-detection v-if="activeTab === 'detection'" />
        <plate-management v-if="activeTab === 'management'" />
      </el-main>
    </el-container>
  </div>
</template>

<script>
import PlateDetection from './PlateDetection.vue'
import PlateManagement from './PlateManagement.vue'

export default {
  name: 'App',
  components: {
    PlateDetection,
    PlateManagement
  },
  data() {
    return {
      activeTab: 'detection'
    }
  },
  methods: {
    handleTabSelect(index) {
      this.activeTab = index
    }
  }
}
</script>

<style>
* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

html, body {
  height: 100%;
  margin: 0;
  padding: 0;
}

#app {
  font-family: Avenir, Helvetica, Arial, sans-serif;
  color: #2c3e50;
  height: 100vh;
  display: flex;
  flex-direction: column;
}

.header-content {
  background-color: #409EFF;
  color: white;
}

.header-content h1 {
  margin: 0;
  padding: 15px 0;
  text-align: center;
  font-size: 24px;
  font-weight: bold;
}

.el-menu {
  border-bottom: none;
  display: flex;
  justify-content: center;
}

.el-menu-item {
  font-size: 16px;
  font-weight: bold;
  height: 60px;
  line-height: 60px;
  padding: 0 30px;
}

.el-main {
  flex: 1;
  background-color: #f5f7fa;
}

/* 响应式设计 */
@media (max-width: 768px) {
  .header-content h1 {
    font-size: 20px;
    padding: 10px 0;
  }
  
  .el-menu-item {
    padding: 0 15px;
    font-size: 14px;
  }
  
  .el-main {
    padding: 10px;
  }
}
</style>