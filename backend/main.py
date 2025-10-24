from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api import setup_routes

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

# 设置API路由
setup_routes(app)

# 启动应用
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)