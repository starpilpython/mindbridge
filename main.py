# FastApi 코드 가동하기 위한 코드

from fastapi import FastAPI
import uvicorn # 배포용 서버 
from routers import router as item_router  # 라우터 가져오기
from starlette.middleware.sessions import SessionMiddleware

app = FastAPI()
app.add_middleware(SessionMiddleware, secret_key="your_secret_key") # 새션 추가 
app.include_router(item_router)  # 라우터 포함시키기

if __name__ == '__main__':
    # 배포용으로 설정 
    uvicorn.run(app,host='0.0.0.0',port=1025, log_level="info")
