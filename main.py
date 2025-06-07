# FastApi 코드 가동하기 위한 코드

from fastapi import FastAPI
import uvicorn # 배포용 서버 
from routers import router as all_router  # 라우터 가져오기
from starlette.middleware.sessions import SessionMiddleware
from fastapi.middleware.cors import CORSMiddleware

# DB 구성하는 코드 
from DB.database import Base, engine
from DB.models import ChatHistory,EmotionMessages

Base.metadata.create_all(bind=engine)  # 앱 시작할 때 테이블 자동 생성


app = FastAPI()
app.add_middleware(SessionMiddleware, secret_key="your_secret_key") # 새션 추가 
app.include_router(all_router)  # 라우터 포함시키기

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],     # 모든 출처(origin) 허용. 개발 단계에서 사용하며, 운영 환경에서는 실제 프론트 주소로 제한해야 함
    allow_credentials=True,  # 쿠키, 인증 헤더 등 자격 증명 포함 요청을 허용
    allow_methods=["*"],     # 모든 HTTP 메서드(GET, POST, PUT 등) 허용
    allow_headers=["*"],     # 모든 HTTP 헤더 허용
)

if __name__ == '__main__':
    # 배포용으로 설정 
    uvicorn.run(app,host='0.0.0.0',port=1025, log_level="info")

