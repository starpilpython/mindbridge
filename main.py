# FastApi 코드 가동하기 위한 코드

from fastapi import FastAPI
import uvicorn # 배포용 서버 
from routers import router as all_router  # 라우터 가져오기
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from routers.c5_converse.router import load_faces
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_faces()
    insert_dummy_member()
    yield  # 여기 이후는 shutdown 시점에 실행됨

# DB 구성하는 코드 
from DB.database import Base, engine, SessionLocal
from DB.models import ChatHistory,EmotionMessages,MemberList
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError


Base.metadata.create_all(bind=engine)  # 앱 시작할 때 테이블 자동 생성

app = FastAPI(lifespan=lifespan)

app.include_router(all_router)  # 라우터 포함시키기
app.mount("/statics", StaticFiles(directory="statics"), name="statics")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://sogang-03e137.gitlab.io"],     # 모든 출처(origin) 허용. 개발 단계에서 사용하며, 운영 환경에서는 실제 프론트 주소로 제한해야 함
    allow_credentials=True,  # 쿠키, 인증 헤더 등 자격 증명 포함 요청을 허용
    allow_methods=["*"],     # 모든 HTTP 메서드(GET, POST, PUT 등) 허용
    allow_headers=["*"],     # 모든 HTTP 헤더 허용
)

# 이 설정이 반드시 필요함
app.mount("/statics", StaticFiles(directory="/home/elicer/mindbridge/statics"))

# 여기에 더미 데이터를 넣는 함수 정의
def insert_dummy_member():
    db: Session = SessionLocal()
    try:
        # 중복 방지를 위해 존재 여부 확인
        exists = db.query(MemberList).filter_by(user_id="test001").first()
        if not exists:
            new_member = MemberList(
                user_id="test001",
                child_name="윤성필",
                audio='char1.mp3',
                character='호야',
                session_id = '1'
            )
            db.add(new_member)
            db.commit()
            print("더미 member 삽입 완료")
        else:
            print("이미 존재하는 member입니다")
    except IntegrityError:
        db.rollback()
        print("무결성 제약 조건 위반 발생")
    finally:
        db.close()

if __name__ == '__main__':
    # 배포용으로 설정 
    uvicorn.run(app,host='0.0.0.0',port=1025, log_level="info")

