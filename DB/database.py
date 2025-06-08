from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

DATABASE_URL = "sqlite:///./chat.db"  # 예시: SQLite 사용 (경로는 필요에 따라 변경)

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    db = SessionLocal()  # 데이터베이스 세션 생성
    try:
        yield db  # DB 세션을 반환하여 사용하도록 함
    finally:
        db.close()  # 사용이 끝난 후 DB 세션 종료




