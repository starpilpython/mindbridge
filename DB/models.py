# DB 구성하는 코드 
from sqlalchemy import Column, Integer, String, Text, Date
from .database import Base
from datetime import date

# 아동-AI 대화 내용 기록 DB
class ChatHistory(Base):
    # 테이블 이름 설정
    __tablename__ = "chat_history"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String(50), nullable=False)
    child_name = Column(String(50), nullable=False)  # 아이 이름 추가
    role = Column(String(10), nullable=False)  # 'user' or 'assistant'
    content = Column(Text, nullable=False)
    date = Column(Date, default=date.today)
    session_id = Column(String(36), nullable=False)  # 세션 ID 추가

# 아동-AI 웹캠 통한 감정 기록 DB
class EmotionMessages(Base):
    # 테이블 이름 설정
    __tablename__ = "emotion_messages"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String(50), nullable=False)
    child_name = Column(String(50), nullable=False)  # 아이 이름 추가
    emotions = Column(Text, nullable=False) # 감정 기록
    date = Column(Date, default=date.today)
    session_id = Column(String(36), nullable=False)  # 세션 ID 추가
