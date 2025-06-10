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


# 아동 데이터 인적사항 추가 
class MemberList(Base):
    # 테이블 이름 설정
    __tablename__ = "member_list"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String(50), nullable=False)
    child_name = Column(String(50), nullable=False)  # 아이 이름 추가
    character = Column(String(50), nullable=True)
    audio = Column(String(50), nullable=True)  # 아이 이름 추가
    session_id = Column(String(36), nullable=True)  # 세션 ID 추가


# 아동 요약 데이터(부모 사이드) 추가 
class ChildShort(Base):
    # 테이블 이름 설정
    __tablename__ = "child_short"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String(50), nullable=False) # 부모 아이디 
    child_id = Column(String(50), nullable=False) # 아동 아이디 
    child_name = Column(String(50), nullable=False)  # 아이 이름
    short_summary = Column(String(50000),default="아직 생성되지 않았습니다.") # 소견 데이터 
    text_list_summray = Column(String(50000),default="아직 생성되지 않았습니다.") # 요약데이터 
    date = Column(Date, default=date.today)
    

