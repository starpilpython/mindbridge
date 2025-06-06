from sqlalchemy import Column, Integer, String, Text, Date
from .database import Base
from datetime import date

class ChatHistory(Base):
    __tablename__ = "chat_history"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String(50), nullable=False)
    child_name = Column(String(50), nullable=False)  # ← 아이 이름 추가
    role = Column(String(10), nullable=False)  # 'user' or 'assistant'
    content = Column(Text, nullable=False)
    date = Column(Date, default=date.today)
