
# 대화 마무리된 후에 애니메이션 영상 생성 

###################################################################

# Fastapi 라우터 설정하는 패키지
from fastapi import APIRouter, UploadFile, Request, Depends
from fastapi.responses import JSONResponse
from routers.c6_webtoon import background_generator, character_pipeline, character_tts, make_template_rag, make_webtoon

# DB 불러오기 
from DB.models import ChatHistory
from DB.database import get_db
from sqlalchemy.orm import Session
from sqlalchemy import desc

###################################################################

#Fastapi 가동 
router = APIRouter()

@router.post("/webtoon")
async def webtoon(db: Session = Depends(get_db)):

    '''기존에 받은 내용 → 적합한 탬플릿 형식으로 바꾸기 '''
    # 가장 최신 대화 세션 ID 조회
    latest_session = (
        db.query(ChatHistory.session_id)
        .order_by(desc(ChatHistory.id))
        .first()
        )
    
    # 해당 session_id에 해당하는 전체 대화 기록 불러오기
    if latest_session:
        session_id = latest_session.session_id
        chat_logs = (
            db.query(ChatHistory)
            .filter(ChatHistory.session_id == session_id)
            .order_by(ChatHistory.id)  # 시간 순 정렬
            .all()
        )

 


