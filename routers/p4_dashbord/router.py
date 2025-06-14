# 아동의 대화를 대시보드화 하는 코드 

###################################################################

# Fastapi 라우터 설정하는 패키지
from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from datetime import date

# DB 불러오기 
from DB.models import ChatHistory, ChildShort,EmotionMessages
from DB.database import get_db
from sqlalchemy.orm import Session
from sqlalchemy import desc
from collections import Counter
import json

# 핵심 기능 호출 
from routers.p4_dashbord import llm_short
###################################################################

#Fastapi 가동 
router = APIRouter()
today = date.today()

# 당일 요약 데이터 생성 코드 
@router.get('/lastest_short')
async def lastest_short(db: Session = Depends(get_db)):
    '''기존에 받은 내용 → 적합한 탬플릿 형식으로 바꾸기 '''
    latest_session = (
        db.query(ChatHistory)
        .order_by(desc(ChatHistory.id))
        .first()
    )
    session_id = latest_session.session_id
    name = latest_session.child_name
    child_id = latest_session.user_id
    make_date = latest_session.date

    # 대화 로그 수집
    if latest_session:
        chat_logs = (
            db.query(ChatHistory)
            .filter(ChatHistory.session_id == session_id)
            .order_by(ChatHistory.id)
            .all()
        )

    dialogue = ""
    for log in chat_logs:
        prefix = "<|assistant|>" if log.role == "assistant" else "<|user|>"
        dialogue += f"{prefix}\n{log.content.strip()}\n"
    dialogue = dialogue.strip()
    dialogue += "\n<|user|>\n그 동안의 이야기만 요약만 해줘.\n"

    # 요약 결과
    short_summary, text_list_summray = llm_short.short_opinion(dialogue)

    # 감정 기록 수집 및 집계
    emotion_rows = (
        db.query(EmotionMessages.emotions)
        .filter(EmotionMessages.session_id == session_id)
        .all()
    )

    emotion_list = []
    for row in emotion_rows:
        if row[0]:
            emotion_list.extend(row[0].strip().split())

    emotion_counts = Counter(emotion_list)

    # ChildShort 저장
    new_entry = ChildShort(
        user_id="PARENT01",
        child_id=child_id,
        child_name=name,
        short_summary=short_summary,
        text_list_summray=text_list_summray,
        emotion_counts=json.dumps(dict(emotion_counts)),  # 문자열로 저장
        date=make_date
    )
    db.add(new_entry)
    db.commit()

    return JSONResponse({
        "child_name": name,
        "session_date": make_date,
        "opinion_summary": short_summary,
        "text_summary": text_list_summray,
        "emotion_counts": dict(emotion_counts)  # 응답에도 포함
    })


# 해당 데이터 베이스 조회 
@router.get("/child_summary_list", response_class=JSONResponse)
async def get_child_summary_list(db: Session = Depends(get_db)):
    summaries = db.query(ChildShort).order_by(ChildShort.date.desc()).all()

    result = []
    for item in summaries:
        result.append({
            "id": item.id,
            "user_id": item.user_id,
            "child_id": item.child_id,
            "child_name": item.child_name,
            "date": str(item.date)
        })

    return JSONResponse(content=result)


# 감정데이터 호출 
@router.get("/emotion_summary", response_class=JSONResponse)
async def get_emotion_summary(db: Session = Depends(get_db)):
    """
    감정 문자열을 공백 기준으로 분리하여 감정별 총 출현 횟수 반환
    """
    results = db.query(EmotionMessages.emotions).all()

    emotion_list = []
    for row in results:
        emotions = row[0]
        if emotions:
            emotion_list.extend(emotions.strip().split())

    emotion_counts = Counter(emotion_list)
    top_emotions = emotion_counts.most_common(10)  # 상위 10개 감정

    return JSONResponse(content=dict(top_emotions))