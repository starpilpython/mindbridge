# 아동의 대화를 대시보드 하는 코드  라우터 구성 

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
    # 감정 개별적으로 카운트 하기 
    emotion_str = emotion_rows[0].emotions  # ← 이게 가능하다면 가장 명확
    practices = emotion_str.split(" ")
    emo_list = [practice for practice in practices if practice != 'NO']
    emotion_counts = Counter(emo_list).most_common(7)
    top_counts = Counter(emo_list).most_common(3)
    # 감정 한국어로 매핑 
    emotion_kor_map = {
                "angry": "분노",
                "disgust": "혐오",
                "fear": "두려움",
                "happy": "행복",
                "sad": "슬픔",
                "surprise": "놀람",
                "neutral": "중립"
                }
    
    emotion_counts = [(emotion_kor_map[x],y) for x, y in emotion_counts]
    top_counts = [(emotion_kor_map[x],y) for x, y in top_counts] # 상위 3개

    # ChildShort 저장
    new_entry = ChildShort(
        user_id="PARENT01",
        child_id=child_id,
        child_name=name,
        short_summary=short_summary,
        text_list_summray=text_list_summray,
        emotion_counts=json.dumps(dict(emotion_counts)),  # 문자열로 저장
        top_counts=json.dumps(dict(top_counts)),  # 문자열로 저장
        date=make_date
    )
    db.add(new_entry)
    db.commit()

    return JSONResponse({
        "child_name": name,
        "session_date": make_date.isoformat(), 
        "opinion_summary": short_summary,
        "text_summary": text_list_summray,
        "emotion_counts": dict(emotion_counts),  # 응답에도 포함
        "top_counts": dict(top_counts)  # 응답에도 포함
    })


# 해당 데이터 베이스 조회 
@router.get("/child_summary_list", response_class=JSONResponse)
async def get_child_summary_list(db: Session = Depends(get_db)):
    summaries = db.query(ChildShort).order_by(ChildShort.date.desc()).all()

    result = []
    for item in summaries:
        # 감정 문자열 파싱 안전하게 처리
        try:
            emotions = json.loads(item.emotion_counts or "{}")
            top_counts = json.loads(item.top_counts or "{}")
        except (json.JSONDecodeError, TypeError):
            emotions = {}  # ← JSON 아니거나 잘못된 경우 빈 dict로
            top_counts = {}  # ← JSON 아니거나 잘못된 경우 빈 dict로

        result.append({
            "id": item.id,
            "user_id": item.user_id,
            "child_id": item.child_id,
            "child_name": item.child_name,
            "date": str(item.date),
            "emotion_counts": emotions,
            "top_counts": top_counts,  
            "opinion_summary": item.short_summary, 
            "text_summary": item.text_list_summray,
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
    top_emotions = emotion_counts.most_common(3)  # 상위 3개 감정

    return JSONResponse(content=dict(top_emotions))


# 세부 대시보드를 만들기 
@router.get("/child_summary_detail", response_class=JSONResponse)
async def get_child_summary_detail(date: str, db: Session = Depends(get_db)):
    item = db.query(ChildShort).filter(ChildShort.date == date).first()
    if not item:
        return JSONResponse(status_code=404, content={"error": "No data found"})

    try:
        emotions = json.loads(item.emotion_counts or "{}")
        top_counts = json.loads(item.top_counts or "{}")
    except (json.JSONDecodeError, TypeError):
        emotions = {}

    return JSONResponse(content={
        "child_name": item.child_name,
        "date": str(item.date),
        "emotion_counts": emotions,
        'top_counts':top_counts,
        "opinion_summary": item.short_summary,
        "text_summary": item.text_list_summray
    })
