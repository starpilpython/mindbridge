# 아동의 대화를 대시보드화 하는 코드 

###################################################################

# Fastapi 라우터 설정하는 패키지
from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from datetime import date

# DB 불러오기 
from DB.models import ChatHistory, ChildShort
from DB.database import get_db
from sqlalchemy.orm import Session
from sqlalchemy import desc

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
    # 가장 최신 대화 세션 ID 조회
    latest_session = (
        db.query(ChatHistory)
        .order_by(desc(ChatHistory.id))
        .first()
    )
    session_id = latest_session.session_id
    name =latest_session.child_name
    child_id =latest_session.child_id
    make_date = latest_session.date

    # 해당 session_id에 해당하는 전체 대화 기록 불러오기
    if latest_session:
        chat_logs = (
            db.query(ChatHistory)
            .filter(ChatHistory.session_id == session_id)
            .order_by(ChatHistory.id)  # 시간 순 정렬
            .all()
        )

    # 템플릿 형식으로 변환
    dialogue = ""
    for log in chat_logs:
        prefix = "<|assistant|>" if log.role == "assistant" else "<|user|>"
        dialogue += f"{prefix}\n{log.content.strip()}\n"
    dialogue = dialogue.strip()  # 마지막 개행 제거
    dialogue += "\n<|user|>\n그 동안의 이야기만 요약만 해줘.\n"
    print(dialogue)

    # 결과반환 
    short_summary, text_list_summray = llm_short.short_opinion(dialogue)

    # 데이터 저장 
    new_entry = ChildShort(
        user_id="PARENT01",
        child_id=child_id,
        child_name=name,
        short_summary=short_summary,
        text_list_summray = text_list_summray,
        date=make_date
    )
    db.add(new_entry)
    db.commit()

    return JSONResponse({
        "child_name": name,
        "session_date": make_date,
        "opinion_summary": short_summary,
        "text_summary":text_list_summray,
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
            "short_data": item.short_data,
            "date": str(item.date)
        })

    return JSONResponse(content=result)















    