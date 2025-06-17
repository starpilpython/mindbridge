
# 대화 마무리된 후에 애니메이션 영상 생성 

###################################################################

# Fastapi 라우터 설정하는 패키지
from fastapi import APIRouter, UploadFile, Request, Depends
from fastapi.responses import JSONResponse
from routers.c6_webtoon import background_generator, character_pipeline, character_tts, make_template_ra1g, make_webtoon
import json

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
        db.query(ChatHistory)
        .order_by(desc(ChatHistory.id))
        .first()
    )

    session_id = latest_session.session_id
    name =latest_session.child_name
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

    # 웹툰 코드에 넣을 형식으로 변환 
    retrieved = make_template_ra1g.retrieve_documents(dialogue)
    story = make_template_ra1g.ask_llm_with_context(dialogue, retrieved)
    try:
        story = make_template_ra1g.clean_json_output(story)
        story = json.loads(story)
    except json.JSONDecodeError as e:
        print("JSON 파싱 오류:", e)
        print("원본 응답:\n", story)
        exit(1)

    print("== 생성된 이야기 ==")
    prompt = make_template_ra1g.convert_to_prompt_format(story)

    '''웹툰 생성 하기'''
    background_generator.generate_all_backgrounds(prompt)
    character_pipeline.character_main(prompt)
    character_tts.tts_main(prompt)
    make_webtoon.webtoon_main(prompt,f"{name}_{make_date}")

    return JSONResponse({"ok": True})
