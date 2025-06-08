# [아동 사이드] 캐릭터 선택창 라우터 설정

from fastapi import APIRouter, Request, Depends
from DB.models import MemberList
from DB.database import get_db
from sqlalchemy.orm import Session
import uuid

router = APIRouter()

def upsert_member(db, user_id: str, child_name: str, character: str, audio: str):
    existing = db.query(MemberList).filter_by(user_id=user_id).first()

    if existing:
        # 존재하면 업데이트
        existing.child_name = child_name
        existing.character = character
        existing.audio = audio
        existing.session_id = str(uuid.uuid4())
    else:
        # 존재하지 않으면 삽입
        new_data = MemberList(
            user_id=user_id,
            child_name=child_name,
            character=character,
            audio=audio,
            session_id = str(uuid.uuid4())
        )
        db.add(new_data)

    db.commit()

# post 방식 주소 뒤에 /c3_char_select 부여
# 캐릭터 선택창 비동기로 진행
@router.post("/c3_char_select")
async def select_character(request: Request, db: Session = Depends(get_db)):
    data = await request.json()
    character = data.get("character")
    audio = data.get("audio")


    upsert_member(
        db=db,
        user_id="test001",
        child_name="윤성필",
        character=character,
        audio=audio
    )

    print(f"선택된 캐릭터: {character}, 오디오 파일: {audio}")

    return {"ok": True}


