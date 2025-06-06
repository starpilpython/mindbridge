# [아동 사이드] 캐릭터 선택창 라우터 설정

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from starlette.middleware.sessions import SessionMiddleware

router = APIRouter()

# post 방식 주소 뒤에 /c3_char_select 부여
# 캐릭터 선택창 비동기로 진행 
@router.post("/c3_char_select")
async def select_character(request: Request):
    data = await request.json()
    character = data.get("character")
    audio = data.get("audio")

    # 세션 저장
    request.session["character"] = character
    request.session["audio"] = audio

    print(f"선택된 캐릭터: {character}, 오디오 파일: {audio}")

    return JSONResponse({"redirect_url": "/frontend/c_4_call.html"})


