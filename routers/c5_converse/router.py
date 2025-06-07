# 아이와 AI가 실시간 대화하는 코드 
# 해당 코드에 실시간 감정 상태 저장하는 코드 추가

###################################################################
# Zonos 경로 불러오기 (패키지 로드하기 위함)

import sys
from pathlib import Path

# 기본 경로 설정 
BASE_DIR = Path(__file__).resolve().parent.parent.parent
ZONOS_PATH = BASE_DIR / 'AI_model' / 'Zonos'
sys.path.append(str(ZONOS_PATH))

from zonos.model import Zonos
from zonos.conditioning import make_cond_dict

# 위스퍼 패키지 로드 
from faster_whisper import WhisperModel
import torchaudio
import subprocess, shutil
from datetime import date

# Fastapi 라우터 설정하는 패키지
from fastapi import APIRouter, UploadFile, Request, Depends
from fastapi.responses import JSONResponse
from routers.c5_converse.livetalk import speech_to_text, ask_llm, text_to_speech
from routers.c4_call import zonos_model, whisper_model

# DB 불러오기 
from DB.models import ChatHistory
from DB.database import get_db
from sqlalchemy.orm import Session

###################################################################

# 각 음성 샘플 참조 데이터 및 결과 데이터 저장소 위치 
REFER_DIR = BASE_DIR / 'static' / 'refer_audio'
RESULT_DIR = BASE_DIR / 'static' / 'result_audio' / 'result.wav'

TMP_DIR = BASE_DIR / "tmp"
TMP_DIR.mkdir(exist_ok=True)

###################################################################

#Fastapi 가동 
router = APIRouter()

# 대화를 하고 이를 db에 정리하는 함수 
@router.post("/converse")
async def converse(request: Request, file: UploadFile = None, db: Session = Depends(get_db)):

    # 세션에 저장된 메세지 리스트를 불러옴. 만일 없으면 빈 리스트로 변환
    messages_list = request.session.get("messages", []).copy()

    # 사용자 ID와 이름을 세션에서 가져옴. 기본값은 "anonymous"로 설정
    user_id = request.session.get("user_id", "anonymous")
    user_name = request.session.get("user_name", "anonymous")

    # 음성 인식 실패 시에 반환할 기본 메시지 설정
    human_ask = "음성 인식이 잘 되지 않았어요."
    ai_answer = "다시 말해 주세요"

    # 업로드된 파일이 없을 경우 사용자에게 오류 메시지 표시
    if file is None:
        messages_list.append({"role": "user", "content": human_ask})
        messages_list.append({"role": "assistant", "content": ai_answer})
        request.session["messages"] = messages_list
        return JSONResponse({"audio_url": None, "text": ai_answer})

    # 업로드된 파일을 temp_input.webm으로 저장
    input_webm = TMP_DIR / "temp_input.webm"
    with open(input_webm, "wb") as f:
        f.write(await file.read())

    # temp_input.webm 파일을 temp_input.wav로 변환
    input_wav = TMP_DIR / "temp_input.wav"
    subprocess.run(["ffmpeg", "-y", "-i", str(input_webm), "-ar", "16000", "-ac", "1", str(input_wav)],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # 변환된 음성을 텍스트로 변환
    human_ask_ = speech_to_text(str(input_wav), whisper_model)

    # 변환된 텍스트가 비어있을 경우 사용자에게 오류 메시지를 추가
    if not human_ask_ or not human_ask_.strip():
        messages_list.append({"role": "user", "content": human_ask})
        messages_list.append({"role": "assistant", "content": ai_answer})
    else:
        messages_list.append({"role": "user", "content": human_ask_})
        try:
            # 대화형 언어 모델에 사용자의 질문을 전달하고 응답을 받음
            ai_answer = ask_llm(human_ask_, messages_list)
            messages_list.append({"role": "assistant", "content": ai_answer})

            # 대화 기록을 DB에 저장
            db.add(ChatHistory(user_id=user_id, user_name=user_name, role="user", content=human_ask_))
            db.add(ChatHistory(user_id=user_id, user_name=user_name, role="assistant", content=ai_answer))
            db.commit()  # 변경 사항을 커밋하여 DB에 저장

        except Exception as e:
            # 오류 발생 시, 오류를 출력하고 응답 메시지를 추가
            print(f"오류 발생: {e}")
            messages_list.append({"role": "assistant", "content": ai_answer})

    # 세션에서 참고할 파일 이름을 가져와 오디오 파일 생성
    refer_filename = request.session.get("audio", "narration.mp3")
    REFER = REFER_DIR / refer_filename
    output_file = text_to_speech(REFER, ai_answer, RESULT_DIR, zonos_model, make_cond_dict)

    # 생성된 오디오 파일을 지정된 위치로 복사
    static_audio_path = BASE_DIR / "static" / "audio" / output_file.name
    shutil.copy(output_file, static_audio_path)

    # 메세지 리스트를 세션에 업데이트하여 다음 대화에 사용
    request.session["messages"] = messages_list

    # 최종 응답 반환: 오디오 파일 URL과 AI의 응답 텍스트
    return JSONResponse({
        "audio_url": f"/static/audio/{output_file.name}",
        "text": ai_answer
    })

# 영상 받는 것을 통해 yolo로 얼굴 검출한 뒤에 감정 체크 하기  

