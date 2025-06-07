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


from DB.models import ChatHistory

###################################################################

# 각 음성 샘플 참조 데이터 및 결과 데이터 저장소 위치 
REFER_DIR = BASE_DIR / 'static' / 'refer_audio'
RESULT_DIR = BASE_DIR / 'static' / 'result_audio' / 'result.wav'

TMP_DIR = BASE_DIR / "tmp"
TMP_DIR.mkdir(exist_ok=True)

###################################################################

#Fastapi 가동 
router = APIRouter()


@router.post("/converse")
async def converse(request: Request, file: UploadFile = None, db: Session = Depends(get_db)):

    messages_list = request.session.get("messages", []).copy()

    human_ask = "음성 인식이 잘 되지 않았어요."
    ai_answer = "다시 말해 주세요"

    if file is None:
        messages_list.append({"role": "user", "content": human_ask})
        messages_list.append({"role": "assistant", "content": ai_answer})
        request.session["messages"] = messages_list
        return JSONResponse({"audio_url": None, "text": ai_answer})

    input_webm = TMP_DIR / "temp_input.webm"
    with open(input_webm, "wb") as f:
        f.write(await file.read())

    input_wav = TMP_DIR / "temp_input.wav"
    subprocess.run(["ffmpeg", "-y", "-i", str(input_webm), "-ar", "16000", "-ac", "1", str(input_wav)],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    human_ask_ = speech_to_text(str(input_wav), whisper_model)

    if not human_ask_ or not human_ask_.strip():
        messages_list.append({"role": "user", "content": human_ask})
        messages_list.append({"role": "assistant", "content": ai_answer})
    else:
        messages_list.append({"role": "user", "content": human_ask_})
        try:
            ai_answer = ask_llm(human_ask_, messages_list)
            messages_list.append({"role": "assistant", "content": ai_answer})

            db.add(ChatHistory(user_id=user_id, role="user", content=human_ask_))
            db.add(ChatHistory(user_id=user_id, role="assistant", content=ai_answer))
            db.commit()

        except Exception as e:
            print(f"오류 발생: {e}")
            messages_list.append({"role": "assistant", "content": ai_answer})

    refer_filename = request.session.get("audio", "narration.mp3")
    REFER = REFER_DIR / refer_filename
    output_file = text_to_speech(REFER, ai_answer, RESULT_DIR, zonos_model, make_cond_dict)

    static_audio_path = BASE_DIR / "static" / "audio" / output_file.name
    shutil.copy(output_file, static_audio_path)

    request.session["messages"] = messages_list

    return JSONResponse({
        "audio_url": f"/static/audio/{output_file.name}",
        "text": ai_answer
    })
