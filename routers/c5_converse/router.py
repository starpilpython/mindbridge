# 설명 추가

###################################################################
# Zonos 경로 불러오기 (패키지 로드하기 위함)

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
ZONOS_PATH = BASE_DIR / 'AI_model' / 'Zonos'
sys.path.append(str(ZONOS_PATH))

from zonos.model import Zonos
from zonos.conditioning import make_cond_dict

# 위스퍼 패키지 로드 
from faster_whisper import WhisperModel
import torchaudio
import subprocess, shutil

# Fastapi 라우터 설정하는 패키지
from fastapi import APIRouter, UploadFile, Request, Depends
from fastapi.responses import JSONResponse
from routers.c5_converse.livetalk import speech_to_text, ask_llm, text_to_speech

###################################################################


# 음성 모델 로드
MODEL_NAME = "Zyphra/Zonos-v0.1-transformer"
DEVICE = "cuda" 
voice_model = Zonos.from_pretrained(MODEL_NAME, device=DEVICE)

TMP_DIR = BASE_DIR / "tmp"
TMP_DIR.mkdir(exist_ok=True)

REFER_DIR = BASE_DIR / 'model_dialog' / 'refer_audio'
RESULT_DIR = BASE_DIR / 'model_dialog' / 'result_audio' / 'result.wav'

model = WhisperModel("base", device="cuda")
user_id = "test_user"
today = date.today()


router = APIRouter()



################################################################



@router.post("/api/converse_init")
async def init_session(request: Request):
    request.session["messages"] = [{
        "role": "system",
        "content": """
        너는 "도우미"라는 이름을 가지고 있고고 4~7세 어린이와 직접 이야기하는 말 친구야.
        다음 규칙을 꼭 지켜.

        1. 아이가 오늘 있었던 일을 스스로 말할 수 있도록, 짧고 쉬운 "한국어"만 사용해 부드럽게 말을 건네줘.
        2. 어려운 말, 영어, 추상적인 표현은 절대 쓰지 마.
        3. 질문이 오면 1문장 안에서 따뜻하게 대답해 줘.
        4. 자연스럽게 오늘 하루를 말하게 유도해 줘.

        예시:
        - 아이: 나 무서워  
        - 너: 그랬구나! 무서웠겠다. 우리 같이 이야기하면서 괜찮아지자~

        너는 이처럼 아이의 감정에 먼저 반응하고, 항상 긍정적으로 이야기해.
        """
    }]
    return JSONResponse({"status": "ok", "message": "초기화 완료"})















@router.post("/api/converse")
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

    human_ask_ = speech_to_text(str(input_wav), model)

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
    output_file = text_to_speech(REFER, ai_answer, RESULT_DIR, voice_model, make_cond_dict)

    static_audio_path = BASE_DIR / "static" / "audio" / output_file.name
    shutil.copy(output_file, static_audio_path)

    request.session["messages"] = messages_list

    return JSONResponse({
        "audio_url": f"/static/audio/{output_file.name}",
        "text": ai_answer
    })
