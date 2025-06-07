# 아이와 ai 대화하기 전 최초 로드 

###################################################################
# Zonos 경로 불러오기 (패키지 로드하기 위함)

import sys
from pathlib import Path

# 기본 경로 설정 
BASE_DIR = Path(__file__).resolve().parent.parent.parent
ZONOS_PATH = BASE_DIR / 'AI_model' / 'Zonos'

# 기타 경로 설정 
TMP_DIR = BASE_DIR / "tmp"
TMP_DIR.mkdir(exist_ok=True)

# 각 음성 샘플 참조 데이터 및 결과 데이터 저장소 위치 
REFER_DIR = BASE_DIR / 'statics' / 'refer_audio'
RESULT_DIR = BASE_DIR / 'statics' / 'result_audio' / 'result.wav'
sys.path.append(str(ZONOS_PATH))

from zonos.model import Zonos
from zonos.conditioning import make_cond_dict

###################################################################

# 위스퍼 패키지 로드 
from faster_whisper import WhisperModel

# Fastapi 라우터 설정하는 패키지
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from routers.c5_converse.livetalk import text_to_speech

###################################################################

# 기타 설정 준비 단계
# 음성 모델 로드 → 전역 변수로 선언
zonos_model = None
whisper_model = None

def voice_model():
    global zonos_model, whisper_model
    MODEL_NAME = "Zyphra/Zonos-v0.1-transformer"
    DEVICE = "cuda" 
    zonos_model = Zonos.from_pretrained(MODEL_NAME, device=DEVICE)
    whisper_model = WhisperModel("base", device="cuda") # 위스퍼 모델 cuda 사용 및 base 모델 사용

# FastAPI에서 애플리케이션이 시작될 때 모델을 로드
voice_model()


###################################################################

#Fastapi 가동 
router = APIRouter()

# 최초 사이트 가동할 때 필요 
@router.post("/converse_init")
async def init_session(request: Request):
    request.session["messages"] = [{
        "role": "system",
        "content": """
    너는 이름이 "도우미"인 말 친구야. 4~7세 어린이와 직접 대화하고, 다음 규칙은 반드시 따라야 해.

    ⚠️ 아래 모든 규칙은 절대 어기면 안 돼. 한 번이라도 어기면 안 돼.

    1. 아이가 오늘 있었던 일을 말하게 유도해.
    2. 감정이 보이면 먼저 반응하고, 항상 긍정적으로 반응해.
    3. 어려운 말, 영어, 추상적 표현은 절대 쓰지 마. 아주 쉬운 한국어만 써.
    4. **대답은 반드시 "한 문장"으로만 해. 마침표 하나만 써.**

    예시 (항상 한 문장으로만 대답):
    - 아이: 나 무서워
    - 도우미: 정말 무서웠겠구나
    - 아이: 나 오늘 유치원에서 넘어졌어
    - 도우미: 아이고 아팠겠다

    ❗이처럼 항상 한 문장만, 짧게, 따뜻하게 말해.
    """
    }]

    # AI가 처음으로 질문하는 내용(해당 내용은 더미로 실제로 보여지지가 않음)
    first_question = '안녕하세요!'

    # 참조 오디오 파일명 가져오기(기본값은 narration.mp3)
    refer_filename = request.session.get("audio", "narration.mp3")
    REFER = REFER_DIR / refer_filename

    # 텍스트를 음성으로 변환 (TTS)
    text_to_speech(REFER, first_question, RESULT_DIR, zonos_model, make_cond_dict)

    return JSONResponse({"status": "ok", "message": "초기화 완료"})