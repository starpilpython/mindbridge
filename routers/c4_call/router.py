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
from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from routers.c5_converse.livetalk import text_to_speech


###################################################################

# DB 관련
from DB.database import get_db
from DB.models import MemberList
from sqlalchemy.orm import Session

# FastAPI 엔드포인트 안에서 사용(음성 제목 조절)
def get_member_audio(user_id: str, db: Session):
    member = db.query(MemberList).filter(MemberList.user_id == user_id).first()
    return member.audio

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
async def init_session(db: Session = Depends(get_db)):
    
    # 오디오 경로 지정 
    refer_filename = get_member_audio("test001", db)
    print(refer_filename)

    # AI가 처음으로 질문하는 내용(해당 내용은 더미로 실제로 보여지지가 않음)
    first_question = '안녕하세요!'

    # 참조 오디오 파일명 가져오기
    REFER = REFER_DIR / refer_filename

    # 텍스트를 음성으로 변환 (TTS)
    text_to_speech(REFER, first_question, RESULT_DIR, zonos_model, make_cond_dict)
    print('음성참조: ',refer_filename)

    return JSONResponse({"ok": True})