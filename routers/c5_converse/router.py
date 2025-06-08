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
from routers.c5_converse.emotion_detection import load_target_faces, detect_faces
from routers.c4_call.router import zonos_model, whisper_model

# DB 불러오기 
from DB.models import ChatHistory, EmotionMessages,MemberList
from DB.database import get_db
from sqlalchemy.orm import Session
from sqlalchemy import desc

# FastAPI 엔드포인트 안에서 사용(음성 및 화자 이름 및 아이디 출력)
def get_member(user_id: str, db: Session):
    member = db.query(MemberList).filter(MemberList.user_id == user_id).first()
    return member

###################################################################

# 각 음성 샘플 참조 데이터 및 결과 데이터 저장소 위치 
REFER_DIR = BASE_DIR / 'statics' / 'refer_audio'
RESULT_DIR = BASE_DIR / 'statics' / 'result_audio' / 'result.wav'

TMP_DIR = BASE_DIR / "tmp"
TMP_DIR.mkdir(exist_ok=True)

###################################################################

#Fastapi 가동 
router = APIRouter()

###################################################################
# 아동 - AI 대화 DB 기록 
@router.post("/converse")
async def converse(request: Request, file: UploadFile = None, db: Session = Depends(get_db)):
    # 사용자 정보
    member = get_member("test001", db)
    user_id = member.user_id
    child_name = member.child_name
    audio = member.audio
    session_id = member.session_id

    # 기본 시스템 메시지
    system_msg = {
        "role": "system",
        "content": """
        너는 이름이 "도우미"인 말 친구야. 4~7세 어린이와 직접 대화하고, 다음 규칙은 반드시 따라야 해.

        ⚠️ 아래 모든 규칙은 절대 어기면 안 돼. 한 번이라도 어기면 안 돼.

        1. 아이가 오늘 있었던 일을 말하게 유도해.
        2. 감정이 보이면 먼저 반응하고, 항상 긍정적으로 반응해.
        3. 어려운 말, 영어, 추상적 표현은 절대 쓰지 마. 아주 쉬운 **한국어**만 써.
        4. **대답은 반드시 "한 문장"으로만 해. 마침표 하나만 써.**

        예시 (항상 한 문장으로만 대답):
        - 아이: 나 무서워
        - 도우미: 정말 무서웠겠구나
        - 아이: 나 오늘 유치원에서 넘어졌어
        - 도우미: 아이고 아팠겠다

        ❗이처럼 항상 한 문장만, 짧게, 따뜻하게 말해.
        """
    }

    # 이전 대화 불러오기
    history_rows = (
        db.query(ChatHistory)
        .filter(ChatHistory.session_id == session_id)
        .order_by(ChatHistory.id.asc())
        .all()
    )

    messages_list = [{"role": row.role, "content": row.content} for row in history_rows]

    # system 메시지가 없다면 맨 앞에 삽입
    if not any(m["role"] == "system" for m in messages_list):
        messages_list.insert(0, system_msg)

    # 음성 인식 실패 시 기본 답변
    human_ask = "음성 인식이 잘 되지 않았어요."
    ai_answer = "다시 말해 주세요"

    if file is None:
        messages_list.append({"role": "user", "content": human_ask})
        messages_list.append({"role": "assistant", "content": ai_answer})
        return JSONResponse({"audio_url": None, "text": ai_answer})

    # 파일 저장 및 변환
    input_webm = TMP_DIR / "temp_input.webm"
    with open(input_webm, "wb") as f:
        f.write(await file.read())

    input_wav = TMP_DIR / "temp_input.wav"
    subprocess.run(["ffmpeg", "-y", "-i", str(input_webm), "-ar", "16000", "-ac", "1", str(input_wav)],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # 음성 → 텍스트
    human_ask_ = speech_to_text(str(input_wav), whisper_model)

    if not human_ask_ or not human_ask_.strip():
        messages_list.append({"role": "user", "content": human_ask})
        messages_list.append({"role": "assistant", "content": ai_answer})
    else:
        messages_list.append({"role": "user", "content": human_ask_})

        try:
            # LLM 호출
            ai_answer = ask_llm(human_ask_, messages_list)
            messages_list.append({"role": "assistant", "content": ai_answer})

            # DB에 각각 저장
            db.add(ChatHistory(user_id=user_id, child_name=child_name, role="user", content=human_ask_, session_id=session_id))
            db.add(ChatHistory(user_id=user_id, child_name=child_name, role="assistant", content=ai_answer, session_id=session_id))
            db.commit()

        except Exception as e:
            print(f"오류 발생: {e}")
            messages_list.append({"role": "assistant", "content": ai_answer})

    # 음성 생성
    REFER = REFER_DIR / audio
    output_file = text_to_speech(REFER, ai_answer, RESULT_DIR, zonos_model, make_cond_dict)

    return JSONResponse({
        "audio_url": f"/statics/result_audio/{output_file.name}",
        "text": ai_answer
    })





















###################################################################

# 아동-AI 대화 웹캠 통한 YOLO로 얼굴 검출한 뒤 DEEPFACE로 감정 분석
# 서버 시작 시 기준점 되는 얼굴 로드 "emotion_detection.py" 참조
def load_faces():
    # 서버 시작 시 실행할 코드 
    load_target_faces()

# 클라이언트가 전송한 이미지에서 얼굴 인식 후 감정 분석
@router.post("/emo_detect")
async def detect(request: Request, file: UploadFile = None, db: Session = Depends(get_db)):
    # 사용자 정보 불러오기
    member = get_member("test001", db)
    user_id = member.user_id
    child_name = member.child_name
    session_id = member.session_id

    # 새로 전송된 이미지에서 감정 분석
    img_bytes = await file.read()
    detected_faces = detect_faces(img_bytes)

    # 새로운 감정들을 문자열로 정리
    new_emotions = " ".join(str(face) for sub in detected_faces for face in sub if face)

    # 기존 감정 기록 불러오기
    latest = (
        db.query(EmotionMessages)
        .filter(
            EmotionMessages.user_id == user_id,
            EmotionMessages.session_id == session_id
        )
        .order_by(desc(EmotionMessages.id))  # 가장 최근 감정 기록
        .first()
    )

    if latest:
        latest.emotions += " " + new_emotions
    else:
        latest = EmotionMessages(
            user_id=user_id,
            child_name=child_name,
            session_id=session_id,
            emotions=new_emotions
        )
        db.add(latest)

    db.commit()

    # 응답
    return JSONResponse(content={"faces": detected_faces})

###################################################################
'''
# 아동 - AI 대화 요약 및 영상 감정 추출 
@router.post('/generate-summary')
async def summary(request: Request, db: Session = Depends(get_db)):

    # 세션에 저장된 id 가져오기 
    user_id = request.session.get("user_id", "child_001")

    # 대화 중 최신 데이터 가져오기 
    latest_message = (
        db.query(ChatHistory)
        .filter(ChatHistory.user_id == user_id)
        .order_by(ChatHistory.date.desc())  # date 필드를 기준으로 정렬
        .first()  # 가장 최신 메시지 하나 반환
    )  

    # 비동기 함수로 요약 및 영상 제작 호출
    summary_result = await create_summary(latest_message)
    video_result = await create_video(latest_message)

    # 결과 반환
    return {"summary": summary_result, "video": video_result}


# 대화 요약을 생성하는 기능
async def create_summary(chat_message):
        # ... 구현 내용 ...
    return "대화 요약 결과"

# 영상 제작 기능
async def create_video(emotion_message):
    
    # ... 구현 내용 ...
    return "영상 제작 결과"

'''