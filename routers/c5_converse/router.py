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
from fastapi import APIRouter, UploadFile, Request, Depends, File
from fastapi.responses import JSONResponse
from routers.c5_converse.livetalk import speech_to_text, ask_llm, text_to_speech
from routers.c5_converse.emotion_detection import load_target_faces, detect_faces
from routers.c4_call.router import zonos_model, whisper_model

# DB 불러오기 
from DB.models import ChatHistory, EmotionMessages
from DB.database import get_db
from sqlalchemy.orm import Session

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
    print("SESSION 내용:", request.session.get("messages"))


    # 세션에 저장된 메세지 리스트를 불러옴. 없으면 system 메시지를 포함해서 초기화
    messages_list = request.session.get("messages")

    if not messages_list:
        messages_list = []

    # 시스템 메시지가 없는 경우에만 삽입 (중복 방지)
    has_system = any(m.get("role") == "system" for m in messages_list)

    if not has_system:
        messages_list.insert(0, {
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
        })
    else:
        messages_list = messages_list.copy()

    # 사용자 ID와 이름을 세션에서 가져옴. 기본값은 "anonymous"로 설정
    user_id = request.session.get("user_id", "anonymous")
    child_name = request.session.get("child_name", "anonymous")

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
            db.add(ChatHistory(user_id=user_id, child_name=child_name, role="user", content=human_ask_, session_id=request.session['session_id']))
            db.add(ChatHistory(user_id=user_id, child_name=child_name, role="assistant", content=ai_answer, session_id=request.session['session_id']))
            db.commit()  # 변경 사항을 커밋하여 DB에 저장

        except Exception as e:
            # 오류 발생 시, 오류를 출력하고 응답 메시지를 추가
            print(f"오류 발생: {e}")
            messages_list.append({"role": "assistant", "content": ai_answer})

    # 세션에서 참고할 파일 이름을 가져와 오디오 파일 생성
    refer_filename = request.session.get("audio", "narration.mp3")
    REFER = REFER_DIR / refer_filename
    output_file = text_to_speech(REFER, ai_answer, RESULT_DIR, zonos_model, make_cond_dict)

    # 메세지 리스트를 세션에 업데이트하여 다음 대화에 사용
    request.session["messages"] = messages_list
    print(messages_list)


    # 최종 응답 반환: 오디오 파일 URL과 AI의 응답 텍스트
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
    # 감정 저장 리스트 
    faces = request.session.get("emotions", []).copy()

    # 사용자 ID와 이름을 세션에서 가져옴. 기본값은 "anonymous"로 설정
    user_id = request.session.get("user_id", "anonymous")
    user_name = request.session.get("user_name", "anonymous")

    # 파일을 읽고 이미지로 변환
    img_bytes = await file.read()
    face = detect_faces(img_bytes)  # 분리된 함수 호출
    
    # 저장 리스트 저장
    faces.append(face)

    # 감정 기록을 세션에 저장 
    request.session["emotions"] = faces

    # 감정 기록을 DB에 저장
    txt = ""
    if faces is None:
        faces = []

    for sublist in faces:
        for face in sublist:
            if face:
                txt += " " + str(face)

    # DB에 적재
    db.add(EmotionMessages(user_id=user_id, child_name=user_name, emotions=txt, session_id=request.session['session_id']))
    db.commit()  # 변경 사항을 커밋하여 DB에 저장

    # JSON 응답 반환
    return JSONResponse(content={"faces": faces})

###################################################################

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