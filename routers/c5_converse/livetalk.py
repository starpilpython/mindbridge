# 실시간 대화에 필요한 핵심 기능을 넣은 함수

from pathlib import Path # 절대경로로 변환
import torchaudio
from llama_cpp import Llama # gguf llm 파일 모델 실행

# ==== 0. gguf 파일 직접 로딩 ==== 
llm = Llama(model_path="/home/elicer/mindbridge/AI_model/gemma-3-4B-it-QAT-Q4_0.gguf",
            verbose=False,    # CPU 병렬 처리 활성화
            n_gpu_layers=1000  , use_mlock=True)  # 로그 억제 


# ==== 1. 음성 → 텍스트 ====

def speech_to_text(filepath, model):
    print("Whisper로 음성을 텍스트로 변환 중...")
    try:
        segments, _  = model.transcribe(filepath)
        segments = list(segments)
        result = ""
        # 문장별로 분할된 결과를 하나로 합치는 과정 
        for segment in segments:
            result = result + " " + segment.text
       
        print("텍스트 변환 결과:")
        print(result)
        return result
        
    except Exception as e:
        print(f"Transcription error: {e}")
        return None


# ==== 2. LLM 질문 ====

# LLM에 보낼 메시지 형식이 올바른지 검사하는 함수(user ↔ assistant 처럼 역할이 변갈아 가는 것)
def validate_message_sequence(messages):
    last_role = None
    for msg in messages:
        if msg["role"] == "system":
            continue
        if last_role == msg["role"]:
            raise ValueError(f"잘못된 메시지 순서: '{msg['role']}'가 연속으로 등장했습니다.")
        last_role = msg["role"]

# 질문 생성 
def ask_llm(question, messages_list):
    if not question or not isinstance(question, str) or not question.strip():
        raise ValueError("ask_llm: 질문은 비어 있지 않은 문자열이어야 합니다.")

    output = llm.create_chat_completion(
        messages=messages_list,
        temperature=0.3
    )

    answer = output["choices"][0]["message"]["content"]

    return answer

# ==== 3. zonos TTS ====

def text_to_speech(audio_path, text, output_path, voice_model, make_cond_dict):
    # 참조 음성 로드 후 임베딩 생성
    audio_path = str(Path(audio_path).resolve())
    wav, sr = torchaudio.load(audio_path)
    speaker = voice_model.make_speaker_embedding(wav, sr)

    print("답변:", text)

    # 조건 생성
    cond = make_cond_dict(
        text=text,
        speaker=speaker,
        language="ko"
    )

    # 음성 생성
    conditioning = voice_model.prepare_conditioning(cond)
    codes = voice_model.generate(conditioning)
    wavs = voice_model.autoencoder.decode(codes).cpu()

    wave = wavs[0]
    if wave.ndim == 1:
        wave = wave.unsqueeze(0)
    elif wave.ndim == 3:
        wave = wave.squeeze(0)

    if output_path.exists():
        output_path.unlink()  # 기존 파일 삭제

    torchaudio.save(output_path, wave, voice_model.autoencoder.sampling_rate)

    print("음성 생성 완료")
    return output_path

