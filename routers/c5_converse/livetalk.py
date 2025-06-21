# 실시간 대화에 필요한 핵심 기능을 넣은 함수

from pathlib import Path # 절대경로로 변환
import torchaudio
from llama_cpp import Llama # gguf llm 파일 모델 실행

# ==== 0. gguf 파일 직접 로딩 ==== 
llm = Llama(model_path="/home/elicer/mindbridge/AI_model/gemma-3-4B-it-QAT-Q4_0.gguf",
            verbose=False,    # CPU 병렬 처리 활성화
            n_gpu_layers=1000  , n_ctx=131072 ,use_mlock=True)  # 로그 억제 


# ==== 1. 음성 → 텍스트 ====
def speech_to_text(filepath, model):
    print(f"[Whisper] 입력 파일 경로: {filepath}")
    
    try:
        # 파일 정보 출력
        info = torchaudio.info(filepath)
        print(f"[Whisper] 샘플레이트: {info.sample_rate}, 채널 수: {info.num_channels}, 길이(초): {info.num_frames / info.sample_rate:.2f}s")
        
        result = model.transcribe(filepath)  # 기본 whisper는 딕셔너리 반환
        print(f"[Whisper] 전체 결과: {result['text'].strip()}")
        return result["text"].strip()

    except Exception as e:
        print(f"[Whisper 오류] {e}")
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

# ==== 테스트 실행 ====
if __name__ == "__main__":
    from faster_whisper import WhisperModel
    import sounddevice as sd
    print(sd.query_devices())
    print(sd.default.device)
    from scipy.io.wavfile import write

    def record_audio(filename="sample.wav", duration=5, fs=16000):
        print(f"{duration}초 동안 마이크로부터 녹음합니다...")
        audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
        sd.wait()
        write(filename, fs, audio)
        print(f"녹음 완료: {filename}")
        return filename

    # 1. 녹음
    input_audio = record_audio("sample.wav", duration=5)

    # 2. Whisper 모델 로딩
    whisper_model = WhisperModel("base", compute_type="float16")

    # 3. 음성 인식
    text = speech_to_text(input_audio, whisper_model)

    # 4. LLM 응답
    if text:
        messages = [
            {"role": "system", "content": "당신은 유용하고 정중한 비서입니다."},
            {"role": "user", "content": text}
        ]
        validate_message_sequence(messages)
        response = ask_llm(text, messages)
        print("LLM 응답:", response)
