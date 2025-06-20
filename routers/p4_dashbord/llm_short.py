from transformers import BartForConditionalGeneration, PreTrainedTokenizerFast
from peft import PeftModel, PeftConfig
import torch
from llama_cpp import Llama # gguf llm 파일 모델 실행
from collections import Counter
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# 요약 
# GEMMA3 활용
def gemma_chat_once(user_input: str) -> str:

    # gguf 파일 직접 로딩 ==== 
    llm = Llama(model_path="/home/elicer/mindbridge/AI_model/gemma-3-4B-it-QAT-Q4_0.gguf",
                verbose=False,    # CPU 병렬 처리 활성화
                n_gpu_layers=1000  , n_ctx=131072 ,use_mlock=True)  # 로그 억제 

    if not user_input or not isinstance(user_input, str) or not user_input.strip():
        raise ValueError("입력은 비어 있지 않은 문자열이어야 합니다.")

    # 대화 메시지 구성
    messages = [
        {"role": "system", "content": "당신은 신중하고 정확하게 요약하고 응답하는 한국어 전문가입니다."},
        {"role": "user", "content": user_input}
    ]

    # Chat completion 실행
    output = llm.create_chat_completion(
        messages=messages,
        temperature=0.3
    )

    # 응답 추출
    return output["choices"][0]["message"]["content"].strip()


# 요약문 만들기 
# KoBert + LoRA 기반 모델 요약 수행 

def short_opinion(text_list):

    # (1) base 모델 정보 불러오기 
    MODEL_DIR = './AI_model/KoBert/local_kobart'
    adapter_path = './AI_model/KoBert/kobart_lora_adapter'

    # (2) bert 모델 로드 및 tokenizer로드
    tokenizer = PreTrainedTokenizerFast.from_pretrained(adapter_path)
    base_model = BartForConditionalGeneration.from_pretrained(MODEL_DIR)

    model = PeftModel.from_pretrained(base_model, adapter_path).to("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()

    # 입력 토크나이징
    inputs = tokenizer(text_list, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # (3) 요약 생성
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=32,
            num_beams=4,
            do_sample=True,  # 확률적 샘플링 비활성화
            no_repeat_ngram_size=8,
            temperature=0.5
        )

    # (4) 결과 디코딩
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("요약 결과:", summary.split(".")[0:1])

    # (5) 결과 요약 
    short_summary = gemma_chat_once(summary)
    print('임상가의 소견: ',short_summary)

    text_list_summray = gemma_chat_once(text_list)
    print()
    print()
    print('오늘 있었던일 요약: ',text_list_summray)

    return short_summary, text_list_summray

# 감성분석: 영상 
def emotions(emo_list):
    count_list = emo_list.split()
    count_list = [x for x in count_list if x != "None"]
    count = Counter(count_list)
    top3 = count.most_common(3)
    return top3


# 테스트 하기 
if __name__ == "__main__":
    import sqlite3
    import json
    from collections import Counter
    from datetime import date

    # DB 연결
    conn = sqlite3.connect("chat.db")
    cursor = conn.cursor()

    # 날짜 기준으로 고유 날짜 목록 가져오기
    cursor.execute("SELECT DISTINCT date FROM chat_history ORDER BY date ASC")
    all_dates = [row[0] for row in cursor.fetchall()]

    # 감정명 매핑
    emotion_kor_map = {
        "angry": "분노", "disgust": "혐오", "fear": "두려움",
        "happy": "행복", "sad": "슬픔", "surprise": "놀람", "neutral": "중립"
    }

    for current_date in all_dates:
        print(f"📅 날짜 처리 중: {current_date}")

        # (1) 해당 날짜 전체 대화 가져오기
        cursor.execute("""
            SELECT user_id, child_name, content, role, session_id
            FROM chat_history
            WHERE date = ?
            ORDER BY id
        """, (current_date,))
        rows = cursor.fetchall()
        if not rows:
            print(f"[SKIP] 대화 없음 - date={current_date}")
            continue

        child_id = rows[0][0]
        child_name = rows[0][1]
        session_id = rows[0][4]

        # 대화 문자열 구성
        dialogue = ""
        for _, _, content, role, _ in rows:
            prefix = "<|assistant|>" if role == "assistant" else "<|user|>"
            dialogue += f"{prefix}\n{content.strip()}\n"
        dialogue = dialogue.strip() + "\n<|user|>\n그 동안의 이야기만 요약만 해줘.\n"

        # (2) 요약 실행
        try:
            short_summary, text_list_summray = short_opinion(dialogue)
        except Exception as e:
            print(f"[ERROR] 요약 실패 - date={current_date} / error={e}")
            continue

        # (3) 감정 가져오기
        cursor.execute("SELECT emotions FROM emotion_messages WHERE session_id = ?", (session_id,))
        row = cursor.fetchone()
        if not row:
            print(f"[SKIP] 감정 없음 - session_id={session_id}")
            continue

        emotion_str = row[0]
        emo_list = [e for e in emotion_str.strip().split() if e != "NO"]
        counts = Counter(emo_list).most_common()
        top3 = counts[:3]

        emotion_counts = [(emotion_kor_map.get(x, x), y) for x, y in counts]
        top_counts = [(emotion_kor_map.get(x, x), y) for x, y in top3]

        # (4) 저장
        cursor.execute("""
            INSERT INTO child_short (user_id, child_id, child_name, short_summary, text_list_summray, emotion_counts, top_counts, date)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            "PARENT01",
            child_id,
            child_name,
            short_summary,
            text_list_summray,
            json.dumps(dict(emotion_counts), ensure_ascii=False),
            json.dumps(dict(top_counts), ensure_ascii=False),
            current_date
        ))

    conn.commit()
    conn.close()
    print("✅ 모든 날짜별 세션에 대해 요약 및 감정 결과 저장 완료.")
