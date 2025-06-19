from transformers import BartForConditionalGeneration, PreTrainedTokenizerFast
from peft import PeftModel, PeftConfig
import torch
from llama_cpp import Llama # gguf llm 파일 모델 실행
from collections import Counter

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

    # python -m routers.p4_dashbord.llm_short

    # DB 불러오기 
    from DB.models import ChatHistory, ChildShort,EmotionMessages
    from DB.database import get_db
    from sqlalchemy.orm import Session
    from sqlalchemy import desc
    from collections import Counter


    # 핵심 기능 호출 
    from routers.p4_dashbord import llm_short

    # 함수 내부에서 수동으로 지정할 때 
    db = next(get_db())


    '''기존에 받은 내용 → 적합한 탬플릿 형식으로 바꾸기 '''
    latest_session = (
        db.query(ChatHistory)
        .order_by(desc(ChatHistory.id))
        .first()
    )
    
    session_id = latest_session.session_id
    name = latest_session.child_name
    child_id = latest_session.user_id
    make_date = latest_session.date

    print(name)

    
    # 대화 로그 수집
    if latest_session:
        chat_logs = (
            db.query(ChatHistory)
            .filter(ChatHistory.session_id == session_id)
            .order_by(ChatHistory.id)
            .all()
        )

    dialogue = ""
    for log in chat_logs:
        prefix = "<|assistant|>" if log.role == "assistant" else "<|user|>"
        dialogue += f"{prefix}\n{log.content.strip()}\n"
    dialogue = dialogue.strip()
    dialogue += "\n<|user|>\n그 동안의 이야기만 요약만 해줘.\n"

    # 요약 결과
    short_summary, text_list_summray = llm_short.short_opinion(dialogue)

    # 감정 기록 수집 및 집계
    emotion_rows = (
        db.query(EmotionMessages.emotions)
        .filter(EmotionMessages.session_id == session_id)
        .all()
    )

    # 감정 개수 세기 
    from collections import Counter
    practices = emotion_rows[0].split(" ")
    emo_list = [practice for practice in practices if practice != 'NO']
    Counter(emo_list)















































