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

if __name__ == "__main__":
    text_list =  " <|user|>\n안녕하세요. 오늘 기분이 좀 안 좋아요\n<|assistant|>\n무슨 일이 있었는지 이야기해 줄 수 있어요?\n<|user|>\n학교에서 친구랑 싸웠어요. 괜히 화가 나서 말실수도 했고요.\n<|assistant|>\n그랬구나. 친구와의 갈등이 속상했겠어요.\n<|user|>\n응, 사과는 하고 싶은데 어떻게 해야 할지 모르겠어요.\n<|assistant|>\n사과하고 싶은 마음이 중요한 첫걸음이에요. 진심을 담아서 이야기해보면 좋을 거예요.\n<|user|>\n그 동안의 이야기만 요약해줘 ."
    print('대화:',text_list)
    print()
    print()
    short_opinion(text_list)

























