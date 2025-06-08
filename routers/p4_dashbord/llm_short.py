from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
import torch

# 요약문 만들기 
# 네이버 하이퍼 클로바 시드 0.5b 사용 
def short_opinion(text_list):

  # (1) base 모델 정보 불러오기 
  MODEL_DIR = './AI_model/Naver/HyperCLOVAX-SEED-Text-Instruct-0.5B/models--naver-hyperclovax--HyperCLOVAX-SEED-Text-Instruct-0.5B/snapshots/b8c527c20d8fede1190c0b9c5e9821f30f42498a'
  adapter_path = './AI_model/Naver/lora_only_adapter'

  # (2) LoRA에 포함된 base 모델 정보 확인
  peft_config = PeftConfig.from_pretrained(adapter_path)

  # (3) base 모델 로드
  base_model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, trust_remote_code=True).to(device='cuda')

  # (4) LoRA adapter 적용
  model = PeftModel.from_pretrained(base_model, adapter_path).to("cuda")
  model.eval()

  # (5) tokenizer는 adapter 경로에서 불러도 무방
  tokenizer = AutoTokenizer.from_pretrained(adapter_path)
  tokenizer.pad_token = tokenizer.eos_token

  # (6). 토크나이즈 및 추론
  inputs = tokenizer(text_list, return_tensors="pt", padding=True)
  inputs = {k: v.to("cuda" if torch.cuda.is_available() else "cpu") for k, v in inputs.items()}

  with torch.no_grad():
      outputs = model.generate(
          input_ids=inputs["input_ids"],
          attention_mask=inputs["attention_mask"],
          max_new_tokens=1024,
          pad_token_id=tokenizer.pad_token_id,
          eos_token_id=tokenizer.eos_token_id,
      )

  result = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
  print("요약:", result)

  return result










