from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
import torch

# 기존 모델  경로
MODEL_DIR = ROOT_DIR / 'HyperCLOVAX-SEED-Text-Instruct-0.5B/models--naver-hyperclovax--HyperCLOVAX-SEED-Text-Instruct-0.5B/snapshots/b8c527c20d8fede1190c0b9c5e9821f30f42498a'

# (1) LoRA adapter 디렉토리
adapter_path = ROOT_DIR / TYPE_DIR / "lora_only_adapter"

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


