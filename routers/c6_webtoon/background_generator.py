# 배경 생성기
# Ghibli 스타일 배경을 생성하는 스크립트
from pathlib import Path
from PIL import ImageFilter
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler

# --- 설정 ---
OUTPUT_DIR = Path("statics/webtoon/backgrounds").resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

generator = torch.Generator(device="cuda").manual_seed(1025)
base_model_path = "./AI_model/Webtoon/Disney/disneyPixarCartoon_v10.safetensors" # 디즈니 픽사버전 얼굴 모델 

# --- 파이프라인 로딩 (xformers 제거) ---
def load_pipeline():

    base_model_path = "./AI_model/Webtoon/Disney/disneyPixarCartoon_v10.safetensors" # 디즈니 픽사버전 얼굴 모델 
    pipe = StableDiffusionPipeline.from_single_file(
        base_model_path,
        torch_dtype=torch.float16,
        safety_checker=None,
        feature_extractor=None
        ).to("cuda")

    return pipe

# --- 배경 생성 ---
def generate_background(pipe, prompt: str, output_path: Path, blur_radius: float = 1):
    styled_prompt = f"{prompt}, watercolor, pastel tones, soft lighting, ghibli style"
    image = pipe(
        prompt=styled_prompt,
        negative_prompt="text, watermark, logo, blurry, distorted, sketch, outline",
        width=1280,
        height=768,
        guidance_scale=7.5,
        num_inference_steps=30,
        generator=generator,
    ).images[0]

    blurred = image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    blurred.save(output_path)
    print(f"배경 저장 완료: {output_path.name}")

# --- 전체 처리 흐름 ---
def generate_all_backgrounds(CUTSCENE_CONFIG):
    
    pipe = load_pipeline()
    print("STEP1. 파이프라인 로딩 완료")

    for i, prompt in enumerate(CUTSCENE_CONFIG, start=1):
        fname = OUTPUT_DIR / f"{prompt['cut_id']}.png"
        print(f"STEP2. {i}번째 배경 생성 시작: {prompt['cut_id']}",fname)
        blur = 3
        generate_background(pipe, prompt['background_prompt'], fname, blur_radius=blur)
        print(f"STEP2. {i}번째 배경 생성 완료")

# --- 실행 ---
if __name__ == "__main__":
    CUTSCENE_CONFIG = [
        {
            "cut_id": "cut01",
            "background_prompt": "a small cozy village in the mountains during sunset"
        },
        {
            "cut_id": "cut02",
            "background_prompt": "a quiet library room with sunlight through windows"
        }
    ]

    generate_all_backgrounds(CUTSCENE_CONFIG)


    