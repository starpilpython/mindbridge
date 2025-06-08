import sys
from pathlib import Path

BASE_DIR = "/home/elicer/mindbridge/AI_model/Webtoon/IP-Adapter"
sys.path.append(str(BASE_DIR))

import torch
from pathlib import Path
from diffusers import StableDiffusionPipeline, DDIMScheduler, AutoencoderKL
from ip_adapter.ip_adapter_faceid import IPAdapterFaceID
from safetensors.torch import load_file
from insightface.app import FaceAnalysis
import cv2
import numpy as np
from PIL import Image
from rembg import remove

# --- 디렉토리 설정 ---
CHARACTER_INPUT_DIR = Path("statics/webtoon/raw_character").resolve()
CHARACTER_INPUT_DIR.mkdir(exist_ok=True)

CHARACTER_OUTPUT_DIR = Path("statics/webtoon/make_character").resolve()
CHARACTER_OUTPUT_DIR.mkdir(exist_ok=True)

# --- 얼굴 임베딩 추출 함수 ---
def extract_face_embedding(image_path):
    app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))
    img = cv2.imread(str(image_path))
    faces = app.get(img)
    if not faces:
        raise ValueError(f"얼굴 감지 실패: {image_path}")
    return torch.from_numpy(faces[0].normed_embedding).unsqueeze(0).to("cuda")

# --- 파이프라인 및 IPAdapter 로딩 ---
def load_faceid_pipeline():

    ''' step1 --- 모델 경로 설정 --- '''
    ip_ckpt = "./AI_model/Webtoon/Disney/ip-adapter-faceid_sd15.bin" #얼굴 이미지 모델
    base_model_path = "./AI_model/Webtoon/Disney/disneyPixarCartoon_v10.safetensors" # 디즈니 픽사버전 얼굴 모델 
    
    ''' step2 --- 모델 로딩 ---'''
    pipe = StableDiffusionPipeline.from_single_file(
        base_model_path,
        torch_dtype=torch.float16,
        safety_checker=None,
        feature_extractor=None
        ).to("cuda")
     
    ''' step3 --- 얼굴 고정시키는 모델 입력 ---'''
    adapter = IPAdapterFaceID(pipe, ip_ckpt, "cuda")       

    return adapter


# --- 이미지 생성 함수 ---
def generate_images(ip_adapter, CUTSCENE_CONFIG):

    ''' step1 --- 프롬프트 기준에 따른 사진 참조 결정 --- '''
    for scene in CUTSCENE_CONFIG:
        cut_id = scene["cut_id"]

        for char in scene["characters"]:
            name = char["name"]
            prompt = char["prompt"]

            # 참조 파일 없으면 etc.png 강제 대입
            ref_file = char.get("ref") or "etc.png"
            ref_path = CHARACTER_INPUT_DIR / ref_file

            # 실제 파일이 존재하지 않으면 etc.png로 다시 설정
            if not ref_path.exists():
                print(f"[!] 참조 이미지 없음: {ref_path}")
                alt_path = CHARACTER_INPUT_DIR / "etc.png"
                if alt_path.exists():
                    print("[!] etc.png로 대체합니다.")
                    ref_file = "etc.png"
                    ref_path = alt_path
                else:
                    print("[!] etc.png도 없음 → 얼굴 없이 생성")
                    faceid_embeds = None
                    ref_path = None

            # ref_path가 존재한다면 임베딩 추출 시도
            if ref_path is not None:
                try:
                    faceid_embeds = extract_face_embedding(str(ref_path))
                    print(f"[✓] 얼굴 임베딩 추출 완료: {ref_path}")
                except ValueError:
                    print(f"[!] 얼굴 인식 실패: {ref_path} → 얼굴 없이 생성")
                    faceid_embeds = None


            # 얼굴 임베딩 유무 관계없이 무조건 생성
            images = ip_adapter.generate(
                prompt=prompt,
                negative_prompt = (
                    "detailed shadows, multiple people, logo, text, artifacts, blur, noise, distortion, "
                ), # 부정 프롬프트
                faceid_embeds=faceid_embeds, # 얼굴 임베딩 
                num_samples=1, # 단 한개만 생성
                width= 672,
                height=1056,
                num_inference_steps=40, # 생성 단계 수(많을 수록 오래걸지만 잘 생성됨)
                seed=1025, # 시드 고정 1025 
                guidance_scale=5.0, # 프롬프트에 영향을 받는 정도 
                scale=0.6 # 얼굴 고정 강도
            )

            removed = images[0] # 이미지중 첫번째 것을 선택 
            removed = remove(removed) # 배경 지우기 
            output_path = CHARACTER_OUTPUT_DIR / f"{cut_id}_{name}.png" # 저장

            # 주인공이 아니라면 고개를 돌린다(보통 좌에서 우로 보는 시선으로 생성되기 때문문)
            if name != "main":
                removed = np.array(removed)
                removed = cv2.flip(removed, 1)
                removed = Image.fromarray(removed)

            removed.save(output_path)
            print(f"[✓] 저장 완료: {output_path}")



# --- 실행 ---
def character_main(CUTSCENE_CONFIG):
    adapter = load_faceid_pipeline()
    generate_images(adapter,CUTSCENE_CONFIG)
    # 생성 후 캐시 정리
    torch.cuda.empty_cache()

# --- 메인 실행 코드 ---
if __name__ == "__main__":

    # 프롬프트 예시 
    import ast
    with open("prompt.txt", "r", encoding="utf-8") as f:
        text = f.read()
    CUTSCENE_CONFIG = ast.literal_eval(text) 
    
    character_main(CUTSCENE_CONFIG)
