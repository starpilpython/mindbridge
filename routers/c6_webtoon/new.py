import torch
from diffusers import LTXConditionPipeline, LTXLatentUpsamplePipeline
from diffusers.pipelines.ltx.pipeline_ltx_condition import LTXVideoCondition
from diffusers.utils import export_to_video, load_video
from PIL import Image

def generate_video_with_ltx(
    image_path: str,
    output_path: str,
    prompt: str,
    negative_prompt: str = "worst quality, inconsistent motion, blurry, jittery, distorted",
    expected_height: int = 832,
    expected_width: int = 480,
    num_frames: int = 32,
    downscale_factor: float = 2/3,
    seed: int = 0
):
    # 모델 로딩
    pipe = LTXConditionPipeline.from_pretrained(
        "Lightricks/LTX-Video-0.9.7-dev", torch_dtype=torch.float16
    )
    pipe_upsample = LTXLatentUpsamplePipeline.from_pretrained(
        "Lightricks/ltxv-spatial-upscaler-0.9.7", vae=pipe.vae, torch_dtype=torch.float16
    )
    pipe.to("cuda")
    pipe_upsample.to("cuda")
    pipe.vae.enable_tiling()

    def round_to_nearest(height, width):
        ratio = pipe.vae_spatial_compression_ratio
        return height - (height % ratio), width - (width % ratio)

    # 이미지 로드 및 전처리
    image = Image.open(image_path).convert("RGB").resize((512, 512))
    video = load_video(export_to_video([image]))
    condition = LTXVideoCondition(video=video, frame_index=0)

    # 크기 조정
    down_h, down_w = int(expected_height * downscale_factor), int(expected_width * downscale_factor)
    down_h, down_w = round_to_nearest(down_h, down_w)

    # 1단계: 영상 생성 (latent)
    latents = pipe(
        conditions=[condition],
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=down_w,
        height=down_h,
        num_frames=num_frames,
        num_inference_steps=30,
        generator=torch.Generator().manual_seed(seed),
        output_type="latent",
    ).frames


    # 2단계: 업스케일
    up_latents = pipe_upsample(latents=latents, output_type="latent").frames
    up_h, up_w = down_h * 2, down_w * 2

    # 3단계: 디노이즈
    video_frames = pipe(
        conditions=[condition],
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=up_w,
        height=up_h, 
        num_frames=num_frames,
        denoise_strength=0.4,
        num_inference_steps=10,
        latents=up_latents,
        decode_timestep=0.05,
        image_cond_noise_scale=0.025,
        generator=torch.Generator().manual_seed(seed),
        output_type="pil",
    ).frames[0]

    # 4단계: 최종 리사이즈 및 저장
    final_frames = [frame.resize((expected_width, expected_height)) for frame in video_frames]
    export_to_video(final_frames, output_path, fps=24)

if __name__ == "__main__":
    # 예시 호출
    generate_video_with_ltx(
        image_path="main.png",
        output_path="output.mp4",
        prompt="A MAN EAT A some slowly"
    )
