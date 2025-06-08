import torchaudio
import numpy as np
from PIL import Image
import textwrap
import numpy as np
from moviepy import ImageSequenceClip, AudioFileClip, concatenate_videoclips
from PIL import ImageDraw, ImageFont
import os
import glob

# 음성 길이 측정   ← 오디오 파일로부터 재생 시간을 초 단위로 계산
def get_audio_duration(path):
    infot = torchaudio.info(path)
    return infot.num_frames / infot.sample_rate

# 흔들림용 사인파 함수   ← 지정된 길이와 FPS로 사인파 형태의 이동량 리스트 생성 (픽셀 단위)
def get_sine_motion_frames(duration, fps=30, amplitude=3):
    frames = int(duration * fps)
    return [int(amplitude * np.sin(2 * np.pi * i / 40)) for i in range(frames)]

# 이미지 흔들림 적용   ← 사인파 이동량(shifts)에 따라 이미지에 x축 또는 y축 흔들림 프레임 생성
def apply_motion(image, shifts, axis='y'):
    frames = []
    for shift in shifts:
        if axis == 'x':
            # x축 기준 이동 변환
            frame = image.transform(image.size, Image.AFFINE, (1, 0, shift, 0, 1, 0))
        else:
            # y축 기준 이동 변환
            frame = image.transform(image.size, Image.AFFINE, (1, 0, 0, 0, 1, shift))
        frames.append(frame)
    return frames


# 자막 생성  ← 이미지 하단에 자막을 추가하는 함수
def draw_subtitle(image_np, text, font_path, width=60):
    # numpy 배열 → PIL 이미지로 변환
    image = Image.fromarray(image_np)
    draw = ImageDraw.Draw(image)

    # 이미지 크기
    w, h = image.size

    # 폰트 로드 (크기는 필요에 맞게 조정)
    font = ImageFont.truetype(font_path, size=32)

    # 줄바꿈 처리
    lines = textwrap.wrap(text, width)

    # 자막 배경 사각형 (하단 20%)
    subtitle_height = int(h * 0.25)
    draw.rectangle([(0, h - subtitle_height), (w, h)], fill=(0, 0, 0))

    # 자막 텍스트 쓰기
    for i, line in enumerate(lines):
        y = h - subtitle_height + 20 + i * 36
        draw.text((20, y), line, font=font, fill=(255, 255, 255))

    # 다시 numpy 배열로 변환해서 반환
    return np.array(image)


# 최종 실행 함수 
def webtoon_main(CUTSCENE_CONFIG,filename):
    final_clips = []
    FONT_PATH = './statics/font/Pretendard-Bold.ttf'

    for scene in CUTSCENE_CONFIG:
        bg = Image.open(f"./statics/webtoon/backgrounds/{scene['cut_id']}.png").convert("RGBA")
        bg_w, bg_h = bg.size  # 배경 이미지 크기 측정

        # 캐릭터 이미지 로딩 및 위치 지정 (좌우 끝에 배치)
        char_data = {}
        for char in scene["characters"]:
            name = char["name"]
            img_path = f"./statics/webtoon/make_character/{scene['cut_id']}_{name}.png"
            img = Image.open(img_path).convert("RGBA")
            fg_w, fg_h = img.size

            if name == "main":
                pos = (int(bg_w * 0.03), 60)  # 왼쪽 여백 3%
            else:
                pos = (bg_w - fg_w - int(bg_w * 0.03), 60)  # 오른쪽 여백 3%

            char_data[name] = {"img": img, "pos": pos}

        for seg_idx, segment in enumerate(scene.get("segments", [])):
            seg_type = segment["type"]
            text = segment["text"]
            audio_path = f"./statics/result_audio/{scene['cut_id']}_{seg_idx}.wav"

            duration = get_audio_duration(audio_path)
            shifts = get_sine_motion_frames(duration)
            total_frames = []

            for i in range(len(shifts)):
                frame = bg.copy()
                for name, data in char_data.items():
                    if seg_type == "dialogue" and name == segment.get("speaker"):
                        fg = apply_motion(data["img"], [shifts[i]])[0]
                    else:
                        fg = data["img"]
                    frame.paste(fg, data["pos"], fg)

                frame_rgb = frame.convert("RGB")
                frame_np = draw_subtitle(np.array(frame_rgb), text, FONT_PATH)  # 자막 함수는 아래 참고
                total_frames.append(frame_np)

            clip = ImageSequenceClip(total_frames, fps=30).with_audio(AudioFileClip(audio_path))
            final_clips.append(clip)


    # 클립 이어붙이고 저장
    final = concatenate_videoclips(final_clips)
    final.write_videofile(f"statics/webtoon/video/{filename}.mp4", fps=30)


    # 이어붙이고 저장
    final = concatenate_videoclips(final_clips)
    final.write_videofile(f"statics/webtoon/video/{filename}.mp4", fps=30)

    # 정리
    for folder in ['./statics/webtoon/make_character/*', './statics/webtoon/backgrounds/*', './statics/result_audio/*']:
        for file_path in glob.glob(folder):
            if os.path.isfile(file_path):
                os.remove(file_path)


if __name__ == "__main__":
    
    # 해당 코드 전사적으로 사용 가능한 코드 
    from background_generator import *
    from character_pipeline import *
    from character_tts import *
    from make_template_rag import *
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.orm import Session
    from sqlalchemy import desc
    import json

    query = '''
            <|assistant|>\n최근에 아픈 곳이 있어?\n<|user|>\n네. 머리가 아파요.\n<|assistant|>\n머리가 어떻게 아파?\n<|user|>\n그냥 지끈거려요. 뭔가 긴장되는 상황이 생기면 머리가 아픈 거 같아요.\n<|assistant|>\n이전에도 해당 부위가 자주 아팠어?\n<|user|>\n네. 자주 그래요. 근데 오래 지속되지는 않아요.\n<|assistant|>\n몸 여기저기에 반창고가 많네. 언제 다친 거야?\n<|user|>\n며칠 전에 아빠한테 혼나서 다쳤어요.\n<|assistant|>\n오늘 이야기 해주어서 고마워\n<|assistant|>\n친구는 뭐 할 때가 즐거워?\n<|user|>\n잘 모르겠어요.\n<|assistant|>\n즐겁다고 느끼지 못한 게 언제부터야?\n<|user|>\n그것도 잘 모르겠어요.\n<|assistant|>\n최근 일주일 동안 짜증이나 화가 난 적 있어?\n<|user|>\n네. 화난 적 있어요.\n<|assistant|>\n무엇 때문에 화가 났어?\n<|user|>\n아빠가 무서워서 아무것도 못하는 제 자신에게 화가 났어요.\n<|assistant|>\n그런 일이 자주 일어나?\n<|user|>\n네.\n<|assistant|>\n오늘 이야기 해주어서 고마워\n<|assistant|>\n몇 시에 자고 몇 시에 일어나?\n<|user|>\n새벽 한 시쯤 자고 일곱 시에 일어나요.\n<|assistant|>\n아침에 일어났을 때 몸 상태는 어때?\n<|user|>\n안 좋아요. 전날에 다치거나 했을 때는 몸이 아프고, 아니면 그냥 안 좋아요. 힘도 없고요.\n<|assistant|>\n오늘 이야기 해주어서 고마워\n<|assistant|>\n평소 아빠는 어떤 표정을 자주 지으셔?\n<|user|>\n화내는 표정이요.\n<|assistant|>\n아빠가 왜 그러시는 거 같아?\n<|user|>\n그건 모르겠지만 항상 화나있어요.\n<|assistant|>\n아빠가 그런 표정을 지으시면 내 기분은 어때?\n<|user|>\n무서워요.\n<|assistant|>\n평소 엄마는 어떤 표정을 자주 지으셔?\n<|user|>\n슬픈 표정이요.\n<|assistant|>\n엄마가 왜 그러시는 거 같아?\n<|user|>\n아빠가 화내서요. 그리고 아빠가 엄마랑 우리들 다 때려서 엄마는 항상 우리한테 미안하다고 다 내 잘못이라고 해요.\n<|assistant|>\n엄마가 그런 표정 지으시면 내 기분은 어때?\n<|user|>\n슬퍼요.\n<|assistant|>\n형제자매가 있어?\n<|user|>\n네. 누나 한 명이랑 남동생 한 명 있어요.\n<|assistant|>\n형제자매랑 사이는 어때?\n<|user|>\n끈끈해요.\n<|assistant|>\n평소에 부모님이 형제자매와 나를 다르게 대할 때가 있어?\n<|user|>\n아니요.\n<|assistant|>\n친구들이랑 주로 뭐 하고 놀아?\n<|user|>\n축구하거나 보드게임하는데 지금은 제가 다친 곳이 많아서 보드게임처럼 앉아서 하는 게임만 해요.\n<|assistant|>\n몇 명이서 놀아?\n<|user|>\n몇 명인지는 잘 모르겠지만 여러 명이서 같이 놀아요.\n<|assistant|>\n학교 담임 선생님은 좋아하는 편이야 안 좋아하는 편이야?\n<|user|>\n좋아해요.\n<|assistant|>\n오늘 이야기 해주어서 고마워\n<|assistant|>\n요즘 가장 큰 고민이 있어?\n<|user|>\n네. 아빠요.\n<|assistant|>\n어떤 것 때문에 고민이 되는 거야?\n<|user|>\n무섭고 싫어요. 맨날 우리들 때리고 엄마 때려요. 아빠가 사라졌으면 좋겠어요.\n<|assistant|>\n요즘 행복하다고 느껴?\n<|user|>\n아니요.\n<|assistant|>\n왜 불행하다고 느끼는 거야?\n<|user|>\n아빠랑 같이 사는 게 불행해요.\n<|assistant|>\n커서 되고 싶은 사람이나 꿈이 있다면 무엇인지 말해줄래?\n<|user|>\n잘 모르겠어요.\n<|assistant|>\n오늘 이야기 해주어서 고마워\n<|assistant|>\n주로 나를 돌봐주는 어른은 누구야?\n<|user|>\n엄마요.\n<|assistant|>\n배가 고픈데 먹을 게 없어서 굶은 적 있어?\n<|user|>\n아니요. 굶은 적 없어요.\n<|assistant|>\n어디 아픈 곳이 있는데 병원에 데려가지 않은 적 있어?\n<|user|>\n그런 적 없어요.\n<|assistant|>\n빨래가 되어 있지 않거나 사이즈가 안 맞는 옷이나 신발을 사용한 적 있어?\n<|user|>\n아니요.\n<|assistant|>\n인터넷이나 휴대폰을 가지고 노는 시간은 하루에 몇 시간 정도야?\n<|user|>\n몇 시간 안 해요. 세 시간 정도.\n<|assistant|>\n피시방은 얼마나 자주 가?\n<|user|>\n자주 안 가요.\n<|assistant|>\n부모님이나 학교 선생님 때문에 슬프거나 화난 적 있어?\n<|user|>\n네. 아빠 때문에 슬펐어요.\n<|assistant|>\n아빠의 어떤 말이나 행동이 슬프게 했어?\n<|user|>\n아빠 마음에 안 들면 현관 가서 숫자 세라고 하고 말 더듬으면 화내면서 때리고 그래서 슬펐어요.\n<|assistant|>\n어른들이 나를 어디 가두고 못 나오게 한 적 있어?\n<|user|>\n그런 적은 없어요.\n<|assistant|>\n다른 사람들 앞에서도 혼나거나 다른 아이들과 비교하기도 해?\n<|user|>\n아니요.\n<|assistant|>\n어른들이 나를 집에서 쫓아내거나 버리겠다고 말하거나 진짜 쫓아낸 적 있어?\n<|user|>\n아니요.\n<|assistant|>\n내가 무엇을 하려고 할 때 무조건 안 된다고 하거나, 하기 싫은데도 무조건 하라고 할 때가 있어?\n<|user|>\n아빠가 숫자 세라고 할 때 안 하면 때리고 화내서 해야 돼요.\n<|assistant|>\n마음이 힘들 때 가족에게 내 마음을 이야기해?\n<|user|>\n엄마랑 누나한테 이야기해요.\n<|assistant|>\n주위 어른들 중에 나를 때린 적이 있는 사람이 있어?\n<|user|>\n네. 아빠요.\n<|assistant|>\n아빠가 나를 때리는 이유가 뭐라고 생각해?\n<|user|>\n잘 모르겠어요. 아빠 마음에 안 들면 그러시는 거 같아요.\n<|assistant|>\n아빠한테 어느 부위를 무엇으로 맞았어?\n<|user|>\n발로 채여서 밟히기도 하고 주먹으로 얼굴 맞기도 하고 아빠가 던진 소주잔 깨져서 얼굴이나 무릎에 박힌 적도 있어요.\n<|assistant|>\n평소에 아빠가 자주 때리셔?\n<|user|>\n네. 일주일에 한 번은 꼭 그래요.\n<|assistant|>\n아빠한테 맞아서 얼마나 다쳤어?\n<|user|>\n피멍이 들기도 하고, 피가 많이 난 적도 있고 긁힌 적도 있고 흔적이 남기도 해요.\n<|assistant|>\n이렇게 다친 것 때문에 병원에 가서 치료를 받은 적 있어?\n<|user|>\n네. 엄마랑 같이 가서 치료받았어요.\n<|assistant|>\n병원에 입원한 적도 있어?\n<|user|>\n아니요. 그렇진 않았어요.\n<|assistant|>\n오늘 이야기 해주어서 고마워\n<|assistant|>\n집에서 어른들끼리 소리 지르면서 싸우는 걸 듣거나 본 적 있어?\n<|user|>\n네. 있어요. 일주일에 한 번 이상 그래요.\n<|assistant|>\n누가 어떻게 싸웠어?\n<|user|>\n엄마랑 아빠가 말로 싸우기도 하고, 아빠가 엄마 때려요.\n<|assistant|>\n그때 어른들이 술을 마신 상태였어?\n<|user|>\n아빠만요.\n<|assistant|>\n어른들이 술을 마시고 집에서 무섭게 행동할 때가 많아?\n<|user|>\n네. 아빠가 자주 그래요.\n<|assistant|>\n그렇게 싸울 때 나한테 안 좋은 소리를 하거나 그 상황에서 내가 다친 적도 있어?\n<|user|>\n네. 안 좋은 소리를 할 때도 있고, 물건을 던질 때도 있었어요.\n<|assistant|>\n이런 일 때문에 집에서 나와 있었어야 한 적 있어?\n<|user|>\n네. 한 달에 두 번 정도는 누나랑 동생이랑 방에 숨어있거나 집 앞에 나와있었거나 했어요.\n<|assistant|>\n이런 일 때문에 집에 경찰이나 다른 사람이 온 적 있어?\n<|user|>\n그런 적은 없어요.\n<|assistant|>\n친구들이 나를 자주 놀리거나 괴롭힌다고 느낀 적 있어?\n<|user|>\n아니요.\n<|assistant|>\n언제 마음이 복잡하거나 힘들어?\n<|user|>\n아빠가 숫자 세라고 하거나 엄마 때릴 때요.\n<|assistant|>\n안 좋은 기억 때문에 괴로워서 죽고 싶다는 생각 한 적 있어?\n<|user|>\n제가 죽고 싶은 생각 말고 아빠가 사라졌으면 좋겠다는 생각은 해본 적 있어요.\n<|assistant|>\n잠잘 때 주로 어떤 꿈을 꿔?\n<|user|>\n아빠가 때리는 꿈이랑 우리 집이 빨갛게 변하는 꿈을 꿔요.\n<|assistant|>\n잠들기 힘들고 잠자는 중간에 자주 깼던 이유가 무섭거나 힘들었던 경험 때문이야?\n<|user|>\n네.\n<|assistant|>\n생각하고 싶지 않은데 자꾸 그 기억이 떠오른 적 있어?\n<|user|>\n네. 일주일에 두 번 이상 그래요.\n<|assistant|>\n이런 싫은 기억이 떠오를 때 어떻게 해?\n<|user|>\n몰라요.\n<|assistant|>\n꿈을 꾸고 일어나면 몸 상태는 어때?\n<|user|>\n울고 있어요.\n<|assistant|>\n오늘 이야기 해주어서 고마워\n<|user|>\n그 동안의 이야기만 요약해줘.
            '''

    retrieved = retrieve_documents(query)
    story = ask_llm_with_context(query, retrieved)
    try:
        story = clean_json_output(story)
        story = json.loads(story)
    except json.JSONDecodeError as e:
        print("JSON 파싱 오류:", e)
        print("원본 응답:\n", story)
        exit(1)

    print("== 생성된 이야기 ==")
    prompt = convert_to_prompt_format(story)

    character_main(prompt)
    generate_all_backgrounds(prompt)
    tts_main(prompt)
    webtoon_main(prompt,"temp")