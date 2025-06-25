# 감정 인식 
from ultralytics import YOLO
from deepface import DeepFace
from sklearn.metrics.pairwise import cosine_similarity 
import numpy as np
import cv2
import os

# YOLO 모델 로드 (사전 학습된 yolov8n.pt 모델 사용)
yolo_model = YOLO('yolov8n.pt')
# 기준 얼굴 임배딩 벡터 (서버 시작 시 한번만 계산)
target_embedding = None

# 서버 시작 시 기준 얼굴 여러장을 로드하여 평균임베딩 벡터 생성
def load_target_faces():
  global target_embedding # 전역변수 설정 
  embedding_list = [] # 임베딩 벡터 모음 리스트 

  folder_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'statics', 'webcam_targets')
  folder_path = os.path.abspath(folder_path)  # 경로 정규화
 
  # 폴더 내 모든 이미지 순환
  for fname in os.listdir(folder_path):
    img_path = os.path.join(folder_path, fname)
    try:
      emb = DeepFace.represent(img_path=img_path,model_name='Facenet')[0]['embedding']
      embedding_list.append(emb)
      print(f'로딩성공:{fname}')
    except Exception as e:
      print(f'{fname} 임베딩 실패: {e}')

  # 임베딩 평균 내기 
  if embedding_list:
    target_embedding = np.mean(embedding_list,axis=0)
    print(f"총 {len(embedding_list)}개의 기준 얼굴 임베딩 평균 완료")
  else:
    print("기준 얼굴이 없습니다.")

# 이미지에서 얼굴 인식 및 감정 분석 수행
def detect_faces(img_bytes):
    global target_embedding
    np_img = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)  # BGR 이미지
    results = yolo_model(frame)
    faces = []

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        width = x2 - x1
        height = y2 - y1

        # 박스 축소 (중심 기준으로 80%)
        shrink_ratio = 0.8
        center_x = x1 + width / 2
        center_y = y1 + height / 2
        new_w = int(width * shrink_ratio)
        new_h = int(height * shrink_ratio)
        new_x1 = max(int(center_x - new_w / 2), 0)
        new_y1 = max(int(center_y - new_h / 2), 0)
        new_x2 = min(int(center_x + new_w / 2), frame.shape[1])
        new_y2 = min(int(center_y + new_h / 2), frame.shape[0])

        face_crop = frame[y1:y2, x1:x2]

        try:
            rep = DeepFace.represent(face_crop, model_name="Facenet", enforce_detection=False)[0]["embedding"]
            sim = cosine_similarity([target_embedding], [rep])[0][0]

            if sim >= 0.6:
                emo = DeepFace.analyze(face_crop, actions=["emotion"], enforce_detection=False)[0]["dominant_emotion"]
                print(f"유사도: {sim:.2f}, 감정: {emo}")
                faces.append({
                    "emotion": emo,
                    "box": {
                        "x": new_x1,
                        "y": new_y1,
                        "width": new_x2 - new_x1,
                        "height": new_y2 - new_y1
                    }
                })
            else:
                print("유사도 낮음 - 다른 사람으로 간주")
                faces.append({
                    "emotion": "NO"
                })

        except Exception as e:
            print("오류:", e)
            continue

    print(faces)
    return faces