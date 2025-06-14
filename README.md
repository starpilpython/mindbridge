# AI 기반 아동 감정 분석 및 상호작용 플랫폼(서강대 AIMBA 프렉티컴)

본 프로젝트는 아동의 감정 상태를 인식하고, 부모에게 임상적 소견 및 애니메이션 기반 콘텐츠를 제공하는 AI 정서지원 시스템입니다.  
실시간 대화, 감정 분석, 요약, 그리고 맞춤형 시각화 콘텐츠를 포함한 전체 파이프라인을 FastAPI 기반으로 구현하였으며, YOLO를 활용한 고속 얼굴 감지 기능도 포함됩니다.

---

## 🎯 주요 기능

- **실시간 아동-AI 대화**: Whisper, GEMMA3, Zonos로 음성 기반 교감
- **얼굴 감정 분석**: YOLO + DeepFace를 통해 정확하고 빠른 얼굴 감정 인식
- **임상 소견 요약**: KoBERT + LoRA 기반 분석 리포트 제공
- **맞춤형 애니메이션**: Stable Diffusion 기반 캐릭터/배경 생성 및 영상 합성
- **부모 대시보드**: 아동의 감정 변화, 요약 리포트, 통계 확인 기능 제공

---

## 🧠 사용 기술

| 분야 | 도구/모델 | 설명 |
|------|-----------|------|
| 백엔드 | FastAPI | 라우터 기반 API 구성 |
| 음성 인식 | Whisper | STT: 음성 → 텍스트 |
| 음성 합성 | Zonos | TTS: 텍스트 → 음성, 클로닝 포함 |
| 대화 모델 | GEMMA3 | 실시간 대화 및 시나리오 줄거리 생성 |
| 감정 분석 | **YOLO + DeepFace** | YOLO로 얼굴 탐지 → DeepFace로 감정 분석 |
| 텍스트 분석 | KoBERT + LoRA | 아동 문답 분석 및 임상 소견 생성 |
| 이미지 생성 | Stable Diffusion | 캐릭터/배경 자동 생성 (디즈니 그림체) |
| 영상 합성 | OpenCV, MoviePy | 이미지 결합 + 자막/음성 삽입 영상 생성 |

---

## 📁 디렉터리 구조

\`\`\`<br>
├── AI_model/<br>
│   ├── KoBert/<br>
│   ├── Rag/<br>
│   ├── Webtoon/<br>
│   └── 준비.txt              # 유일하게 추적되는 AI 파일<br>
│<br>
├── DB/<br>
│   ├── database.py<br>
│   └── models.py<br>
│<br>
├── routers/<br>
│   ├── c3_char_select/<br>
│   ├── c4_call/<br>
│   ├── c5_converse/<br>
│   ├── c6_webtoon/<br>
│   ├── c7_view/<br>
│   ├── c8_diary/<br>
│   └── p4_dashbord/<br>
│<br>
├── statics/<br>
│   └── ... (오디오, 폰트 등 리소스)<br>
│<br>
├── main.py (예상)<br>
├── README.md<br>
└── requirements.txt<br>
\`\`\`<br>

---

## 🚀 실행 방법

<pre> ```bash
# 가상환경 활성화
source myenv/bin/activate

# 필요 패키지 설치
pip install -r requirements.txt

# 서버 실행
python main.py
``` </pre>

---

## 📌 주의사항

- `AI_model/` 내 대부분 파일은 `.gitignore` 처리되며, `준비.txt`만 추적됨
- YOLO 모델은 `ultralytics` 또는 `yolov5-face` 기반 사전학습 가중치 필요
- GPU 기반 실행 환경 권장 (특히 Whisper, Stable Diffusion)

---

## 🧾 참고

- 프로젝트 구조 및 기술 흐름: `프랙티컴 백엔드 설명.pdf` 참고
- 부모/자녀 사이드 구조 분리 및 API 흐름 반영
- YOLO 기반 얼굴 탐지 + DeepFace 감정 분석 → 속도 및 정확도 개선 구조

---

## 🎯 최종 목표

AI 기술을 통해 아이의 감정을 정량적으로 이해하고,  
부모-자녀 관계 강화 및 아동의 건강한 정서 발달을 지원하는  
**미래형 정서 인지 기반 상호작용 플랫폼** 구축
