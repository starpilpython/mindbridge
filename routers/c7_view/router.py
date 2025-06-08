# 오늘자 생성된 애니메이션을 화면에 로드 

###################################################################

# Fastapi 라우터 설정하는 패키지
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from datetime import date
from urllib.parse import quote
###################################################################

#Fastapi 가동 
router = APIRouter()
today = date.today()

@router.get("/get_video_url")
async def webtoon():
  # 해당 부분은 하드 코딩으로 추후 변경 필요 
  name = '윤성필'
  # 오늘 날짜 지정 필요
  today = date.today().strftime("%Y-%m-%d")  # 예: 2025-06-08
  filename = f"webtoon/video/{name}_{today}.mp4"
  encoded_filename = quote(filename)
  video_url = f"https://nfredmpfjwwrknjr.tunnel.elice.io/statics/{encoded_filename}"
  return JSONResponse({'video_url': video_url})