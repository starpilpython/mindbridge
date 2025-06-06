# 각 화면별 라우터를 모두 집합하는 코드 
from fastapi import APIRouter
from .c3_char_select.router import router as c3_char_selecter

# 메인 라우터 호출 
router = APIRouter()

router.include_router(c3_char_selecter, prefix="/child", tags=["child"])


