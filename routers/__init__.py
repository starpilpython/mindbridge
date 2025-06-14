# 각 화면별 라우터를 모두 집합하는 코드 
from fastapi import APIRouter
from .c3_char_select.router import router as c3_char_selecter
from .c4_call.router import router as c4_call
from .c5_converse.router import router as c5_converse
from .c6_webtoon.router import router as c6_webtoon
from .c7_view.router import router as c7_view
from .p4_dashbord.router import router as p4_dashbord

# 메인 라우터 호출 
router = APIRouter()

# 라우터 추가 
router.include_router(c3_char_selecter, prefix="/child", tags=["child"])
router.include_router(c4_call, prefix="/child", tags=["child"])
router.include_router(c5_converse, prefix="/child", tags=["child"])
router.include_router(c6_webtoon, prefix="/child", tags=["child"])
router.include_router(c7_view, prefix="/child", tags=["child"])
router.include_router(p4_dashbord, prefix="/parent", tags=["parent"])