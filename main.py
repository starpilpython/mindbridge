# FastApi 코드 가동하기 위한 코드
from fastapi import FastAPI
import uvicorn # 배포용 서버 

app = FastAPI()


@app.get("/")
async def read_root():
    return {"Hello": "World"}

@app.get("/items/{item_id}")
async def read_item(item_id: int, q: str = None):
    return {"item_id": item_id, "q": q}


if __name__ == '__main__':
    # 배포용으로 설정 
    uvicorn.run(app,host='0.0.0.0',port=1025, log_level="info")
