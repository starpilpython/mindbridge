from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from llama_cpp import Llama
import atexit
import json

# 모델 로드
llm = Llama(
    n_ctx=131072,
    model_path="/home/elicer/mindbridge/AI_model/gemma-3-4B-it-QAT-Q4_0.gguf",
    n_gpu_layers=26,
    use_mlock=True,
    verbose=False
)

def safe_close():
    global llm
    try:
        if llm and hasattr(llm, "close") and callable(llm.close):
            llm.close()
    except Exception as e:
        print("LLM 종료 중 오류:", e)
    finally:
        llm = None

atexit.register(safe_close)

model = SentenceTransformer("jhgan/ko-sbert-nli")
client = QdrantClient(path="/home/elicer/mindbridge/AI_model/Rag/qdrant_local")
collection_name = "stories"

dimension = model.get_sentence_embedding_dimension()
if not client.collection_exists(collection_name):
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=dimension, distance=Distance.COSINE)
    )

def retrieve_documents(query: str, top_k: int = 5):
    query_vec = model.encode([query], convert_to_numpy=True)[0]
    results = client.search(
        collection_name=collection_name,
        query_vector=query_vec.tolist(),
        limit=top_k,
        with_payload=True
    )
    return [hit.payload["text"] for hit in results if "text" in hit.payload]

def ask_llm_with_context(query: str, context_docs: list):
    context = "\n\n".join(context_docs)

    system_prompt = (
        "너는 유치원~초등 저학년 어린이들이 읽는 감정 기반 만화 시나리오 작가야. "
        "사용자가 준 이야기를 바탕으로 5컷의 JSON 구조로 된 만화 시나리오를 작성해줘. "
        "각 컷은 아이들이 이해하기 쉬운 이야기 흐름과 감정 변화를 담고 있어야 해.\n\n"
        "각 컷에는 다음 필드를 반드시 포함해야 해:\n"
        "- cut: 컷 번호 (1~5)\n"
        "- 배경: 구체적인 장소와 분위기 묘사 (한 문장)\n"
        "- 나레이션: 상황이나 감정을 설명하는 문장 (1문장 이상, 필수)\n"
        "- 대사: 반드시 2명 이상의 화자가 등장하고, 각 대사는 2문장 이상으로 구성. 형식은 다음과 같아:\n"
        "  {\"화자\": \"역할명\", \"말\": \"대사\", \"행동\": \"행위\"}\n\n"
        "⚠️ 나레이션만 있는 컷은 만들지 마. 반드시 대사도 포함되어야 해.\n"
        "⚠️ 대사는 각 컷마다 각 인물마다 2문장 이상 말해야해 서로 다른 인물 간의 상호작용으로 구성하고 반복되지 않도록 해.\n"
        "⚠️ 전체 5컷은 5컷 단위로 하나의 장소에서 진행되도록 구성해줘. 예: 컷 1~5는 방, 6~10은 거실 등.\n\n"
        "사용 가능한 역할명:\n"
        "- 호야, 친구, 선생님, 엄마, 아빠, 할머니, 할아버지, 아기, 강아지, 가게주인\n\n"
        "응답은 반드시 유효한 JSON 리스트로 출력해. 키 이름은 절대 바꾸지 마."
    )

    json_template = [
        {
            "cut": 0,
            "배경": "호야의 방, 햇살이 들어오는 오후",
            "나레이션": "호야는 조용히 누워 하루를 되새긴다.",
            "대사": [
                {"화자": "호야", "말": "오늘은 뭔가 속이 복잡해. 엄마 나 진짜 힘들어", "행동": "침대에 누워 천장을 바라본다"},
                {"화자": "엄마", "말": "호야야, 무슨 일 있니? 나에게 말 좀 하려무나, 힘들때는 서로 이야기 하는게 중요해", "행동": "걱정스런 얼굴로 다가온다"}
            ]
        }
    ] + [{"cut": x, "배경": "", "나레이션": "", "대사": []} for x in range(1, 6)]

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"""
다음은 JSON 템플릿이야. 아래 형식 그대로 값을 채워줘. 키는 절대 바꾸지 말고, 유효한 JSON으로 출력해.

{json.dumps(json_template, ensure_ascii=False, indent=2)}

[문맥]
{context}

[이야기]
{query}
"""}
    ]

    response = llm.create_chat_completion(
        messages=messages,
        temperature=0.5,
        stop=["[끝]", "THE END"],
        max_tokens=8000
    )
    return response["choices"][0]["message"]["content"]

def clean_json_output(raw: str):
    raw = raw.strip()
    start_idx = raw.find("[")
    if start_idx != -1:
        raw = raw[start_idx:]
    if raw.startswith("```json"):
        raw = raw[len("```json"):].strip()
    if raw.endswith("```"):
        raw = raw[:-3].strip()
    return raw

def fix_background_by_block(story_json):
    for i in range(0, 10, 5):
        base_bg = story_json[i]["배경"]
        for j in range(i, i + 5):
            story_json[j]["배경"] = base_bg
    return story_json

def convert_to_prompt_format(simple_json_list):
    base_prompt_dict = {
        "호야": "a cheerful boy wearing a red hoodie and navy pants, short black hair, watercolor style, studio ghibli",
        "친구": "a lively boy in a yellow shirt and green shorts, short brown hair, watercolor style, studio ghibli",
        "엄마": "a kind mother in an apron and light blue blouse, short hair, watercolor style, studio ghibli",
        "아빠": "a gentle father in a white shirt and khaki pants, glasses, watercolor style, studio ghibli",
        "선생님": "a warm teacher with a rose cardigan and long brown skirt, holding a storybook, watercolor style, studio ghibli",
        "할머니": "a wise elderly woman in a floral blouse and gray skirt, tied white hair, watercolor style, studio ghibli",
        "할아버지": "a calm elderly man in a beige vest and slacks, white beard, watercolor style, studio ghibli",
        "아기": "a baby in a yellow onesie crawling on the floor, watercolor style, studio ghibli",
        "강아지": "a fluffy white dog wagging its tail, watercolor style, studio ghibli",
        "가게주인": "a friendly man with an apron behind a counter, watercolor style, studio ghibli"
    }

    converted = []

    for item in simple_json_list:
        cut_num = item.get("cut")
        cut_id = f"cut{cut_num:02d}"

        background_prompt = item.get("배경", "").strip()
        narration_text = item.get("나레이션", "").strip()
        dialogues = item.get("대사", [])

        segments = []
        if narration_text:
            segments.append({
                "type": "나레이션",
                "text": narration_text
            })

        characters_dict = {}

        for d in dialogues:
            role = d.get("화자", "").strip()
            line = d.get("말", "").strip()
            if not role or not line:
                continue

            segments.append({
                "type": "대사",
                "speaker": role,
                "text": line
            })

            if role not in characters_dict:
                characters_dict[role] = []
            characters_dict[role].append(line)

        characters = []
        for role, lines in characters_dict.items():
            prompt = base_prompt_dict.get(role, f"a child named {role}, ghibli style")
            characters.append({
                "name": role,
                "prompt": prompt,
                "ref": f"{role}.png",
                "lines": lines
            })

        converted.append({
            "cut_id": cut_id,
            "background_prompt": background_prompt,
            "characters": characters,
            "segments": segments
        })

    return converted

# 실행
if __name__ == "__main__":
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

    story = fix_background_by_block(story)
    prompt = convert_to_prompt_format(story)

    print(prompt)
