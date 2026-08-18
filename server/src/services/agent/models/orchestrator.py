from config import BASE_URL, API_KEY, LVLM
from langchain_openai import ChatOpenAI

orchestrator = ChatOpenAI(
    model=LVLM,
    base_url=BASE_URL,
    api_key=API_KEY,
    top_p=0.3,
    max_retries=3,
)
