from langchain_openai import ChatOpenAI
from config import SUMMARIZER, API_KEY, BASE_URL

summarizer = ChatOpenAI(
    base_url=BASE_URL,
    model=SUMMARIZER,
    api_key=API_KEY,
    top_p=0.3
)