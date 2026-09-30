from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
load_dotenv() # vscode 자체에서 없어도 가져옴. 하지만 안전빵으로 적어두는 게 좋다.

api_key = os.environ["MONOROUTER_API_KEY"].strip() # 공백이나 줄바꿈 삭제함
base_url = "https://monogpt.kr/api/monorouter/v1"

llm = ChatOpenAI(
    model_name='gpt-5.6-terra',
    temperature=0, # 창작하지 않고 쓰겠다.
    # openai_api_key=openai_api_key,
    api_key=api_key,
    base_url=base_url,

)

# response = llm.invoke('나는 윤영선이야. 나는 잘생겼지. 이해했니?')
response = llm.invoke('안녕하세요?')
print(response.content)