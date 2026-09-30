# LCEL = Lanchain Expression Language
# chain = prompt | model | output_parser

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.environ["MONOROUTER_API_KEY"].strip() # 공백이나 줄바꿈 삭제함
base_url = "https://monogpt.kr/api/monorouter/v1"

prompt = PromptTemplate.from_template("{topic}에 대해 쉽게 설명해주세요.")

model = ChatOpenAI(
    # model_name='gpt-5.6-terra',
    model_name='gpt-5.6-terra',
    temperature=0, # 창작하지 않고 쓰겠다.
    # openai_api_key=openai_api_key,
    api_key=api_key,
    base_url=base_url,

)

chain = prompt | model # 랭체인에서 재정의 했다.
input = {"topic" : "양자컴퓨터 학습 원리"}

response = chain.invoke(input)
print(response.content)