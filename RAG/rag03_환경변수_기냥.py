from langchain_openai import ChatOpenAI
import os
# 윈도우 환경 변수에 키 넣고 저장

llm = ChatOpenAI(
    model_name='gpt-5.6-terra',
    temperature=0, # 창작하지 않고 쓰겠다.
    # openai_api_key=openai_api_key,

)

# response = llm.invoke('나는 윤영선이야. 나는 잘생겼지. 이해했니?')
response = llm.invoke('나는 누구게?')
print(response.content)