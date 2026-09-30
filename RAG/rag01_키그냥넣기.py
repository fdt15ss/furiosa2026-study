from langchain_openai import ChatOpenAI

openai_api_key = ''
# 원래는 코드에다 api key 직접 넣으면 절대 안됨.
llm = ChatOpenAI(
    model_name='gpt-5.6-terra',
    temperature=0, # 창작하지 않고 쓰겠다.
    openai_api_key=openai_api_key,

)

# response = llm.invoke('나는 윤영선이야. 나는 잘생겼지. 이해했니?')
response = llm.invoke('나는 누구게?')
print(response.content)