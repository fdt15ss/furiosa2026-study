# lg08_Memory02.py 베이스

from langchain_openai import ChatOpenAI
import os

from dotenv import load_dotenv
load_dotenv()

api_key = os.environ['MONOROUTER_API_KEY'].strip()
base_url = 'https://monogpt.kr/api/monorouter/v1'

llm = ChatOpenAI(
    model_name = 'gpt-6-luna',
    temperature=0,
    max_tokens=2000,
    api_key=api_key,
    base_url=base_url,
)

#################### 0. Tool 정의 ####################

from langchain_tavily import TavilySearch

web_search = TavilySearch(
    max_results=5,
)

tools = [web_search]
llm_with_tools = llm.bind_tools(tools)

from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

# 메모리 추가 첫번째
from langgraph.checkpoint.memory import InMemorySaver

#################### 1. State 정의 ####################

class State(TypedDict):
    messages : Annotated[list, add_messages]
    search_result : str
    analysis_result : str

#################### 2. Node 정의 #####################

# 검색 에이전트
def search_agent(state : State):
    print('검색 에이전트 실행')
    question = state['messages'][-1].content

    result = web_search.invoke({
        'query' : question
    })
    return {'search_result' : str(result)}

def analysis_agent(state : State):
    print('분석 에이전트 실행')
    question = state['messages'][-1].content

    response = llm.invoke(
        f'다음 질문을 기존 지식을 바탕으로 분석해줘.\n{question}'
    )
    return {'analysis_result' : response.content}

def merge_agent(state : State):
    result = (
        state['search_result'] + '\n\n' +
        '==============================' + '\n' +
        state['analysis_result']
    )
    return {'messages' : [('assistant', result)]}

#################### 3. Graph 구성 ####################

builder = StateGraph(State)

# builder.add_node('search_agent', search_agent)
# builder.add_node(analysis_agent)
# builder.add_node(merge_agent)

builder.add_node('search', search_agent)
builder.add_node('analysis', analysis_agent)
builder.add_node('merge', merge_agent)

########## 병렬 실행 ##########

builder.add_edge(START, 'search')
builder.add_edge(START, 'analysis')

########## 병합 ##########

builder.add_edge('search', 'merge')
builder.add_edge('analysis', 'merge')

builder.add_edge('merge', END)

# 컴파일
graph = builder.compile()

#################### 6. 그래프 실행 ####################

query = '엔비디아의 미래 전망에 대해 알려줘'

response = graph.invoke(
    {'messages' : [('user', query)]},
)

print('Assistant :', response['messages'][-1].content)

import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO

png_data = graph.get_graph().draw_mermaid_png()     # 그래프 구조 → Mermaid PNG (bytes)

img = Image.open(BytesIO(png_data))

# 그림 출력
plt.figure(figsize=(12, 8))
plt.imshow(img)
plt.axis('off')
plt.show()

def chatbot(state : State):
    return {'messages' : [llm_with_tools.invoke(state['messages'])]}

#################### 4. Graph 구성, Node 추가 ##########

graph_builder = StateGraph(State)
graph_builder.add_node('chatbot', chatbot)

tool_node = ToolNode(tools)

# graph_builder.add_node('tools', tool_node)
graph_builder.add_node('tools', ToolNode(tools))

#################### 5. Edge 추가 #####################
graph_builder.add_edge(START, 'chatbot')

# graph_builder.add_edge('chatbot', 'tools')
graph_builder.add_conditional_edges(
    'chatbot',
    tools_condition
)

graph_builder.add_edge('tools', 'chatbot')

#################### 6. Compile ######################

# 메모리 추가 두번째
memory = InMemorySaver()

graph = graph_builder.compile(checkpointer=memory)

#################### 7. Graph 실행 ####################

# 메모리 추가 세번째
config = {'configurable' : {'thread_id' : 'user01'}}

# 첫번째 질문
response = graph.invoke(
    {'messages' : [('user', '내 이름은 Mr.Song 이야')]},
    config=config
)

print(response['messages'][-1].content)

# 두번째 질문
print('========== 두번째 질문 ==========')
response = graph.invoke({
    'messages' : [('user', '내 이름이 뭐라고 했지?')]},
    config=config)

print(response['messages'][-1].content)