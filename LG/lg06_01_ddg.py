# lg05 카피

from langchain_openai import ChatOpenAI
import os

from dotenv import load_dotenv
load_dotenv()

api_key = os.environ['MONOROUTER_API_KEY'].strip()
base_url = 'https://monogpt.kr/api/monorouter/v1'

llm = ChatOpenAI(
    model_name='gpt-6.1-sol',
    temperature=0,
    max_tokens=2000,
    api_key=api_key,
    base_url=base_url,
)

#################### 0. Tool 정의 ####################
# pip install langchain-tavily
# from langchain_tavily import TavilySearch

# # 웹 검색 도구 생성 (검색창에서 검색하는 것과 같은 단계다.)
# web_search = TavilySearch(
#     max_results=5,
# )

# pip install ddgs
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_community.tools import DuckDuckGoSearchRun

web_search = DuckDuckGoSearchRun(
    max_results=5,
)


# result = web_search.invoke({
#     'query' : '삼성전자의 오늘 주가 얼마'
# })
# print(result)
# print(result['results'][-1]['content'])

# exit()

tools = [web_search]

llm_with_tools = llm.bind_tools(tools)          # 위에 llm 모델과 tools라는 검색 엔진을 서로 엮어주는 과정

from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

#################### 1. State 정의 ####################
class State(TypedDict):
    messages : Annotated[list, add_messages]

#################### 2. node 정의 ####################
def chatbot(state : State):
    return{'messages' : [llm_with_tools.invoke(state['messages'])]}


#################### 3. graph 구성 ####################
# 그래프 생성
graph_builder = StateGraph(State)

graph_builder.add_node('chatbot', chatbot)

tool_node = ToolNode(tools)            # 위에 정의한 tools를 그대로 상속받아 그 모든 기능을 ToolNode가 쓸수있게하고 node화 시켰다.
# graph_builder.add_node('tools', tool_node)
graph_builder.add_node('tools', ToolNode(tools))        # 윗 줄과 같은데 표현만 다르다. 편한거 쓰면 된다.

#################### 4. edge 추가 ####################
graph_builder.add_edge(START, 'chatbot')

# graph_builder.add_edge('chatbot', 'tools')
graph_builder.add_conditional_edges(
    'chatbot',
    tools_condition,          # tool_calls가 있냐?? --------> 있으면 'tools' 반환, 없으면 'END' 반환(tools_condition에 'END'가 포함되서 따로 edge추가 필요없음)

)

graph_builder.add_edge('tools', 'chatbot')


#################### 5. graph compile ####################
graph = graph_builder.compile()

#################### 6. graph 실행 ####################
query = '삼성전자의 오늘 주가 얼마'

response = graph.invoke({'messages' : [('user', query)]})

print('AI : ', response['messages'][-1].content)