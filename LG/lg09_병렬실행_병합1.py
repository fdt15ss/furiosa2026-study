from typing import TypedDict
from langgraph.graph import StateGraph, START, END

#1. State 정의
class State(TypedDict):
    question : str
    search_result : str
    analysis_result : str
    final_result : str

#2. 검색노드
def search_agent(state: State):
    print("검색 에이전트 실행")
    return {"search_result" : "엔비디아는 AI GPU 시장의 주요 기업이다."}

#3. 분석 노드
def analysis_agent(state : State):
    print("분석 에이전트 실행")
    return {"analysis_result" : "AI 반도체 수요 증가가 성장 요인이다."}

#4. 결과 통합 노드
def merge_agent(state : State):
    result = (
        state["search_result"] + "\n" +
        state["analysis_result"]
    )
    return {'final_result' : result}

#5. 그래프 구성
builder = StateGraph(State)

builder.add_node("search", search_agent)
builder.add_node("analysis", analysis_agent) # 노드 이름이 빠져있으면 자동으로 그 함수명이 노드 이름으로 된다.
builder.add_node("merge", merge_agent)

############## 병렬 실행 ###################
builder.add_edge(START, "search")
builder.add_edge(START, "analysis")

################# 병합 ###################
builder.add_edge("search", "merge")
builder.add_edge("analysis", "merge")

builder.add_edge("merge", END)

# 컴파일
graph = builder.compile()

# 실행
response = graph.invoke({
    "question" : "엔비디아의 미래 전망은?"
})

print(response["final_result"])


import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO

png_data = graph.get_graph().draw_mermaid_png()

img = Image.open(BytesIO(png_data))
# with open(path + 'lg07_ToolNode.png', 'wb') as f:
#     f.write(png_data)

# 그림 출력
plt.figure(figsize=(12, 8)) # 인치
plt.imshow(img)
plt.axis('off')
plt.show()