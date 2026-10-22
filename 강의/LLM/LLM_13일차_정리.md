# LLM 13일차 정리 — 랭그래프 툴 콜링(Tool Calling)

> 녹취 중 잡담·음성인식 오류는 제외하고, 파이썬/랭그래프 학습 내용 위주로 정리했습니다.

---

## 0. 오늘의 큰 흐름

| 순서 | 내용 | 핵심 키워드 |
|---|---|---|
| 1 | 전날(11일차) 복습 | State, Node, Edge, add_messages(리듀서), compile, invoke |
| 2 | 외부 기능 연결 — 웹 검색 툴 | Tavily, DuckDuckGo, Tool |
| 3 | 모델에 툴 연결 | `bind_tools` |
| 4 | 툴을 노드로 만들기 | `ToolNode` |
| 5 | 조건부 엣지 | `add_conditional_edges`, `tools_condition` |
| 6 | 그래프 시각화 | `get_graph().draw_mermaid_png()` |
| 7 | 파이썬 기초 보충 | `with open(...)`, 바이너리 쓰기 |

오늘 핵심 한 줄: **"툴 콜링을 위해 `bind_tools` + `ToolNode` + `tools_condition` 세 가지를 이해하면 된다."**

---

## 1. 복습 — 랭그래프 기본 구조 (11일차)

랭그래프는 **State → Node → Edge** 순서로 만든다.

- **State**: 그래프 전체에서 공유하는 데이터. `TypedDict`를 상속해서 정의
- **Node**: 함수로 구성. **State를 입력으로 받아** 갱신할 값을 딕셔너리로 반환
- **Edge**: 노드와 노드를 연결 (`add_edge`)
- **리듀서 `add_messages`**: 메시지 리스트를 덮어쓰지 않고 **append(누적)** 해 준다
  - 리듀서가 없으면 리스트가 매번 덮어써짐 → `Annotated[list, add_messages]`로 해결
- `compile()` 후 `invoke()`하면 메시지가 순서대로 쌓인다 (질문 → 답변 → … ), 각 메시지는 `HumanMessage` / `AIMessage` 타입과 `content`를 가진다.

```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

class State(TypedDict):
    messages: Annotated[list, add_messages]   # 리듀서: 메시지를 계속 append

def chatbot(state: State):                    # State를 받아서
    return {"messages": [llm.invoke(state["messages"])]}  # 갱신할 값을 반환
```

> 용어 정리: `add_node` / `add_edge`는 케라스 함수형 API에서 레이어를 연결하는 것과 비슷한 개념.
> 노드 이름은 자유롭게 지어도 된다 (오늘 수업에서는 빌더 변수명을 `graph_builder`로 통일).

---

## 2. 툴 콜링(Tool Calling)이 필요한 이유

- LLM은 **학습 시점(보통 출시 2~3달 전)** 까지의 데이터만 알고 있다 → 최신 정보(오늘 주가 등)를 모른다.
- ChatGPT 웹에서는 이미 내부적으로 검색 기능이 같이 돌고 있지만, **API로 모델을 쓸 때는 직접 검색 기능을 붙여줘야** 하는 경우가 많다.
- 그래서 모델에 **외부 기능(툴)** 을 연결해 주는 것이 툴 콜링.
- 툴을 쓸지 말지는 **모델(LLM)이 스스로 판단**한다 → 이것이 에이전트의 핵심.

---

## 3. 웹 검색 툴 ① — Tavily

- 한 달 **1,000회까지 무료**, API 키 발급 필요 (`.env`에 저장)
- 순수 **웹 검색**일 뿐, AI 모델이 도는 것이 아님

```python
from langchain_community.tools.tavily_search import TavilySearchResults

web_search = TavilySearchResults(max_results=5)   # 결과 개수 제한
result = web_search.invoke("삼성전자 주가")
```

### 검색 결과에서 필요한 내용만 꺼내기 (딕셔너리/리스트 탐색)

결과는 **딕셔너리를 원소로 갖는 리스트** 형태다. 한 겹씩 파고 들어가면 된다.

```python
print(result)                    # 전체 구조 확인
print(result[-1])                # 리스트의 마지막 원소(딕셔너리)
print(result[-1]["content"])     # 그 딕셔너리의 content 키 → 본문만 출력
```

- 리스트 → 인덱스(`[-1]`, `[0]`) / 딕셔너리 → 키(`["content"]`)
- 검색 시점·검색엔진마다 결과가 조금씩 다른 것은 정상 (네이버도 사람마다 다르게 나오는 것과 같음)

---

## 4. 모델에 툴 연결 — `bind_tools`

```python
tools = [web_search]                      # 리스트 → 툴을 2개 이상도 넣을 수 있음
llm_with_tools = llm.bind_tools(tools)    # 모델 + 툴을 묶은 새 모델
```

- 어제까지: **모델만** 있었음 → 오늘: **"모델 + 툴이 연결된 모델"** 을 준비
- 이 단계에서는 연결만 해둔 것이고, 실제 실행은 나중에 `invoke` 할 때 이뤄진다.
- 챗봇 노드는 이제 `llm`이 아니라 **`llm_with_tools`** 를 호출하도록 바꾼다.

```python
def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}
```

---

## 5. 툴을 노드로 — `ToolNode`

랭그래프에서 **툴(웹 검색)도 하나의 노드**로 만들어 그래프에 붙인다.

```python
from langgraph.prebuilt import ToolNode, tools_condition

tool_node = ToolNode(tools)                 # tools 리스트를 받아 노드로 만들어 줌
graph_builder.add_node("chatbot", chatbot)  # 노드 1
graph_builder.add_node("tools", tool_node)  # 노드 2 (툴 노드)
```

- 한 줄로 `graph_builder.add_node("tools", ToolNode(tools))`처럼 써도 동일
- `langgraph.prebuilt`에는 **`ToolNode`**, **`tools_condition`** 이 이미 만들어져 있음
  → LLM 쪽은 "잘 만들어진 도구를 어떻게 가져다 쓰느냐"가 중요
  (모델 쪽은 수학·통계·알고리즘 지식이 도움이 되고, 랭체인/랭그래프 쪽은 활용 능력이 중요)

---

## 6. 조건부 엣지 (Conditional Edge) ⭐

### 6-1. 왜 필요한가
- 단순 엣지(`START → chatbot → tools → END`)는 **무조건 검색**하고 끝남
- 우리가 원하는 흐름:
  - 챗봇이 **검색이 필요하다고 판단** → `tools` 노드로 이동
  - **필요 없다고 판단** → 바로 `END`
  - 툴 실행 후에는 **다시 `chatbot`으로 돌아가서** 결과를 바탕으로 최종 답변 생성

### 6-2. 코드

```python
graph_builder.add_edge(START, "chatbot")

graph_builder.add_conditional_edges(
    "chatbot",          # 이 노드가 끝난 뒤에
    tools_condition,    # 이 조건 함수로 판단해서
)                       # → tools 로 갈지, END 로 갈지 결정

graph_builder.add_edge("tools", "chatbot")   # 툴 실행 후 다시 챗봇으로
```

- `tools_condition`: 챗봇의 마지막 메시지에 **툴 호출 요청(tool_calls)이 있으면 `"tools"`**, 없으면 **`END`** 로 보내 줌
- 판단 주체는 **모델(LLM)** 이다 (우리가 if문으로 직접 판단하지 않음)
- `END`로 가는 경로는 `tools_condition`이 자동 처리하므로 **`END` 엣지를 따로 쓰지 않아도 된다**
- 엣지는 두 종류로 기억: ① 일반 엣지 `add_edge` ② 조건부 엣지 `add_conditional_edges` (= "조건부 엣지")

### 6-3. 개발자 기본기
> 조건문(`if`)과 반복문(`for`) 두 가지만 확실히 알면 어떤 언어로든 개발이 가능하다.
> 조건부 엣지 = 그래프 버전의 `if`, "툴 → 챗봇" 되돌아오는 구조 = 그래프 버전의 반복.

---

## 6-A. `tools_condition` 자세히 풀어보기 🔍

### 한 문장 정의
> **`tools_condition`은 "챗봇의 마지막 답변이 툴을 쓰겠다는 요청인가?"를 검사해서, 다음에 갈 노드 이름(`"tools"` 또는 `END`)을 돌려주는 미리 만들어진 분기 함수(라우터)다.**

조건부 엣지는 **"어디로 갈지 결정하는 함수"** 를 필요로 한다. 직접 if문으로 짜는 대신, 툴 콜링용으로 자주 쓰는 로직을 랭그래프가 `tools_condition`으로 이미 만들어 둔 것이다.

### `tool_calls`란? (먼저 알아둘 개념) 📌

> **`tool_calls` = "모델이 직접 검색을 하는 게 아니라, '이 툴을 이 값으로 실행해 주세요'라고 적어서 보내는 요청서(주문서)"**

**핵심 포인트: LLM은 툴을 직접 실행하지 못한다.**
모델이 하는 일은 *"검색이 필요하네 → `web_search`라는 툴을 `"삼성전자 주가"`라는 인자로 호출해 줘"* 라는 **요청을 만드는 것**까지다. 실제 실행은 `ToolNode`가 한다.

#### 어디에 들어 있나?
- 모델의 답변인 **`AIMessage` 객체의 속성**이다. (`state["messages"][-1].tool_calls`)
- 일반 답변이면 `content`에 텍스트가 들어가고 `tool_calls`는 **빈 리스트(`[]`)**
- 툴이 필요하다고 판단하면 `content`는 비어 있고 `tool_calls`에 **호출 요청 목록**이 들어간다. (리스트이므로 한 번에 여러 개도 가능)

#### 생김새 예시

```python
# ① 일반 답변 (툴 불필요)
AIMessage(content="안녕하세요! 무엇을 도와드릴까요?", tool_calls=[])

# ② 툴 호출 요청 (검색 필요)
AIMessage(
    content="",
    tool_calls=[
        {
            "name": "tavily_search_results_json",      # 어떤 툴을 쓸지 (툴 이름)
            "args": {"query": "오늘 삼성전자 주가"},      # 툴에 넘길 인자
            "id": "call_abc123",                        # 호출 요청 고유 ID
            "type": "tool_call",
        }
    ],
)
```

| 필드 | 의미 |
|---|---|
| `name` | 호출할 **툴 이름** (`bind_tools`로 연결해 둔 툴 중에서 모델이 고름) |
| `args` | 툴에 넘길 **인자** (검색어 등) — 모델이 질문을 보고 직접 만들어 냄 |
| `id` | 요청 고유 번호 — 나중에 툴 결과(`ToolMessage`)가 **어떤 요청에 대한 답인지** 짝을 맞추는 데 사용 |

#### 이름이 `tool_calls`인 이유
- **tool**(도구) + **call**(호출) + **s**(여러 개 가능) = **"도구 호출 요청들"**
- 수업에서 말한 **"툴 콜링(Tool Calling)"** 이 바로 이 `tool_calls`를 주고받는 방식이다.

#### 전체 메시지 흐름에서의 위치

```
HumanMessage   : "오늘 삼성전자 주가 알려줘"
AIMessage      : content="" , tool_calls=[{name: 검색툴, args: {query: ...}}]   ← 모델의 "요청서"
ToolMessage    : (ToolNode가 검색을 실행한 결과 텍스트)                          ← 툴의 "실행 결과"
AIMessage      : "삼성전자 주가는 ○○원입니다."  (tool_calls=[])                  ← 최종 답변
```

→ 마지막 AIMessage는 `tool_calls`가 비어 있으므로 `tools_condition`이 `END`로 보낸다.

#### 직접 확인해 보는 법
```python
response = llm_with_tools.invoke("오늘 삼성전자 주가 알려줘")
print(response.tool_calls)     # 검색이 필요하다고 판단하면 요청 목록이 출력됨
print(response.content)        # 이때 content는 보통 빈 문자열
```
- `llm`(툴 연결 전)로 호출하면 `tool_calls`는 항상 비어 있다 → **`bind_tools`로 툴을 알려줘야** 모델이 요청서를 만들 수 있다.

### 동작 순서

```
chatbot 노드 실행 (llm_with_tools.invoke)
        │
        ▼
state["messages"]의 마지막 메시지(AIMessage)를 확인
        │
        ├─ tool_calls 가 있다 (모델이 "검색해야겠다"고 판단)  → "tools" 반환 → ToolNode 실행
        │
        └─ tool_calls 가 없다 (모델이 바로 답변을 완성)        → END 반환   → 그래프 종료
```

1. `llm_with_tools`(= `bind_tools`로 툴이 연결된 모델)가 질문을 보고 **검색이 필요하면 답변 대신 "이 툴을 이 인자로 호출해줘"라는 요청**을 만든다. 이 요청이 `AIMessage.tool_calls`에 담긴다.
2. `tools_condition`은 **그 `tool_calls`가 비어 있는지만** 본다.
3. 비어 있지 않으면 → `"tools"` 노드로, 비어 있으면 → `END`로 보낸다.

즉 **"검색할지 말지"를 판단하는 주체는 모델**이고, `tools_condition`은 **그 판단 결과를 읽어서 길을 안내하는 역할**이다. (모델 = 판단, `tools_condition` = 길 안내)

### 직접 만든다면? (같은 동작의 코드)

`tools_condition`이 내부적으로 하는 일은 아래와 거의 같다.

```python
from langgraph.graph import END

def my_tools_condition(state: State):
    last_message = state["messages"][-1]          # 마지막 메시지 = 챗봇이 방금 한 답변
    if getattr(last_message, "tool_calls", None): # 툴 호출 요청이 있으면
        return "tools"                            # → 툴 노드로
    return END                                    # 없으면 → 종료

graph_builder.add_conditional_edges("chatbot", my_tools_condition)
```

- 오늘 수업의 "조건문(`if`)만 알면 된다"는 말이 그대로 적용되는 부분
- 이렇게 직접 만들 수도 있지만, **기본 구조에서는 `tools_condition`을 import 해서 쓰면 끝**

### 함께 알아둘 점

| 항목 | 설명 |
|---|---|
| 반환값 | 노드 이름 문자열 `"tools"` 또는 `END` |
| 노드 이름 규칙 | 기본적으로 **툴 노드 이름을 `"tools"`로** 지어야 연결된다 (다른 이름이면 `path_map`으로 매핑하거나 직접 조건 함수를 작성) |
| `END` 엣지 불필요 | 종료 분기를 `tools_condition`이 처리하므로 `add_edge("chatbot", END)`를 따로 쓰지 않는다 |
| 쌍으로 쓰는 것 | `ToolNode` — `tools_condition`이 `"tools"`로 보내면 `ToolNode`가 **실제로 툴을 실행**하고 결과를 메시지(`ToolMessage`)로 State에 추가 |
| 되돌아오는 엣지 | `add_edge("tools", "chatbot")` — 툴 결과를 보고 챗봇이 **최종 답변**을 만들게 함 |

### 역할 분담 한눈에 보기

| 구성요소 | 하는 일 | 비유 |
|---|---|---|
| `bind_tools` | 모델에게 "이런 툴들을 쓸 수 있다"고 알려줌 | 연장통을 쥐어줌 |
| 챗봇 노드(LLM) | 툴이 필요한지 **판단**하고 호출 요청 생성 | 현장 반장이 "망치 가져와" 지시 |
| **`tools_condition`** | 지시가 있는지 확인해서 **갈림길 안내** | 교통 신호등 |
| `ToolNode` | 요청받은 툴을 **실제 실행** | 연장을 들고 일하는 작업자 |
| `add_edge("tools","chatbot")` | 결과를 챗봇에게 **돌려줌** | 작업 결과 보고 |

### 실행 예시 (흐름 추적)

**① 검색이 필요한 질문: "오늘 삼성전자 주가 알려줘"**
1. `chatbot` → 모델이 검색 필요하다고 판단 → `tool_calls` 생성
2. `tools_condition` → `"tools"` 반환
3. `tools` 노드 → 웹검색 실행, 결과를 State에 추가
4. `tools → chatbot` → 검색 결과를 바탕으로 최종 답변 생성 (이번엔 `tool_calls` 없음)
5. `tools_condition` → `END` 반환 → 종료

**② 검색이 필요 없는 질문: "안녕, 너 이름이 뭐야?"**
1. `chatbot` → 바로 답변 (`tool_calls` 없음)
2. `tools_condition` → `END` 반환 → 종료

### 자주 하는 실수 / 디버깅 팁
- **검색을 안 한다** → `tools_condition` 문제가 아니라 **모델이 툴이 필요 없다고 판단**한 경우가 많다. (질문에 "최신/오늘/실시간" 같은 단서를 넣거나, 챗봇 노드가 `llm`이 아닌 **`llm_with_tools`** 를 쓰고 있는지 확인)
- **`tools` 노드로 안 넘어간다** → 노드 이름이 `"tools"`가 맞는지 확인
- **무한 루프처럼 반복된다** → 모델이 계속 툴을 호출하는 경우. 질문/프롬프트를 조정하거나 재귀 한도(`recursion_limit`)를 확인
- `print(response["messages"])`로 전체 메시지 흐름(Human → AI(tool_calls) → Tool → AI)을 눈으로 확인하면 이해가 빠르다

---

## 7. 컴파일 & 실행

```python
graph = graph_builder.compile()

response = graph.invoke({"messages": [("user", "오늘 삼성전자 주가 알려줘")]})
print(response["messages"][-1].content)     # 가장 마지막 메시지의 내용만 출력
```

- `response["messages"]`는 리스트 → `[-1]`로 **마지막 메시지**, `.content`로 본문
- 검색을 하지 않는 경우도 있다 → 모델이 "검색 불필요"로 판단했거나, 프롬프트/모델 성능 문제일 수 있음
- ⚠️ **오늘 숙제:** 삼성전자 주가를 알아내도록 만들어 보기 (질문 문구를 바꿔가며 시도)

---

## 8. 웹 검색 툴 ② — DuckDuckGo (완전 무료)

Tavily와 구조는 같고 **import 부분과 툴 정의만 다르다.** (API 키 불필요)

```python
# pip install -U duckduckgo-search ddgs   (설치 안 되는 경우)
from langchain_community.tools import DuckDuckGoSearchRun   # 또는 DuckDuckGoSearchResults

web_search = DuckDuckGoSearchRun()
tools = [web_search]
llm_with_tools = llm.bind_tools(tools)
```

- 결과 **출력 모양이 Tavily와 달라서** `result[-1]["content"]` 같은 접근은 그대로 안 될 수 있음 → 직접 print로 구조 확인
- 동시 접속이 많으면 일시적으로 에러가 날 수 있음 (잠시 후 재실행)
- 유료 검색 API를 써도 되고, 각자 원하는 것을 선택

### 8-1. 툴 2개 동시에 쓰기
```python
tools = [tavily_search, duckduckgo_search]
llm_with_tools = llm.bind_tools(tools)
```
- **어떤 툴을 쓸지도 모델이 판단**한다 (이번엔 Tavily, 이번엔 DuckDuckGo …)

---

## 9. 그래프 시각화 & 파일 저장

그래프가 의도대로 연결되었는지 **이미지로 확인**할 수 있다.

```python
png_data = graph.get_graph().draw_mermaid_png()   # PNG 바이너리 데이터 생성

with open("_save/image/lg_07_graph.png", "wb") as f:   # wb = write + binary
    f.write(png_data)
```

- 사전 준비: `_save` 폴더 아래에 **`image` 폴더**를 만들어 둘 것
- 저장된 PNG를 열어 `START → chatbot ⇄ tools → END` 형태인지 확인

---

## 10. 파이썬 기초 보충 — `with open`

### 10-1. 두 코드는 같은 동작

```python
# (A) 수동으로 열고 닫기
f = open("test.png", "wb")
f.write(png_data)
f.close()                 # 반드시 닫아야 함

# (B) with 문 — 블록이 끝나면 자동으로 close
with open("test.png", "wb") as f:
    f.write(png_data)
```

| 모드 | 의미 |
|---|---|
| `"w"` | write (쓰기) |
| `"b"` | binary (바이너리) |
| `"wb"` | 바이너리로 쓰기 → 이미지 같은 파일 저장에 사용 |

### 10-2. 알아둘 점
- `open(..., "wb")`만 실행해도 **빈 파일은 생성**된다. 데이터는 `write()` 해야 들어감.
- 파일을 안 닫으면 메모리 문제가 생길 수 있음 (요즘은 파이썬 가비지 컬렉션이 잘 처리하지만 닫는 습관이 좋음)
- **`with`는 "~와 함께"**: `with open(...) as f:` → 블록 안에서 `f`를 쓰고, **끝나면 자동 close**
- 파일 입출력은 대부분 `with open` 형태로 쓴다 → **외우거나 이해하거나 둘 중 하나는 해둘 것** (계속 쓰임)

---

## 11. 메모리(대화 기억) — 예고

- 이전 프롬프트(대화)들을 모아서 기억하는 기능 → 이전에 Gradio 챗봇에서 했던 **대화 히스토리** 개념과 연결
- 랭그래프에서의 메모리는 다음 시간에 이어서 진행 (수업 후반에 오프닝만 언급)

---

## ✅ 오늘의 핵심 요약

1. **툴 콜링 = 모델 + 외부 기능 연결.** 최신 정보가 필요할 때 검색 툴을 붙인다.
2. `llm.bind_tools(tools)` → 모델에 툴 연결 (툴은 리스트라서 여러 개 가능)
3. `ToolNode(tools)` → 툴을 그래프의 **노드**로 만들어 준다.
4. `add_conditional_edges("chatbot", tools_condition)` → 모델의 판단에 따라 **툴로 갈지 / 끝낼지** 분기
5. `add_edge("tools", "chatbot")` → 툴 결과를 가지고 **다시 챗봇**으로 (루프 구조)
6. 검색 툴은 **Tavily(월 1,000회 무료, 키 필요)** / **DuckDuckGo(완전 무료)**
7. 결과 접근: `response["messages"][-1].content`, 검색결과는 `리스트[-1]["content"]`
8. `get_graph().draw_mermaid_png()` + `with open(..., "wb")` → 그래프 이미지 저장
9. **숙제:** 삼성전자 주가 알아내는 툴 콜링 챗봇 완성하기

---

## 📝 복습 체크리스트

- [ ] 일반 엣지와 조건부 엣지의 차이를 설명할 수 있는가?
- [ ] `bind_tools`, `ToolNode`, `tools_condition` 각각의 역할을 말할 수 있는가?
- [ ] 툴 실행 후 왜 다시 `chatbot` 노드로 돌아가야 하는가?
- [ ] `with open(..., "wb")`가 왜 `close()`를 생략해도 되는지 설명할 수 있는가?
- [ ] 검색 결과(리스트 + 딕셔너리)에서 원하는 값만 꺼낼 수 있는가?
