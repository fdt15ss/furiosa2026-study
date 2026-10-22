# LLM 12일차 정리 — 랭그래프(LangGraph) 입문

> 오늘의 핵심: **State(상태) · Node(노드) · Edge(엣지)** 3가지 + 플러스 알파로 **Reducer(리듀서, `add_messages`)**
> 랭그래프 = 우리가 만들었던 **랭체인(retriever → model → 체크 → 결과)을 "그래프"로 바꿔서** 각 기능을 노드로 두고 연결하는 방식

---

## 1. 랭그래프 큰 그림

### 1-1. 왜 그래프인가?
- 랭체인으로 하던 흐름(retriever 만들기 → 모델 호출 → invoke)을 **노드 단위로 쪼개서** 그래프로 구성
- 예시 흐름 (수업 중 PPT 설명)
  1. 질문이 들어옴
  2. 모델(GPT 등) 호출
  3. 관련성(relevance) 체크 노드
  4. 결과 종합
  5. 마음에 안 들면 **다시 앞 노드로 되돌아가서(재시도)**, 마음에 들면 **END**
- 작은 예제에서는 랭체인과 차이가 없지만, **규모가 커질수록** 랭그래프의 장점이 체감됨
- 구조가 Keras의 **함수형 모델**과 비슷 → 위로도, 아래로도, 거꾸로(되돌아가기)도 연결 가능

### 1-2. 오늘 꼭 외울 4가지 키워드

| 키워드 | 한 줄 정의 |
|---|---|
| **State (상태)** | 노드 사이에 **전달되는 데이터**. 딕셔너리 형태 |
| **Node (노드)** | 각각 **하나의 기능**을 하는 **함수** (retriever, model, 체크 …) |
| **Edge (엣지)** | 노드와 노드를 **연결한 선** |
| **Reducer (리듀서)** | State의 값을 **덮어쓰지 않고 이어 붙이는(append)** 규칙 → `add_messages` |

- 노드/엣지 개념은 이미 해봤고, 오늘은 **State**까지 정리해서 한 세트를 완성
- 노드의 기능 구현은 **함수 vs 클래스** 중 **함수**로 한다 (훨씬 쉬움)
  - 함수 = 기능만 있음
  - 클래스 = 기능 + 잡다한 것(속성 등) 전부 → 함수보다 개념이 넓음

---

## 2. 파이썬 기초 문법 (오늘 새로 나온 것)

### 2-1. `TypedDict`

```python
from typing import Annotated, TypedDict
```

- `typing`은 파이썬이 **기본 제공**하는 모듈 (pip install 필요 없음)
- 대문자로 시작 → **클래스**
- `TypedDict` = **딕셔너리의 키/값 타입을 정해놓는 클래스**

### 2-2. 타입 힌트(Type Hint)

```python
from typing import Dict

sample: Dict[str, str] = {"name": "홍길동", "age": "39", "job": "개발자"}
```

- `Dict[str, str]` → "딕셔너리인데 키도 `str`, 값도 `str`로 쓰겠다"
- **힌트일 뿐 강제성이 없다** → 다른 타입을 넣어도 에러는 안 남
- 하지만 코드를 읽는 사람이 **"이 안에는 str:str 딕셔너리가 있겠구나"** 하고 바로 판단 가능 → 실무에서 많이 씀
- 만약 `int`로 쓴다고 힌트를 줬으면 `Dict[str, int]` 형태가 되어야 함
- 딕셔너리 값 꺼내기는 기존과 동일: `sample["name"]`

> **Q&A**: `'name'`처럼 작은따옴표로 감싼 게 저장되는 건가요?
> → 따옴표는 문자열 표시일 뿐, 저장되는 건 `name`이라는 키 자체. 부를 때 `sample["name"]`으로 부르면 됨. 큰따옴표/작은따옴표 차이는 없음.

### 2-3. 클래스 상속과 `TypedDict`

```python
class Person(TypedDict):
    name: str
    age: int
    job: str
```

- 괄호 안에 클래스를 넣으면 → **함수의 인자가 아니라 "상속"**
- **상속**: 부모 클래스의 모든 기능 + 그 안에 들어있는 것들을 가져다 쓸 수 있음
  - 수업 비유: 아빠에게 *시그니엘*(→ 마이바흐)을 상속받으면 거기 있는 욕탕·화장실·배달 주문까지 다 쓸 수 있는 것과 같음
- `Person(TypedDict)` = "딕셔너리 형태의 **틀(껍데기)** 만 정의"해둔 상태 → **실제 값은 아직 없음**
- 값을 넣어서 사용할 땐 기존 클래스 쓰듯이 인자를 넣음

```python
p = Person(name="홍길동", age=40, job="개발자")
```

- 첫 번째 방식(`Dict[str, str]`)은 **값까지 직접 넣은 것**, `TypedDict`는 **틀(스키마)을 먼저 만드는 것**
- 클래스도 하나의 **타입**이다 → `Person` 은 `Person` 타입, 스트링처럼 타입으로 쓸 수 있음

### 2-4. `Annotated` (주석 / 부가정보)

```python
from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages

class State(TypedDict):
    messages: Annotated[list, add_messages]
```

- `Annotated[타입, 부가정보]` → "이 필드는 `list` 타입이고, **`add_messages`라는 규칙**을 같이 적용한다"
- 지금은 깊게 몰라도 되고, **"리스트 + add_messages 한 쌍"** 으로 외워두기 (이해하려 하지 말고 일단 암기)
- 메시지 말고 다른 키(`a`, `b`, `c` …)도 State에 자유롭게 추가 가능 / 안 써도 됨 (취향 문제)

---

## 3. 리듀서(Reducer) = `add_messages`

### 3-1. 왜 필요한가?
- 대화는 계속 쌓여야 함

  ```
  사람: 삼성전자 얼마야?
  AI:   257,000원입니다
  사람: (다음 질문)
  AI:   (다음 답변)
  ```
- 그냥 리스트에 값을 넣으면 **뒤에서 답변할 때 앞의 값을 덮어써버림**
- 그래서 **덮어쓰지 않고 리스트에 자동으로 append** 해주는 놈이 필요 → **리듀서**
- 오늘의 가장 중요한 키워드 중 하나 (랭그래프에선 항상 쓰게 됨)

### 3-2. 임포트

```python
from langgraph.graph.message import add_messages   # 랭그래프에서 제공
from langchain_core.messages import HumanMessage, AIMessage
```

### 3-3. 실습 ① — 리듀서가 이어 붙이는 모습

```python
msg1 = HumanMessage(content="안녕", id="1")
msg2 = AIMessage(content="반가워", id="2")

result = add_messages(msg1, msg2)
print(result)
```

- 결과: **리스트 안에 두 메시지가 모두 들어감** (`[HumanMessage, AIMessage]`)
- `print(msg1)` → 메시지 전체가 나오고, `msg1.content` → `"안녕"`만 나옴
- 휴먼/AI 메시지가 섞여 들어가도 상관없음 (둘 다 리스트에 들어감)

### 3-4. 실습 ② — **ID가 같으면 덮어쓴다!**

```python
msg3 = HumanMessage(content="...", id="3")
msg4 = HumanMessage(content="...", id="3")   # 같은 id

result2 = add_messages(msg3, msg4)
```

- 결과: **하나만 남음** → `add_messages`는 **id가 동일하면 덮어씀(교체)**
- 평소에는 **id가 서로 다르게 자동 생성**되므로 보통은 신경 쓸 일 없음
- 정리: **"값을 덮어쓰지 않고 일단 연결(append)한다"** (단, id가 같으면 교체)

---

## 4. 랭그래프 만드는 6단계

```
① State 정의  →  ② Node 정의  →  ③ 그래프 구성(add_node)  →  ④ Edge 연결(add_edge)  →  ⑤ compile  →  ⑥ invoke
```

### ① State 정의

```python
from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages

class State(TypedDict):
    messages: Annotated[list, add_messages]
```

- 변수 이름은 `messages`를 가장 많이 씀
- 아직은 **껍데기만 만든 상태** (값 없음)

### ② Node 정의 (함수로)

```python
def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}
```

- **항상 State를 받아서**, **State 형태(딕셔너리)로 결과를 반환**
- 노드는 **State에 종속적**이다 → 노드 함수를 만들 땐 `state: State` 로 받고, 결과 메시지를 `{"messages": [...]}` 로 돌려준다
- `llm.invoke(state["messages"])` 결과는 AI 메시지 → 리스트에 담아 반환 → 리듀서가 기존 리스트에 자동 append
- **위에서 받은 State → 아래로 전달할 State** (바통 터치 구조)

### ③ 그래프 구성

```python
from langgraph.graph import StateGraph, START, END

graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
```

- `StateGraph(State)` : 랭그래프를 준비. Keras의 `Sequential()` 처럼 만들고 `add` 하는 느낌
- `add_node("이름", 함수)` : "이런 노드가 있다"고 **등록만** 하는 단계 (**아직 연결 아님**)
  - 이름 문자열의 **따옴표 주의**
- `START`, `END` : 시작과 끝을 나타내는 **특수 지점** (노드가 아니라 "여기서부터 시작 / 여기서 끝"이라는 표시)

### ④ Edge 연결

```python
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)
```

- 시작 → chatbot → 끝
- 노드가 5개면 `add_edge` 도 그만큼 더 연결해주면 됨
- **노드는 위에서 정의, 연결은 엣지로**

### ⑤ 컴파일

```python
graph = graph_builder.compile()
```

- 그래프는 **훈련이 아니라 추론 개념** → 만든 그래프를 컴파일해서 실행 준비
- Keras의 `compile`과 비슷하지만 **훈련(fit)이 아님**

### ⑥ 실행 (`invoke`)

```python
response = graph.invoke({"messages": [("user", "삼성전자 주가를 예측해줘")]})
```

- 입력도 **딕셔너리 형태** (`messages` 키)
- `("user", "...")` 튜플 → **휴먼 메시지**로 들어감 (역할 이름은 `user` / `assistant` 등 **관용적으로 쓰는 이름에 맞춰줘야** 함)
- ⚠️ 수업 중 실수: 메시지만 넣고 **답변이 비어 있던 경우** → 입력 형식/노드 반환을 다시 확인하고 재실행하니 정상 출력

### 결과 확인

```python
response["messages"][0]            # 휴먼 메시지 (사용자 질문)
response["messages"][1]            # AI 메시지
response["messages"][1].content    # ← 답변 텍스트만 출력
response["messages"][0].type       # "human" / "ai" 타입 확인
```

- `response["messages"]` 는 **리스트** → 0번째는 사람, 1번째는 AI
- 답변만 보고 싶으면 **`.content`**

---

## 5. 확장 실습 — 노드 3개 만들기 (prepare → chatbot → finish)

> 성능 향상이 목적이 아니라 **노드와 엣지 구성 연습**이 목적. 필요 없는 노드가 추가되면 오히려 성능이 떨어질 수도 있음.

```python
def prepare(state: State):
    return {"messages": [AIMessage(content="질문을 접수했습니다")]}

def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}

def finish(state: State):
    return {"messages": [AIMessage(content="답변을 처리했습니다")]}

graph_builder = StateGraph(State)
graph_builder.add_node("prepare", prepare)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("finish", finish)

graph_builder.add_edge(START, "prepare")
graph_builder.add_edge("prepare", "chatbot")
graph_builder.add_edge("chatbot", "finish")
graph_builder.add_edge("finish", END)

graph = graph_builder.compile()
response = graph.invoke({"messages": [("user", "삼성전자 주가를 예측해줘")]})
```

### 결과 출력 (`enumerate` 활용)

```python
for i, msg in enumerate(response["messages"]):
    print(f"{i} {msg.type} : {msg.content}")
```

- 최종 `messages` 길이 = **4개**
  | idx | type | 내용 |
  |---|---|---|
  | 0 | human | 삼성전자 주가를 예측해줘 |
  | 1 | ai | 질문을 접수했습니다 (prepare 노드) |
  | 2 | ai | chatbot(LLM)의 답변 |
  | 3 | ai | 답변을 처리했습니다 (finish 노드) |
- **리듀서 덕분에** 각 노드의 결과가 덮어쓰이지 않고 전부 이어 붙음
- 리스트 → **이터레이터 → for문**, 순서와 값이 필요하면 → **`enumerate`** (파이썬 중급 이상 필수 문법)
- `f"{i} {msg.type}"` : f-string으로 출력문 안에 변수 값 삽입 (`{}` 위치에 순서대로 들어감)

---

## 6. 한눈에 보는 요약

```
State  = 노드 사이에 전달되는 상태 (TypedDict, 딕셔너리 형태)
Node   = 기능 하나를 하는 함수 (State를 받아서 State 형태로 반환)
Edge   = 노드들을 연결하는 선 (START → ... → END)
Reducer(add_messages) = 값을 덮어쓰지 않고 리스트에 이어 붙임 (id 같으면 교체)

[순서]  State 정의 → Node 정의 → add_node → add_edge → compile → invoke
```

### 체크리스트
- [ ] `TypedDict` + `Annotated[list, add_messages]` 로 State를 만들 수 있다
- [ ] 노드 함수는 `state: State`를 받고 `{"messages": [...]}` 를 반환한다
- [ ] `add_node`는 "등록", `add_edge`가 "연결"이라는 걸 구분한다
- [ ] `START` / `END` 가 특수 지점이라는 걸 안다
- [ ] `compile()` 후 `invoke()` 로 실행하고, `response["messages"][i].content` 로 결과를 본다
- [ ] 같은 id의 메시지는 덮어써진다는 걸 안다

---

## 7. 파이썬 복습 포인트

| 문법 | 오늘 나온 쓰임 |
|---|---|
| `typing` 모듈 | `TypedDict`, `Annotated`, `Dict` (기본 제공 모듈) |
| 타입 힌트 | 강제성 없음, 가독성·의도 전달용 |
| 클래스 상속 | `class Person(TypedDict)` → 부모 기능 + 알파를 가져다 씀 |
| 클래스도 타입 | `Person` 은 `Person` 타입 |
| 딕셔너리 접근 | `d["key"]` |
| `enumerate` | 리스트 순회 시 인덱스 + 값 |
| f-string | `f"{i} {msg.type}"` |

---

## 8. 기타 (수업 외 메모)

- **미니(팀) 프로젝트**: 진도는 짧게 나가고 **프로젝트 진행에 집중**, 이번엔 **무조건 아웃풋**을 내야 함
  - 쉬는 시간 외 자리 이탈 시 **2층 내에서만 이동**, 집합 호출 시 **5분 내 복귀**
  - 문제가 생기면 바로 강사님께 공유
- **프로젝트 팁 (팀 논의 내용)**
  - 대용량 문서를 LLM으로 메타데이터 추출 시 **처리 시간이 매우 큼** (문장 1개 약 6.5초 → 전체 시 수십~100시간 단위 예상)
  - 로컬 소형 모델(E2B) 사용, 스레드를 늘려도 **차이가 크지 않음**
  - **전처리 단계에서 메타데이터를 잘 뽑아두면** 검색·활용이 쉬워짐 → 중간 저장(JSON 파일)하고 **끊어서 이어 돌리는** 방식 고려
- 강사님 한마디: GPT에만 의존하지 말고 **스스로 찾아보면서 해야 실력이 늚**

> ⚠️ 원본 녹취에 음성 인식 오류(예: "엘지" = LangGraph(LG), "리듀서" = Reducer, "엣지/노드" 등)가 많아, 문맥에 맞춰 용어를 바로잡아 정리했습니다.
