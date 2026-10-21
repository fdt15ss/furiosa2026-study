# LLM 13일차 정리

> 주제: **랭그래프(LangGraph) 심화 — 그래프 시각화 · 메모리(체크포인터) · 병렬 실행(Fan-out / Fan-in)**
> 참고: 녹취(STT) 품질이 낮아 인식이 깨진 부분은 문맥으로 복원했습니다. 코드 예시는 강의 흐름에 맞춰 정리한 것이므로, 실제 실습 파일과 변수명은 다를 수 있습니다.

---

## 0. 오늘 한눈에 보기

| 순서 | 내용 | 핵심 키워드 |
|---|---|---|
| 1 | 그래프 시각화 | `get_graph().draw_mermaid_png()`, `IPython.display`, `plt.imshow`, 바이너리 저장(`wb`) |
| 2 | 메모리 기능 추가 | `MemorySaver`(InMemorySaver), `compile(checkpointer=)`, `thread_id` |
| 3 | 병렬 실행 그래프 | `START → (검색, 분석) → 병합 → END`, `Annotated[list, operator.add]` |
| 4 | 메모리 + 병렬 적용 | 검색 노드 / 모델 노드 / 병합 노드 3개 |
| 5 | 특별 탭 | 팀 프로젝트 발표·면접 대비, GPU/NPU 메모리 이야기 |

> 선생님 말씀: 커리큘럼이 쉽게 가다가 **어제부터 난이도가 확 올라갔다.** 오늘 내용은 "대부분 오타 문제"이니 걱정 말고, **어떻게 흘러가는지 구조를 이해**하는 것이 핵심.

---

## 1. 그래프 시각화

### 왜 시각화하나?
- 코드 전체의 흐름(`start → … → end`)을 **한눈에** 볼 수 있다.
- PPT에는 코드만이 아니라 **차트/그래프/아키텍처 그림**이 항상 같이 들어가야 "아름다운 PT"가 된다.
- 회사 대표님께 코드를 설명하면 못 알아듣는다 → **"이런 아키텍처로 되어 있어서 비용을 줄이거나 이렇게 바꿀 수 있습니다"** 처럼 **그림으로** 보여줘야 한다.

### 방법 1: 주피터 노트북에서 바로 출력

```python
from IPython.display import Image, display

display(Image(graph.get_graph().draw_mermaid_png()))
```

- 주피터 같은 노트북에서는 **출력 이미지가 바로 나온다.**

### 방법 2: 일반 `.py` 파일에서 — 메모리(BytesIO) 위에서 이미지 열기

강의에서는 `.py` 파일로 실행하므로 `matplotlib`으로 띄웠다.

```python
import io
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

png = graph.get_graph().draw_mermaid_png()   # 바이너리(bytes) 데이터

img = mpimg.imread(io.BytesIO(png))          # 실제 파일을 만들지 않고 메모리 안에 '가상 파일' 생성
plt.imshow(img)
plt.axis("off")                              # 축(X, Y) 숨기기 → 깔끔하게
plt.show()
```

- `io.BytesIO` : **실제 파일을 만들지 않고 메모리 안에 파일처럼 쓸 수 있는 객체**를 만들어 준다. 이미지를 거기에 넣어서 보는 것.
  - 선생님: "여기까지 깊게 알 필요 없고, **그냥 갖다 붙여 쓰면 된다.**"
- `plt.axis("off")` 로 옆 축 눈금을 없애면 깔끔해진다.
- 파일명은 `langgraph_06` 처럼 번호를 붙여 관리 (`LG08`, `LG08_memory` 등).
- 그림이 너무 작으면 `plt.figure(figsize=(...))` 로 크기 조절.

### 방법 3: 파일로 저장 (바이너리 쓰기 `wb`)

```python
with open("graph.png", "wb") as f:     # w = write, b = binary
    f.write(png)
```

- 주석 처리해 두었던 코드 — **`wb` = 바이너리로 쓴다.**
- 질문: "흑백/컬러 구분인가요?" → **아니다.** 흑백·컬러가 아니라 **이진(바이너리) 데이터 형태**라는 뜻 (텍스트 `w` vs 바이너리 `wb`).
- 저장하고 싶으면 **방법 1**, 화면으로 보고 싶으면 **방법 2**, 둘 다 하고 싶으면 둘 다 써도 된다.
- 즉, 이미지 확인은 **메모리에서 보기 / 파일로 저장하기** 두 갈래.

---

## 2. 랭그래프 메모리 기능 (Checkpointer)

### 개념
- 메모리 = **이전 대화를 기억하게 해주는 기능**. 쉽게 말해 프롬프트에 이전 대화가 이어 붙는 것인데, 이를 프롬프트에 직접 하지 않고 **랭그래프의 기능으로** 처리한다.
- 확인 방법 (질문을 **두 번** 한다):
  1. 첫 번째 질문: **"내 이름은 ○○야"**
  2. 두 번째 질문: **"내가 누구야?" / "내 이름이 뭐야?"**
  - 메모리가 있으면 알아듣고, 없으면 못 알아듣는다.
  - 두 번째 질문을 **똑같은 `thread_id`로 해야** 같은 사람과의 대화로 인정해 준다. (강의에서는 `user_0` 같은 값 사용)

### 메모리 넣는 곳은 딱 **3군데**

> 선생님 팁: 내일이면 "3군데라고 했는데 어디였지?" 하게 되니까, **주석에 `# 메모리 추가 1/2/3`** 이라고 달아 두고 `Ctrl+F`로 "메모리"를 검색하면 바로 찾을 수 있다.

```python
# ✅ 메모리 추가 1 — import
from langgraph.checkpoint.memory import MemorySaver   # (InMemorySaver 이름으로도 사용)

# ✅ 메모리 추가 2 — 저장소 객체 만들고, compile 할 때 checkpointer로 지정
memory = MemorySaver()
graph = builder.compile(checkpointer=memory)

# ✅ 메모리 추가 3 — invoke 할 때 config(thread_id) 전달
config = {"configurable": {"thread_id": "user_0"}}

graph.invoke({"messages": [("user", "내 이름은 ○○야")]}, config=config)
graph.invoke({"messages": [("user", "내가 누구야?")]},   config=config)
```

| # | 위치 | 하는 일 |
|---|---|---|
| 1 | import 부분 | `MemorySaver` 가져오기 |
| 2 | `compile()` | `checkpointer=memory` 로 **메모리 저장소 지정** |
| 3 | `invoke()` | `config`의 **`thread_id`** 로 "누구의 대화인지" 구분 |

- `config` 는 **딕셔너리**이고 **`invoke` 바깥에서 정의**해서 넘긴다.
- **`thread_id`가 다르면 다른 사람(다른 대화방)** → 기억하지 못한다. 이름이 중요한 게 아니라 **같은 thread_id인지**가 중요.
- 앞으로 랭그래프에서 메모리 기능을 추가할 때는 **이 세 가지만** 넣으면 된다.
- 에러가 나면: 경로·shape 문제가 아니라 **100% 오타**일 가능성이 높다 → 스스로 찾아 해결해 보기. (`print`/`dir`로 확인하며 추적)

### 메모리의 휘발성 ⚠️
- 이 메모리는 **RAM(램)에 올라가 있다.** → **프로그램을 껐다 켜면 다 날아간다(휘발).**
- 모든 코드는 실행 시 램에 올려서 돌아간다.
- **영구 저장**하고 싶으면 **DB(백엔드 DB 등)에 저장**하는 별도 체크포인터를 사용해야 한다. (이번 수업에서는 인메모리까지만, DB 연동은 다음에)

### 예전 방식과 비교 (Gradio 챗봇 복습)
- 지난번 Gradio 챗봇에서는 **히스토리를 직접 관리**했다 (`MessagesPlaceholder` 등).
- Gradio는 이런 용도로는 **약하고**, 랭그래프에서는 **메모리 기능이 아주 강력**하다.

---

## 3. 병렬 실행 그래프 (Fan-out → Fan-in)

### 지금까지 vs 오늘
- **이전(순차 + 툴 루프)**: `START → LLM → (툴이 필요하면) tools → 다시 LLM … → END`
  - 무조건 LLM을 먼저 호출한 뒤 **"툴로 갈지 말지 판단"**
- **오늘(병렬)**: `START`에서 **동시에** 두 노드를 실행하고 **한 곳에서 합친다(merge).**
  - "병렬 실행이라고 하니 있어 보이지만, **한마디로 같이 실행하겠다는 것**"

```text
            ┌──▶ search_agent ───┐
 START ─────┤                    ├──▶ merge_agent ──▶ END
            └──▶ analyst_agent ──┘
```

### Step 1. State 정의

```python
import operator
from typing import Annotated, TypedDict, List

class State(TypedDict):
    messages: Annotated[list, add_messages]      # 대화 메시지
    search_result: Annotated[list, operator.add] # 검색 결과 (리스트로 누적)
    analyst_result: Annotated[list, operator.add]# 분석 결과 (리스트로 누적)
    final_result: str                            # 최종 병합 결과
```

- `Annotated[list, operator.add]` 는 **"이 키는 `list` 타입이고, 새 값이 들어오면 덮어쓰지 말고 `+` 로 이어 붙여라"** 라는 뜻이다. (자세한 설명은 아래 **「3-1. 심화」** 참고)
- 강의에서는 결과를 **리스트 형태**로 받아 두 결과(검색/분석)를 모으는 구조를 사용했다. 참고로 위 예시처럼 키가 서로 다르고 노드가 한 번씩만 쓰면 리듀서가 꼭 필요하진 않지만, **여러 노드가 같은 키에 쓰는 구조로 확장할 때** 반드시 필요해진다.
- 선생님 팁: **State가 헷갈리면 State를 빼고 생각하라.** 노드는 결국 *"State를 받아서 State(의 일부)를 반환하는 함수"* 일 뿐이다.

### Step 2. 노드 = 함수 (모든 노드는 함수로 만든다)

```python
def search_agent(state: State):
    # (연습용 껍데기) 아무것도 실행하지 않고 흐름만 확인
    return {"search_result": ["검색 결과"]}

def analyst_agent(state: State):
    return {"analyst_result": ["분석 결과"]}

def merge_agent(state: State):
    merged = state["search_result"] + state["analyst_result"]   # 단순 붙이기
    return {"final_result": merged}
```

- 받는 인자 = `State`, 반환 = `State`에 반영될 딕셔너리.
- 병합(merge)도 **하나의 노드**다.

### 병합(merge) 방식 2가지
1. **단순 붙이기(concat)** — 두 결과 리스트를 그냥 합친다. (오늘 실습)
2. **프롬프트로 재조합** — 두 결과를 프롬프트에 넣어 **모델에 다시 넣어 재추론**시킨다.

### Step 3. 그래프 구성 (`StateGraph`)

```python
from langgraph.graph import StateGraph, START, END

builder = StateGraph(State)

# 노드 추가
builder.add_node("search_agent",  search_agent)
builder.add_node("analyst_agent", analyst_agent)
builder.add_node("merge_agent",   merge_agent)

# 엣지 연결
builder.add_edge(START, "search_agent")        # START → 검색
builder.add_edge(START, "analyst_agent")       # START → 분석  (동시에 시작 = 병렬)
builder.add_edge("search_agent",  "merge_agent")
builder.add_edge("analyst_agent", "merge_agent")
builder.add_edge("merge_agent", END)

graph = builder.compile()
```

- `StateGraph` 를 쓰는 것은 이전 `Sequential` 모델과 **엄밀히는 다르지만 느낌은 비슷**하다.
- 엣지는 **`START` 에서 두 갈래**로 나가고, **`merge` 에서 다시 하나로 모여 `END`** 로 간다.
- 마지막은 `merge → END` (merge가 END로 연결되어야 한다).

### Step 4. 실행으로 확인

```python
result = graph.invoke({"messages": [("user", "MPU의 미래 전망은?")]})
```

- 이 버전은 **LLM도 없고 검색엔진도 없는 "껍데기"** 이다. → **프로세스가 이렇게 흘러간다**는 것만 확인하는 용도.
- 각 노드가 실행될 때 `print` 를 찍어 보면 **"분석 에이전트도 실행, 검색 에이전트도 실행"** 로그가 둘 다 나오는 것을 확인할 수 있다.
- 입력은 **`messages` 에 넣어서** `invoke` 한다.

---

## 3-1. 심화: `Annotated[list, operator.add]` 완전 정복

```python
search_result: Annotated[list, operator.add]
```

이 한 줄은 **3개 조각**으로 나눠서 보면 쉽다.

| 조각 | 정체 | 역할 |
|---|---|---|
| `list` | 타입 | 이 키에 들어갈 값은 **리스트**다 |
| `operator.add` | 함수 | 값이 합쳐질 때 쓸 **합치는 규칙(리듀서, reducer)** |
| `Annotated[ , ]` | 포장지 | 타입(`list`)에 **추가 정보(`operator.add`)를 붙여 주는 도구** |

### 0) 먼저, `Annotated` 라는 단어의 뜻

| 단어 | 뜻 |
|---|---|
| **annotate** (어노테이트) | **주석을 달다, 메모를 붙이다** (라틴어 *notare* = "표시하다"에서 유래) |
| **annotation** (어노테이션) | **주석, 메모, 꼬리표** |
| **annotated** (어노테이티드) | **주석이 달린, 메모가 붙은** (annotate의 과거분사 = 형용사) |

→ 그래서 **`Annotated[list, operator.add]` = "`operator.add` 라는 메모가 붙은 `list`"** 라고 그대로 읽으면 된다.

**비슷한 단어들 (이미 들어봤을 것)**
- **annotated bibliography** : 설명(주석)이 달린 참고문헌 목록
- **데이터 어노테이션(Annotation)** : AI에서 이미지에 박스·라벨을 붙이는 **라벨링 작업** (= 데이터에 정답 메모를 붙임)
- **Java의 `@Override`, `@Test`** : 코드에 붙이는 표시 → 이것도 어노테이션
- **파이썬 변수 어노테이션** : `x: int = 3` 에서 **`: int` 부분**을 파이썬에서는 어노테이션(= 타입 힌트)이라고 부른다.

```python
x: int = 3
print(__annotations__)     # {'x': <class 'int'>}   ← 파이썬이 저장해 두는 '어노테이션'
```

**비유**: 교재(= `list`, 타입)에 **포스트잇(= `operator.add`, 메모)** 을 붙여 둔 것.
- 교재 내용(타입)은 그대로이고, 포스트잇은 **그걸 읽는 사람(= 랭그래프)** 만 보고 "아, 이렇게 처리하라는 거구나" 하고 따른다.
- 파이썬 자신은 포스트잇을 **무시**한다.

즉 **`Annotated` = "타입 힌트(어노테이션)에 추가 메모(메타데이터)를 더 붙일 수 있게 해주는 파이썬 표준 도구"** 라는 이름 그대로의 뜻이다. (`typing.Annotated`, 파이썬 3.9부터 표준, PEP 593)

### 1) `operator.add` 는 무엇인가?

파이썬 기본 모듈 `operator` 에 들어 있는 **`+` 연산자를 함수로 만든 것**이다.

```python
import operator

operator.add(1, 2)          # 3          (== 1 + 2)
operator.add("a", "b")      # "ab"       (== "a" + "b")
operator.add([1], [2, 3])   # [1, 2, 3]  (== [1] + [2, 3])
```

- 리스트에 `+` 를 쓰면 **이어 붙이기(concat)** 가 된다. 그래서 `operator.add` 는 **"기존 리스트 뒤에 새 리스트를 붙여라"** 라는 규칙이 된다.
- 랭그래프에서는 이런 규칙을 **리듀서(reducer)** 라고 부른다. 형태는 항상 **`(기존 값, 새 값) → 합친 값`**.

### 2) 리듀서가 왜 필요한가? — 랭그래프의 기본 동작은 "덮어쓰기"

노드가 `return {"키": 값}` 을 하면, 랭그래프는 State의 해당 키를 **그 값으로 갈아 끼운다(덮어쓰기).**

```python
class State(TypedDict):
    results: list            # ❌ 리듀서 없음 → 기본 동작 = 덮어쓰기
```

```text
노드 A: return {"results": ["A의 결과"]}   → results = ["A의 결과"]
노드 B: return {"results": ["B의 결과"]}   → results = ["B의 결과"]   ← A의 결과가 사라짐!
```

그런데 **병렬 실행(Fan-out)** 에서는 노드 A와 B가 **같은 단계(step)에서 동시에** 같은 키에 값을 쓴다.
- 덮어쓰기 규칙으로는 **누구 값을 남길지 정할 수 없다** → 랭그래프가 에러를 낸다.
  (`InvalidUpdateError: Can receive only one value per step. Use an Annotated key to handle multiple values.`)
- 이때 **"둘 다 살려서 합쳐라"** 라고 알려주는 것이 리듀서다.

```python
class State(TypedDict):
    results: Annotated[list, operator.add]   # ✅ 리듀서 있음 → 덮어쓰기 대신 이어 붙이기
```

```text
노드 A: return {"results": ["A의 결과"]}   → results = ["A의 결과"]
노드 B: return {"results": ["B의 결과"]}   → results = ["A의 결과", "B의 결과"]   ← 둘 다 보존
```

즉 내부적으로는 이렇게 동작한다고 생각하면 된다.

```python
새_state["results"] = operator.add(기존_state["results"], 노드가_반환한_값)
```

(처음에는 키에 값이 없으므로 **빈 리스트 `[]`에서 시작**해서 첫 값이 붙는다.)

### 3) 그럼 왜 `Annotated` 가 들어가는가? ⭐

핵심 이유: **타입 힌트 자리에 "합치는 방법"이라는 추가 정보를 넣을 곳이 필요해서**다.

- `results: list` 라고만 쓰면 **"리스트다"** 라는 정보밖에 없다. **"어떻게 합칠지"** 는 쓸 자리가 없다.
- 파이썬 표준 `typing.Annotated` 는 **타입에 메타데이터(꼬리표)를 붙여 주는 도구**다.

```python
Annotated[타입, 메타데이터1, 메타데이터2, ...]
Annotated[list, operator.add]
#         ↑타입    ↑꼬리표(메타데이터)
```

동작 원리는 다음과 같다.

1. **파이썬 자체**는 `Annotated` 의 메타데이터를 **무시**한다. 타입 검사 관점에서는 그냥 `list` 와 똑같다.
2. 하지만 **메타데이터는 런타임에 읽을 수 있다.**
3. **랭그래프가 `State(TypedDict)` 의 각 키 타입을 읽다가** `Annotated` 의 두 번째 값이 **함수(callable)** 이면 → **"아, 이 키는 이 함수로 합치라는 뜻이구나"** 하고 **리듀서로 등록**한다.

직접 확인해 볼 수 있다.

```python
from typing import Annotated, TypedDict, get_type_hints
import operator

class State(TypedDict):
    results: Annotated[list, operator.add]

hints = get_type_hints(State, include_extras=True)
print(hints["results"])                  # typing.Annotated[list, <built-in function add>]
print(hints["results"].__origin__)       # <class 'list'>              ← 진짜 타입
print(hints["results"].__metadata__)     # (<built-in function add>,)  ← 꼬리표 (랭그래프가 읽는 부분)
```

> 💡 **한 줄 요약**: `list` = **무엇을** 저장하나, `operator.add` = **어떻게** 합치나, `Annotated` = **둘을 한 변수에 같이 적게 해주는 포장지.**
> 기존 타입 시스템을 건드리지 않고 랭그래프에게 **추가 지시**를 전달하는 파이썬 표준 방식이다.

### 4) 이미 써 본 같은 패턴: `add_messages`

11일차에 배운 `Annotated[list, add_messages]` 도 **완전히 같은 구조**다.

```python
messages: Annotated[list, add_messages]    # 메시지는 덮어쓰지 않고 '추가'
results:  Annotated[list, operator.add]    # 결과도 덮어쓰지 않고 '이어 붙이기'
```

| | `operator.add` | `add_messages` |
|---|---|---|
| 합치는 방식 | 단순 `+` (이어 붙이기) | 메시지 전용 리듀서 |
| 메시지 id | 신경 쓰지 않음 | **같은 id면 덮어써서 갱신**, 새 id면 추가 |
| 입력 변환 | 리스트만 가능 | `("user", "안녕")` 같은 튜플/딕셔너리를 **메시지 객체로 자동 변환** |
| 용도 | 일반 리스트 누적 (검색/분석 결과 등) | 대화 메시지 누적 |

→ 둘 다 **"(기존 값, 새 값)을 받아 합친 값을 돌려주는 함수"** 라는 점은 같다. 그래서 **직접 만든 함수도 리듀서로 쓸 수 있다.**

```python
def keep_unique(old: list, new: list) -> list:      # 중복 제거하며 합치기
    return old + [x for x in new if x not in old]

class State(TypedDict):
    results: Annotated[list, keep_unique]
```

### 5) 주의사항 / 자주 하는 실수

| 실수 | 설명 | 해결 |
|---|---|---|
| 노드가 **문자열**을 반환 | `[] + "abc"` → `TypeError` (리스트 + 문자열 불가) | **리스트로 감싸서** 반환: `return {"results": ["abc"]}` |
| `Annotated` 를 **빼먹음** | 병렬에서 같은 키에 쓰면 `InvalidUpdateError` | 병렬로 쓰는 키에는 **리듀서를 반드시** 지정 |
| **import 누락** | `NameError: Annotated / operator` | `from typing import Annotated` / `import operator` |
| 순서에 의존 | 병렬 노드의 결과 순서가 항상 같다고 가정 | **순서에 의존하지 않도록** 설계 (필요하면 결과에 라벨을 붙이기) |
| 리듀서 있는 키를 **초기화**하려는 시도 | 덮어쓰기가 아니라 계속 **누적**됨 | 초기화가 필요하면 별도 키/커스텀 리듀서 사용 |
| `operator.add` 를 숫자 키에 사용 | 리스트가 아닌 `int` 에 쓰면 **합산**이 됨 | 의도한 것인지 확인 (카운터로 쓰면 유용) |

### 6) 오늘 병렬 그래프에 대입해 보기

```text
          ┌─▶ search_agent  ── return {"results": ["검색 결과"]} ──┐
 START ───┤                                                        ├─▶ merge_agent ─▶ END
          └─▶ analyst_agent ── return {"results": ["분석 결과"]} ──┘
```

- 두 노드가 **같은 `results` 키**에 동시에 쓰는 구조라면 → `Annotated[list, operator.add]` 덕분에 `merge_agent` 가 받는 `state["results"]` 는 `["검색 결과", "분석 결과"]` **(둘 다 보존)**.
- `merge_agent` 는 이 리스트를 **그대로 붙여 출력(단순 붙이기)** 하거나, **프롬프트에 넣어 LLM으로 재요약**할 수 있다. (3장의 병합 방식 2가지)
- 강의에서 선생님이 "리스트형이라 나간다"고 짚은 부분이 바로 **여러 결과를 리스트로 모으는 구조**다.

> ✅ **암기 문장**: *"병렬로 같은 키에 쓰면 덮어쓰기가 안 되니까, `Annotated` 로 `operator.add` 리듀서를 붙여서 **이어 붙이기**로 바꾼다."*

---

## 4. 실제 모델 + 검색에 병렬 + 메모리 적용 (`LG08` 시리즈)

### 이전 구조(툴 호출)와의 차이

| | 이전 (툴 호출 루프) | 오늘 (병렬) |
|---|---|---|
| 시작 | **무조건 LLM 먼저** 호출 | START에서 **검색과 모델을 동시에** |
| 판단 | LLM이 툴을 쓸지 **판단** | 판단 없음, **둘 다 무조건 실행** |
| 끝 | tools 결과로 다시 LLM | **merge**에서 합쳐 END |

### 작업 순서
1. 이전 파일을 복사해서 시작 (`LG08_2` → 병렬 파일 / 메모리 파일은 `LG08_memory`).
2. **임포트 · 모델 · 검색 도구**는 이전 코드 그대로 재사용.
3. **State 정의**는 그대로 (바뀔 게 없음).
4. **노드 3개** 정의:
   - `search_agent` : 웹 검색 실행
   - `analyst_agent` (모델 노드) : **LLM**에 질문 (`llm.invoke(...)`)
   - `merge_agent` : 두 결과 병합
5. **그래프 구성**은 `add_node` → `add_edge` → `compile` 순서.
   - 그래프 구성부는 이전 파일(`08_1`)에서 **그대로 복사**해서 사용 (노드·엣지 병합 구조가 동일하므로).
6. `invoke` 시 질문 예: **"MPU(NPU)의 미래 전망은?"** → `messages` 에 넣어 호출.
7. 메모리가 필요 없으면 **"메모리는 없는 척"** 하고 그냥 `messages` 만 넣어서 먼저 돌려보기.

### 모델
- 강력한 모델을 사용 (`gpt-5-nano` 등). 선생님 말씀: **"여러분이 쓰고 싶은 모델 쓰면 된다"** (Claude 등도 무관).

---

## 5. 자주 나는 오류 / 체크리스트

- ✅ **대부분 오타**다. 경로·shape가 틀릴 일은 거의 없음 → 철자, 따옴표, 괄호, 들여쓰기부터 확인.
- ✅ **`State` 정의를 빠뜨리거나** 키 이름이 노드 반환값과 다르면 오류.
- ✅ 이전 파일을 복사해 쓸 때 **남아 있는 코드(지워야 할 것)** 를 정리했는지 확인.
- ✅ **`thread_id`** 를 두 번 다 **같은 값**으로 넣었는지.
- ✅ 병렬로 같은 키에 쓰는 경우 **리듀서(`operator.add`)** 가 있는지.
- ✅ 에러가 나면 **`print` 로 찍어 보고** 직접 추적해서 해결하는 습관.
- 선생님: "항상 틀려요. 저도 항상 틀려요. 걱정하지 마세요." / 내일 처음부터 **다시 정리해서** 설명 예정.

---

## 6. 오늘 꼭 기억할 것 (요약)

1. **시각화**: `get_graph().draw_mermaid_png()` → 노트북은 `display`, `.py`는 `BytesIO + plt.imshow`, 저장은 `open(..., "wb")`
2. **메모리 3군데**: ① import ② `compile(checkpointer=memory)` ③ `invoke(config={"configurable": {"thread_id": ...}})`
3. **인메모리는 휘발성** → 영구 저장은 DB 체크포인터
4. **병렬 실행**: `START`에서 두 갈래 → `merge`에서 합쳐 `END`
5. **병렬 결과 병합**은 `Annotated[list, operator.add]` 리듀서로 리스트 누적 — `list`(타입) + `operator.add`(합치는 규칙) + `Annotated`(둘을 묶는 포장지), 기본 동작은 덮어쓰기 (→ 3-1장)
6. **노드 = 함수**: `State`를 받아 `State`(의 일부)를 반환
7. **State가 헷갈리면 State를 빼고** 노드의 입력/출력만 생각하기

---

## ⭐ 특별 탭 — 강의 외 이야기 (프로젝트 · 취업 · 장비)

### 팀 프로젝트 발표 / 면접 대비
- 최종 발표: **10분 컷** (10분 넘으면 바로 자름) + 이후 **질의응답**.
- 면접에서 회사는 **취미나 이성 관계가 아니라 "기술"** 을 묻는다. 기술을 가장 간단하게 확인하는 방법 = **프로젝트를 발표시키는 것**.
- **본인이 구현하지 않은 부분도** PPT에 있는 건 **다 질문**한다 → 자기 역할만 파면 안 되고, **전체를 끝까지 완벽히 이해**해서 **그대로 발표할 수 있어야** 한다. (발표 2~3일 전부터는 다른 역할 팀원 파트까지 이해)
- 아키텍처에 **있어야 할 기능이 빠져 있으면** "왜 빠졌나요?"를 물어본다 → 답할 수 있어야 한다.
- 면접 단골 질문: "이 프로젝트 팀장이 누구죠? 본인 역할은?"
  - "데이터 수집만 했습니다" → 회사는 **일괄적으로 전부 처리할 수 있는 사람**을 원한다.
  - 시간 관계상 업무를 나누는 건 어쩔 수 없지만, **전체 이해는 필수.**
- 선배들이 취업이 잘 된 이유: **프로젝트 전체를 이해하고 발표를 잘했기 때문.**
- 시각화 + 아키텍처 그림이 PPT에 반드시 들어가야 한다.

### 오늘부터의 수업 일정
- 중간을 돌았으므로 **하루 8시간 수업**으로 진행.
- 일부 외부 행사/발표 참석 희망자는 **매니저님께 사전에 말씀**하기 (수업은 출석 인정, 진도는 따로 안 나감).

### GPU / NPU 메모리 이야기
- 프로젝트 규모상 **NPU/GPU 메모리**가 중요: 기존 회사 사례로 **48GB급** 장비, 학생 개인 장비는 **8GB** 수준 → 한계가 있음.
- 개인 구매 팁: **3090 / 4090 (VRAM 24GB)** 정도는 있어야 도전적인 프로젝트가 가능. 5090도 고려 대상 (32GB).
- **직접 써본 경험**(큰 VRAM, 파인튜닝 등)이 취업 후 차이를 만든다.
- 교육용 NPU는 **수급 문제**(물량 부족, 1인당 지급 불가, 양자화 권한·컴파일 제약 등)로 협의가 길어지고 있음 → 잘 안 되면 선생님도 같이 가서 해결해 주겠다고 함.
- 선생님은 **내년 2월쯤** 인프런/패스트캠퍼스 등에 강의를 올릴 계획.

---

## 📝 복습 과제 아이디어

- [ ] 병렬 그래프(`START → 검색/모델 → merge → END`)를 **직접 처음부터 타이핑**해 보기
- [ ] 메모리 3군데를 주석(`# 메모리 추가 1/2/3`)으로 표시해서 적용하고 **thread_id 바꿔가며** 기억 여부 확인
- [ ] 그래프 시각화 이미지를 저장(`wb`)해서 **발표용 PPT에 넣기**
- [ ] `merge_agent`를 **단순 붙이기 → 프롬프트 재조합(LLM)** 방식으로 바꿔 보기
- [ ] 우리 팀 프로젝트의 **다른 팀원 파트**를 직접 설명해 보며 전체 이해 점검
