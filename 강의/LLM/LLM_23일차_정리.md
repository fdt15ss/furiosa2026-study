# LLM 23일차 정리 — LangChain Agent로 계산기 만들기

> 오늘 주제: 툴 콜링(Tool Calling)을 한 단계 발전시킨 **LangChain `create_agent`** 실습
> 덧셈 → 곱셈 → 뺄셈·나눗셈(사칙연산) → 실행 횟수 제한 → 예외 처리 → **CLI 계산기 에이전트** 완성
> 모델: Qwen3 (로컬 실행. 같은 코드를 GPU 환경 등에서도 그대로 사용 가능)

---

## 0. 공지사항 요약

| 항목 | 내용 |
|---|---|
| 지각 기준 | 9시 정각. 시위·폭설·천재지변이라도 **지각 처리** (사유 인정 안 함) |
| 팀 프로젝트 | 이번 주 금요일 시작 → **다음 주 화요일 시작 발표**, 이후 **매주 화요일 진행상황 발표** |
| 논문 리뷰 | 팀별로 어디까지 할지 선택 (4번째 / YOLO까지 / 그 이상). 화·목 발표 |
| 선택 제출 | 이번 주 안에 **메일로** "어디까지 하겠다" 정리해서 제출. 제출 후 변경 불가 |
| 다음 주부터 | 팀 프로젝트 + 논문 + 보강 수업 시작 → 미리 준비해 둘 것 |
| 이번 주 목요일 | **ReAct 논문 코드 리뷰** — 오늘 코드 + **관찰(Observation) 부분 추가**해서 구현 |

### 논문 리뷰 작성 규칙 (YOLO)
- v1 ~ v11 기본 + 연도별 버전(26까지)
- **v1 ~ v5는 기본(Loss 함수 위주로)**, 이후 버전은 **"무엇을 어떻게 바꿨는지" 표로 명시**
- 버전마다 **최소 한 페이지 할당**해서 디테일하게 설명

---

## 1. 에이전트 결과에서 최종 답변 뽑기

에이전트를 `invoke` 하면 결과(`result`)에 `messages` 리스트가 담겨 옴.
**가장 마지막 메시지(`[-1]`)의 `content`가 최종 답변**이다.

```python
result = agent.invoke({"messages": [{"role": "user", "content": "3 곱하기 4 더하기 5는?"}]})

print(result["messages"][-1].content)   # 최종 답변
```

> 이전 시간에 직접 구현한 툴 콜링 루프보다 훨씬 간단하다.
> 에이전트 = "툴 콜링을 자동으로 반복해 주는 것" 정도로 이해하면 된다.

### messages 안에 들어있는 메시지 종류

| 타입 | 의미 |
|---|---|
| `HumanMessage` | 사용자 질문 |
| `AIMessage` | LLM 응답 (도구 호출 요청 `tool_calls` 포함 가능) |
| `ToolMessage` | 도구 실행 결과 |
| 마지막 `AIMessage` | 최종 답변 |

---

## 2. 어떤 도구를 썼는지 출력하기 — `isinstance` + `tool_calls`

```python
from langchain_core.messages import AIMessage   # 직접 쓰려면 import 필요

for msg in result["messages"]:
    if isinstance(msg, AIMessage) and msg.tool_calls:
        for call in msg.tool_calls:
            print("도구 이름:", call["name"])
            print("전달 인자:", call["args"])
```

### 파이썬 포인트
- **`isinstance(객체, 클래스)`** : 객체가 그 클래스의 **인스턴스(객체)** 인지 확인 → `True/False`
  - `HumanMessage`, `AIMessage`는 각각 **클래스**이고, `messages` 안의 요소들은 그 **인스턴스**
- **`and msg.tool_calls`** : 리스트가 비어 있으면 `False` → 도구 호출이 있는 AIMessage만 통과
  - HumanMessage → 탈락 / 도구 호출 없는 AIMessage(최종 답변) → 탈락 / 도구 호출 있는 AIMessage → 통과
- `name`과 `args`는 **LLM이 자연어를 보고 스스로 생성**한 것 (우리는 자연어로만 질문함)

> 메시지 개수를 일일이 셀 필요는 없음. 흐름만 이해하고 결과를 확인하면 된다.

---

## 3. 사칙연산 도구 4개 + 여러 질문 반복

```python
from langchain_core.tools import tool

@tool
def add(a: float, b: float) -> float:
    """두 수를 더합니다."""
    return a + b

@tool
def subtract(a: float, b: float) -> float:
    """첫 번째 수에서 두 번째 수를 뺍니다."""
    return a - b

@tool
def multiply(a: float, b: float) -> float:
    """두 수를 곱합니다."""
    return a * b

@tool
def divide(a: float, b: float) -> float:
    """첫 번째 수를 두 번째 수로 나눕니다."""
    return a / b

tools = [add, subtract, multiply, divide]   # 리스트로 따로 정리 → 가독성 ↑

agent = create_agent(model=llm, tools=tools, system_prompt="적절한 도구를 사용해 계산하세요.")

questions = [
    "100에서 35를 빼줘",
    "12 곱하기 7은?",
    "...",
    "...",
]

for question in questions:
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})

    for msg in result["messages"]:                       # ← 반드시 바깥 for문 안쪽
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for call in msg.tool_calls:
                print(call["name"], call["args"])

    print("최종 답변:", result["messages"][-1].content)
```

### 주의 1 — 복붙 실수
`divide` 함수 안에 `a + b`가 그대로 남아 있는 실수가 수업 중에 있었다. 도구를 복사해서 만들 땐 **본문 연산자까지 꼭 확인**.

### 주의 2 — for문 들여쓰기 버그 (수업 중 실제 발생)
- 증상: 질문 4개를 돌렸는데 **같은 결과(100−35=65)만 4번** 출력됨
- 원인: 출력 부분의 **들여쓰기 단계가 한 칸 어긋나** 의도와 다른 블록에서 실행됨
- 해결: 질문 for문 → `invoke` → 출력 for문이 **같은 반복 블록 안**에 있도록 들여쓰기 정리
  - VS Code: `Tab` 들여쓰기 / `Shift + Tab` 내어쓰기

### 프롬프트 팁
- 마침표, 존댓말, "please" 유무 등은 요즘 모델에선 **거의 신경 쓸 필요 없음**
- 적당한 오타도 괜찮다 (모델 성능이 좋아졌기 때문)

---

## 4. 복합 문장 — 도구 호출 횟수는 LLM이 결정

```python
question = "100에서 35를 뺀 후, 거기서 35를 빼고 3을 곱해줘"
# 100-35=65 → 65-35=30 → 30×3=90
```

- 한 문장이므로 질문 for문이 필요 없음 → `Shift+Tab`으로 바깥으로 꺼냄
- 도구 호출이 **3번** 일어남 — **몇 번 호출할지는 LLM이 자연어를 보고 판단**

---

## 5. 실행 횟수 제한 — `recursion_limit`

메시지 흐름: Human → AI(도구 요청) → Tool → AI → Tool → … → AI(최종)
호출이 100번, 1000번씩 반복되면 문제가 되므로 **최대 실행 단계를 제한**할 수 있다.

```python
result = agent.invoke(
    {"messages": [{"role": "user", "content": question}]},
    config={"recursion_limit": 20},    # 최대 실행 단계
)
```

- `20` → 정상 동작
- `2` → 바로 `GraphRecursionError` 발생
- 제한을 넘으면 **에러가 터지는 방식** (코드를 바꾸면 에러 없이 처리하는 것도 가능)

---

## 6. 파이썬 함정 — 콤마 하나로 튜플이 된다

`system_prompt`를 여러 줄로 쓸 때 헷갈렸던 부분.

```python
# (1) 괄호 안 문자열을 콤마 없이 나열 → 그냥 이어 붙은 "하나의 문자열"
prompt = (
    "너는 계산기야. "
    "도구를 사용해서 계산해."
)
print(type(prompt))   # <class 'str'>

# (2) 끝에 콤마를 찍으면 → 튜플!
prompt = (
    "너는 계산기야. "
    "도구를 사용해서 계산해.",
)
print(type(prompt))   # <class 'tuple'>  → 에이전트에서 400 Bad Request 등 에러 원인
```

> 인접한 문자열 리터럴은 자동으로 연결된다(implicit concatenation). **불필요한 콤마 금지.**

---

## 7. 예외 처리 — 0으로 나누기

```python
question = "10을 0으로 나눠줘"

try:
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    # ... 도구 출력 + 최종 답변
    print(result["messages"][-1].content)
except ZeroDivisionError as e:
    print("0으로 나눌 수 없습니다:", e)
except Exception as e:                  # 예상 못 한 다른 에러 대비
    print("에이전트 실행 중 오류:", e)
```

### 결과는 3가지 중 하나
1. 모델이 똑똑해서 스스로 "0으로 나눌 수 없다"고 **최종 답변**까지 감
2. `ZeroDivisionError`로 떨어짐
3. 다른 에러(`Exception`)로 떨어짐

> 일단 try/except 틀을 잡아두고 나중에 살을 붙이면 된다.

---

## 8. 최종 실습 — CLI 계산기 에이전트

질문을 코드에 하드코딩하지 않고 **터미널에서 직접 입력**받는 버전.

```python
def run_agent(question: str):
    print(f"\n질문: {question}")
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})

    for msg in result["messages"]:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for call in msg.tool_calls:
                print(f"  도구: {call['name']}, 인자: {call['args']}")

    print("최종 답변:", result["messages"][-1].content)
    return result


while True:
    question = input("\n계산할 내용을 입력하세요 (종료: q): ")

    if question.lower() == "q":      # 대문자 Q도 소문자로 바꿔서 비교
        print("계산기를 종료합니다.")
        break

    try:
        run_agent(question.strip())  # 앞뒤 공백 제거 후 전달
    except Exception as e:
        print("오류:", e)
```

### 파이썬 포인트
| 코드 | 의미 |
|---|---|
| `input("...")` | 터미널에 안내 문구 출력 + 사용자 입력 받기 (문자열) |
| `.lower()` | 소문자로 변환 → `Q`/`q` 모두 종료 |
| `.strip()` | 앞뒤 공백 제거 |
| `break` | `while True` 무한 루프 탈출 |
| 함수 + try/except | 결과 생성 로직을 함수로 묶으면 **함수 호출부만 예외 처리**하면 됨 |

> 실행 예: `34 곱하기 7 곱하기 8` → 1904. LLM이 질문을 잘못 해석해 엉뚱한 도구를 쓸 수도 있으니 **도구 이름·인자 로그로 검증**하는 습관!

---

## 9. 오늘 내용의 의미 — ReAct 개념

- 오늘 만든 것 = **아주 단순한 에이전트**, 툴 콜링을 조금 업그레이드한 것
- 사용한 `create_agent`는 LangChain이 제공하는 **가장 간단한 기본형 에이전트**
- 사실 이것이 **ReAct**(Reasoning + Acting)의 구조

```
Thought(추론) → Action(행동: 도구 호출) → Observation(관찰: 결과 확인) → 반복 → 최종 답변
```

- 오늘 코드에는 **관찰(Observation) 부분이 명시적으로 빠져 있음**
- ✅ **목요일 ReAct 논문 코드 리뷰 과제**: 오늘 코드를 사용해도 되지만 **관찰 단계를 추가**해서 구현
  - 다른 에이전트 방식을 써도 됨

---

## 10. 한눈에 요약

1. `result["messages"][-1].content` → 최종 답변
2. `isinstance(msg, AIMessage) and msg.tool_calls` → 도구 호출 메시지만 골라내기
3. `call["name"]`, `call["args"]` → LLM이 스스로 만든 도구 이름/인자
4. 도구는 리스트로 정리, 복붙 시 연산자 확인
5. for문 들여쓰기 한 칸 차이로 결과가 완전히 달라짐
6. `config={"recursion_limit": N}` → 최대 실행 단계 제한 (초과 시 에러)
7. 문자열 끝 콤마 하나 → 튜플로 변해서 에러
8. `try / except ZeroDivisionError / except Exception`
9. `input()` + `lower()` + `strip()` + `while True/break` → CLI 계산기
10. 에이전트 = ReAct (추론 → 행동 → 관찰), 목요일엔 **관찰 단계 추가**
