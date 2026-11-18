# LLM 21일차 정리 — 데코레이터 & 툴 콜링(Tool Calling)

> 오늘의 핵심: **데코레이터 → `@tool` → `bind_tools` → `tool_calls` → 도구 실행**
> 오전은 논문/발표 준비, 오후부터 진도. 다음 시간은 **트랜스포머 논문 발표**.

---

## 1. 파이썬 데코레이터 (Decorator)

### 1-1. 개념
- 함수는 매개변수로 **값(변수)** 뿐 아니라 **함수**도 받을 수 있다.
- **데코레이터 = 함수를 받아서 감싸는 함수** (선생님 정의: "함수를 불러오는 함수")
- 일반적인 정의: **기존 함수를 수정하지 않고 앞뒤에 기능을 추가하는 문법**
- 비유: 미용실(데코레이터 함수)에 들어갔다 나오면 머리에 꽃이 꽂혀서 나온다 → 기존 함수가 더 "강화된" 함수가 됨

### 1-2. 기본 예제
```python
def deco(func):
    def wrapper():
        print("함수 실행 전")
        func()
        print("함수 실행 후")
    return wrapper

def hello():
    print("안녕하세요")

@deco                 # hello2를 deco 함수 안에 집어넣는다
def hello2():
    print("안녕하세요")

hello()    # 안녕하세요
hello2()   # 함수 실행 전 → 안녕하세요 → 함수 실행 후
```
- `@deco` 는 사실상 `hello2 = deco(hello2)` 와 같다.
- `func` 라는 이름은 가독성을 위한 것일 뿐, `aaa`, `bbb` 등 아무 이름이어도 동작은 같다.

### 1-3. 실전 예제 — 실행 시간 측정 타이머
```python
import time

def timer(func):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"걸린 시간: {round(end - start, 4)}초")
        return result
    return wrapper

@timer
def hello():
    time.sleep(1)      # 1초 일시정지
    print("Hello")

@timer
def study():
    time.sleep(1.5)
    print("공부 중")

hello()
study()
```
- 매번 코드 앞뒤로 `start`, `end` 를 쓸 필요 없이 `@timer` 한 줄로 해결.
- 모델 학습/추론 시간 측정에 특히 많이 씀.
- **로그 기록, 예외 처리** 같은 공통 기능도 데코레이터로 붙이면 편하다.

### 1-4. 읽는 요령
- 코드에서 `@` 가 보이면 겁내지 말 것 → **바로 아래 함수를 `@` 뒤 함수 안에 넣는다**는 뜻.
- 내가 정의하지 않은 데코레이터라면 `import` 된 어딘가에 본체가 숨어 있다 (예: `langchain_core.tools.tool`).

---

## 2. 랭체인 `@tool` 데코레이터

### 2-1. 함수를 LangChain Tool 객체로 변환
```python
from langchain_core.tools import tool

@tool
def add(a: int, b: int) -> int:
    """두 정수를 더합니다."""
    return a + b

print(add.name)          # add  (원래 함수명)
print(add.description)   # 두 정수를 더합니다.  (docstring)
print(add.args)          # {'a': {'title': 'A', 'type': 'integer'}, 'b': {...}}
```
- `@tool` 이 `add` 를 **LangChain 툴 객체**로 바꿔준다 → `name`, `description`, `args` 속성이 새로 생김.
- 원래 함수에는 없던 속성들 → 데코레이터가 "감염"시켜서 생긴 것.

### 2-2. ⚠️ docstring(블록 주석)은 필수
- `"""두 정수를 더합니다."""` 를 지우면 **에러** 발생.
- 툴의 `description` 이 되기 때문 → LLM이 **어떤 도구를 고를지 판단하는 근거**가 됨.
- 따라서 **설명을 명확하게** 써야 툴 선택이 정확해진다.

### 2-3. 데코레이팅된 함수는 `invoke` 로 실행
```python
add.invoke({"a": 21, "b": 32})   # 53
```
- 이제 `add` 는 그냥 파이썬 함수가 아니라 **랭체인 툴** → 실행하려면 `.invoke()` 를 써야 한다.

---

## 3. 툴 콜링(Tool Calling) — LLM에 도구 엮기

### 3-1. 기본 흐름
```python
from kingkeras import get_llm          # 수업에서 만든 모듈(모듈/패키지 실습)
from langchain_core.tools import tool

llm = get_llm(max_tokens=200)

# 도구 등록
llm_with_tools = llm.bind_tools([add])   # 리스트 → 여러 개 엮기 가능

# 질문
response = llm_with_tools.invoke("21과 32를 더해줘")

print(response.content)             # 일반 응답 텍스트
print(response.tool_calls)          # LLM이 호출하려는 툴 정보
print(response.response_metadata)   # 메타데이터
```
- `bind_tools` : LLM에 툴을 **바인드(엮다)** → 이름 그대로 가독성 높은 API.
- 프롬프트 체이닝처럼, 이번엔 **도구를 붙여서 invoke** 하는 것뿐.

### 3-2. `tool_calls` 구조
```python
[
  {
    "name": "add",
    "args": {"a": 21, "b": 32},
    "id": "call_xxxxx",        # 자동 생성
    "type": "tool_call"
  }
]
```
- **리스트 형태** → 툴 호출이 2개 이상일 수 있다는 뜻 → `for` 문으로 처리.

### 3-3. 툴 실행 결과 뽑기
```python
for call in response.tool_calls:
    print("이름:", call["name"])
    print("인자:", call["args"])
    print("ID:", call["id"])

    result = add.invoke(call["args"])
    print("결과:", result)        # 53
```
- 지금은 리스트 원소가 1개라 1번만 돌지만, 여러 개 대비해서 for문으로 작성.
- `21 + 32` 를 툴 콜링으로 하는 건 낭비지만, 핵심은 **임의의 함수를 LLM과 엮을 수 있다**는 것
  (예: 블랙-숄즈 옵션 가격 모델, 퀀트 트레이딩 함수 등을 툴로 넣고 LLM이 사용).

---

## 4. 멀티 툴 — LLM이 도구를 "선택"

```python
from kingkeras import get_llm
from langchain_core.tools import tool

@tool
def add(a: int, b: int) -> int:
    """두 정수를 더합니다."""
    return a + b

@tool
def multiply(a: int, b: int) -> int:
    """두 정수를 곱합니다."""
    return a * b

@tool
def subtract(a: int, b: int) -> int:
    """두 정수를 뺍니다."""
    return a - b

tools = [add, multiply, subtract]
tool_map = {t.name: t for t in tools}   # {"add": add, "multiply": ..., "subtract": ...}

llm = get_llm(max_tokens=200)
llm_with_tools = llm.bind_tools(tools)

questions = [
    "15와 27을 더해줘",
    "12와 8을 곱해줘",
    "200에서 150을 빼줘",
]

for q in questions:
    response = llm_with_tools.invoke(q)
    print(f"\n질문: {q}")
    for call in response.tool_calls:
        print("선택한 도구:", call["name"])   # LLM이 고른 툴
        print("인자:", call["args"])
        result = tool_map[call["name"]].invoke(call["args"])
        print("실행 결과:", result)
```

**실행 결과**
| 질문 | LLM이 선택한 도구 | 결과 |
|---|---|---|
| 15와 27을 더해줘 | `add` | 42 |
| 12와 8을 곱해줘 | `multiply` | 96 |
| 200에서 150을 빼줘 | `subtract` | 50 |

### 포인트
- **도구 선택은 LLM(머신)이 한다.** 사람은 도구를 제시하고 프롬프트를 잘 다듬기만 하면 됨.
- LLM이 무엇을 골랐는지 미리 모르므로 `add.invoke` 처럼 직접 호출할 수 없음
  → `tool_map[call["name"]]` 로 **이름 → 툴 객체** 매핑해서 실행.
- 로봇(피지컬 AI)도 같은 원리: 음성 인식, 모터 제어 등 기능을 전부 **함수화 → 리스트화 → 툴 콜링**.

### 디버깅 팁
- 랭체인/랭그래프/LLM 쪽 데이터는 거의 다 **리스트 + 딕셔너리** 구조.
- 갑자기 모르는 속성(`name` 등)이 나오면 **일단 print 해서 찍어보기** → 대부분 딕셔너리 키에 숨어 있음.
- `print(tool_map)` → 키는 이름, 값은 툴 객체 전체. `print(tool_map["add"])` 처럼 하나만 보면 깔끔.

---

## 5. 퓨리오사 NPU 서버 설정 (리눅스)

- 퓨리오사 NPU 서빙은 **툴 콜링을 기본 지원하지 않음** → 실행 시 `enable auto tool choice` 관련 에러 발생.
- 서버 실행 명령에 옵션 2개 추가 (vLLM 계열 옵션):
```bash
--enable-auto-tool-choice --tool-call-parser hermes
```
  - `--enable-auto-tool-choice` : 툴 콜링 시 툴을 자동으로 선택하게 해주는 옵션
  - `--tool-call-parser hermes` : 툴 호출 응답 파싱 방식 지정
- 지난주 금요일에 넣은 **overlap scheduling 옵션**도 함께 유지.
- 서버 재시작: **`Ctrl + C` 로 종료** (`Ctrl + Z` 는 일시정지라 프로세스가 남으니 쓰지 말 것) → base 환경에서 옵션 넣고 재실행.
- 에러 나면 **99% 오타 아니면 띄어쓰기** 문제.
- 정상 기동 시 `Running ... 127.0.0.1 ...` 같은 로그가 뜸.
- 결론: **GPU에서 하는 작업을 NPU에서도 할 수 있다.**

---

## 6. 오늘의 요약

1. **데코레이터** = 함수를 받아서 감싸는 함수 (`@` 아래 함수를 넣는다)
2. **`@tool`** = 함수를 랭체인 툴로 변환 (docstring 필수 → description)
3. **`llm.bind_tools([...])`** = LLM과 도구 엮기
4. **`response.tool_calls`** = 리스트 of 딕셔너리 (`name`, `args`, `id`, `type`)
5. **`tool_map[name].invoke(args)`** = LLM이 고른 도구 실행
6. NPU 서버엔 `--enable-auto-tool-choice --tool-call-parser hermes` 꼭 추가

## 7. 공지 / 다음 일정
- 다음 시간: **트랜스포머 논문 발표** — 발표자뿐 아니라 전원이 완벽히 이해할 것.
- 팀원 발표 내용은 **반드시 직접 귀로 확인** (믿고 맡기면 다음 날 펑크남).
- 다음 주: 에이전트 파트 마무리 → **OpenClaw / Hermes 등 에이전트 설치·연결해서 나만의 자비스 만들기**.
- 파이썬 기초는 **클래스만 남음** (PyTorch 할 때 진행 예정).
- 오늘 내용 복습: **데코레이터부터 확실히**.
