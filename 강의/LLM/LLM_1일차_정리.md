# LLM 1일차 정리 — LangChain(랭체인) 기초 & OpenAI API 세팅

> 강의 녹취(음성 인식본)를 바탕으로 랭체인 파이썬 실습 위주로 정리했습니다.
> 녹취에 인식 오류가 많아 **코드 예시는 강의 흐름에 맞춰 재구성한 참고용**입니다. 모델명·base URL 등은 강의에서 제공된 값으로 바꿔 쓰세요.

---

## 1. 오늘의 큰 흐름

1. 개발 환경 준비 (스터디 폴더 → 가상환경 → 패키지 설치)
2. OpenAI 플랫폼 가입 / 크레딧 충전 / API Key 발급
3. **API Key를 안전하게 다루는 방법** 4가지
4. 랭체인의 핵심 개념: **프롬프트 → 모델 → 출력 파서**, 그리고 이를 잇는 **LCEL**
5. RAG(검색 증강 생성) 개념 소개

---

## 2. RAG란?

**RAG = Retrieval(검색) + Augmented(증강) + Generation(생성)**

| 단계 | 설명 |
|---|---|
| 질문 | 사용자가 질문을 입력 |
| Retrieve (검색) | 일반 검색이 아니라 **벡터 스토어(Vector Store)** 에서 유사한 문서를 검색 |
| Augment (증강) | 검색된 내용을 프롬프트에 붙여 넣어 보강 |
| Generate (생성) | LLM이 보강된 프롬프트를 보고 답변 생성 |

- 이번 수업 중에 앞에서부터 하나씩 구현해 나갈 예정 (수업 스타일: **반복**하며 익히기)
- 다음 단계 예고: **임베딩(Embedding)** 과 **코사인 유사도** → 임베딩은 TensorFlow/PyTorch로 구현할 수 있어야 함

---

## 3. 개발 환경 세팅

### 3-1. 작업 폴더 & 가상환경

1. 스터디 폴더 생성 후 VS Code에서 열기
2. 터미널(CMD)에서 **새 가상환경** 생성 (기존 TensorFlow 환경과 분리, 강의에서 지정한 Python 버전 사용)
3. 가상환경 **activate**
4. 패키지를 **하나씩** 설치하며 각각이 무슨 기능인지 확인

```bash
pip install langchain
pip install langchain-openai
```

- `langchain` : 랭체인 코어
- `langchain-openai` : OpenAI 모델 연동 (강의에서 "랭체인, 랭체인 AI" 두 개 설치)

### 3-2. VS Code 팁

- 패키지 설치 후 인식이 안 되면 → `Ctrl + Shift + P` → **Reload Window** (또는 인터프리터를 가상환경으로 선택)
- 환경 변수/.env 변경 후에는 VS Code를 재시작해야 반영되는 경우가 있음
- VS Code를 여러 개 띄워 GPU 번호를 나눠(예: 0, 1, 2, 3) 동시에 작업하는 것도 가능

---

## 4. OpenAI API Key 준비

1. OpenAI **Platform** 접속 → 회원가입/로그인
2. 결제 수단 등록 후 **Add credit** 로 소액 충전 (실습은 약 **$5** 수준이면 충분, 필요 이상 충전 X)
3. **자동 충전(Auto recharge)** 옵션은 의도치 않게 크게 결제될 수 있으니 주의
4. API Key 발급 → 메모장 등에 **복사만** 해 둠 (발급 시 한 번만 보임)
5. 수업에서는 키와 함께 **base URL**(교육용 엔드포인트)이 제공되므로 **키 + URL을 같이** 설정

### 주의할 점

- **API 호출은 실행할 때마다 토큰(=돈)이 소모**됨. `Ctrl+F5` 등으로 실행할 때마다 과금
- 코드에 메모리 기능이 없으면 같은 질문도 매번 새로 호출됨
- 모델마다 **가중치·성능·속도·가격이 달라서 같은 프롬프트라도 답변이 다름** (실습은 지정된 모델로 통일)
- API로 보낸 데이터는 학습에 쓰이지 않는다고 설명 / 웹 ChatGPT는 대화가 메모리에 일시 저장됨
- **할루시네이션(Hallucination)** = AI가 그럴듯한 거짓말을 하는 현상 → 항상 검증

---

## 5. API Key 관리 4가지 방법 ⭐

| # | 방법 | 특징 | 평가 |
|---|---|---|---|
| 1 | **코드에 직접 입력** (하드코딩) | 가장 단순 | ❌ 코드가 GitHub 등에 올라가면 키 유출 → 과금 폭탄 / 최악 |
| 2 | **`os.environ`으로 코드 안에서 지정** | 실행 중에만 유지(휘발성) | 단기 테스트용, 장기 작업에는 불편 |
| 3 | **시스템 환경 변수에 등록** | 컴퓨터 전체에서 사용 가능 | 자주 쓰이지만, 공용 PC/옆 사람이 시스템 환경 변수를 보면 노출 위험 |
| 4 | **`.env` 파일 + `python-dotenv`** | 프로젝트별로 키를 파일에 분리 | ✅ **가장 많이 사용** (3번과 4번이 주류, 4번이 조금 더) |

### 5-1. `os.environ` 방식

```python
import os
os.environ["OPENAI_API_KEY"] = "sk-..."   # 실습용, 코드에 남기지 말 것
```

### 5-2. `.env` 방식 (권장)

**① 프로젝트 폴더에 `.env` 파일 생성**

```env
OPENAI_API_KEY=sk-xxxxxxxx
OPENAI_BASE_URL=https://...   # 강의에서 제공된 URL
```

**② 패키지 설치**

```bash
pip install python-dotenv
```

**③ 코드에서 로드**

```python
from dotenv import load_dotenv
load_dotenv()          # .env를 찾아 환경변수로 등록
```

### 5-3. `.env` 사용 시 체크리스트

- ✅ `.env`는 **현재 작업 디렉터리**에 있어야 `load_dotenv()`가 자동으로 찾음
  - 다른 폴더에 있다면 `load_dotenv("경로/.env")` 처럼 **경로를 지정**
  - 예: 작업 폴더가 `study`인데 `.env`가 `data/` 안에 있으면 경로를 잡아줘야 함
- ✅ 변수 이름 철자 확인 (`OPENAI_API_KEY` 등, 랭체인이 기본으로 읽는 이름을 사용)
- ✅ **GitHub 업로드 전 반드시 `.gitignore`에 `.env` 추가**
  - 키가 올라가는 순간 = **퇴사각** (수업 중 강조한 포인트)
- ✅ 회사 컴퓨터에서 남의 키를 보게 되어도 쓰지 말 것 / 자리를 비울 때 환경 변수 노출 조심
- ✅ 키가 없거나 틀리면 에러가 남 → **에러 메시지를 직접 읽는 습관** 들이기

```
# .gitignore
.env
```

---

## 6. LangChain 핵심 개념

### 6-1. 웹 ChatGPT를 코드로 구현한다는 것

ChatGPT 창에 글을 쓰고 엔터를 치면 답이 나오는 과정을 코드로 옮기면 다음과 같다.

```
프롬프트(Prompt) ──▶ 모델(Model) ──▶ 출력 파서(Output Parser)
   명령어              LLM 호출            결과를 문자열 등으로 정리
```

- **Prompt** : 모델에 넣을 명령어(템플릿)
- **Model** : LLM (예: OpenAI 챗 모델)
- **Output Parser** : 모델 응답에서 필요한 내용만 뽑아 원하는 형태로 변환

### 6-2. Chain(체인) = 연결된 파이프라인

> "어렵게 생각하지 말고, **파이프라인으로 연결**해 주세요."

프롬프트 → 모델 → 파서를 순서대로 이어 놓은 것이 **체인**이다. 결국 이 순서대로 연결만 하면 된다.

### 6-3. LCEL (LangChain Expression Language)

- `|`(파이프) 연산자로 컴포넌트를 연결하는 랭체인의 표현 방식
- 파이썬의 `|` 연산자를 **오버라이딩(재정의)** 해서 "얘와 얘는 연결 관계로 작업하겠다"는 뜻을 부여한 것

```python
chain = prompt | model | parser
```

---

## 7. 실습 코드 예시 (재구성)

### 7-1. 가장 단순한 모델 호출

```python
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

model = ChatOpenAI(
    model="강의에서 지정한 모델명",
    # base_url, api_key는 .env의 환경변수를 자동으로 읽음
    # (자동 인식이 안 되면 base_url=..., api_key=... 로 명시)
)

response = model.invoke("한국의 수도는 어디인가요?")
print(response.content)
```

### 7-2. 프롬프트 템플릿 + 모델 + 파서 = 체인

```python
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# 1) Prompt: {변수}가 들어가는 템플릿
prompt = ChatPromptTemplate.from_template(
    "{topic}에 대해 설명해 주세요."
)

# 2) Model
model = ChatOpenAI(model="강의에서 지정한 모델명")

# 3) Output Parser
parser = StrOutputParser()

# 4) Chain (LCEL)
chain = prompt | model | parser

# 5) 실행
result = chain.invoke({"topic": "디지털 샤머니즘"})
print(result)
```

### 7-3. 실습하며 확인한 것들

- **프롬프트를 바꿔가며 여러 번** 실행해 보기 (템플릿의 `{변수}`만 바꿔 재사용)
- **모델을 바꾸면 답변·속도가 달라짐**을 직접 비교
- 수강생 30명이 동시에 호출하면 **속도가 느려질 수 있음** (서버 부하)
- 프롬프트 템플릿은 나중에 검색 결과(context)를 넣는 자리로 확장됨 → RAG로 연결
- 질문이 같아도 모델에 따라 결과가 달라서 **아웃풋 파서**로 형태를 정리하는 것이 중요

---

## 8. 한 장 요약 (복습용)

- **RAG** = 질문 → 벡터 스토어 검색 → 프롬프트 증강 → 생성
- **LangChain 체인** = `prompt | model | parser`
- **API Key**는 코드에 직접 쓰지 말고 **`.env` + `load_dotenv()`** 로 관리, `.gitignore`에 필수 등록
- `.env`는 작업 디렉터리에 두거나 경로를 지정
- API 호출은 **돈이 나간다** → 크레딧은 소액, 불필요한 반복 실행 지양
- 에러가 나면 **복붙 전에 먼저 직접 읽기**
- **할루시네이션** 조심: AI의 답은 검증할 것

---

## 부록. 같은 날 함께 다룬 딥러닝 복습 내용 (짧게)

랭체인 수업과 별개로 녹취에 섞여 있던 딥러닝 실습 내용입니다.

- **양방향(Bidirectional) RNN**
  - `Bidirectional` 레이어는 모델이 아니라 **래퍼(Wrapper)**: 안에 `SimpleRNN`/`LSTM`/`GRU` 같은 레이어를 넣어 감싸는 방식
  - 적용하면 **파라미터 수가 정확히 2배** (예: 120 → 240) → `model.summary()`로 확인
  - 속도는 느려질 수 있고, **성능 향상 여부는 실험으로 비교** (RMSE 기준)
- **스케일링**
  - X뿐 아니라 **Y도 스케일링**해야 큰 값 때문에 loss·gradient가 불안정해지는 것을 방지
  - 풍향처럼 0~360도 값은 **sin/cos 변환**으로 순환성(350°와 1°가 가깝다는 점)을 반영
  - 예측 대상(온도)이 입력 피처에 들어가면 안 됨 → 피처에서 **drop**
- **위치 인코딩**: sin/cos를 이용한 방식이 있고 다른 방식도 있음 (임베딩 수업 때 다룰 예정)

---

## 다음 수업 준비

- [ ] 임베딩 개념 & 코사인 유사도 복습
- [ ] TensorFlow / PyTorch 임베딩 구현 준비
- [ ] `.env` 세팅 재확인 (`.gitignore` 포함)
- [ ] 벡터 스토어 + 검색 체인 구현 예습
