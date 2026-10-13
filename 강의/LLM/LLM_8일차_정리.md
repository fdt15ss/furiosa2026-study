# LLM 8일차 정리

> 녹취 기반 정리 (랭체인 · 임베딩 · 딥러닝 등 파이썬 공부 위주)
> 녹취가 음성 인식 결과라 오타·잡음("양념장", "고춧가루" 등 반복 문구)이 많아, 의미가 명확한 내용만 추려 정리했습니다.

---

## 0. 오늘의 큰 흐름

| 시간대 | 주제 | 핵심 키워드 |
|---|---|---|
| 오전 | 딥러닝(텐서플로/케라스) | 랜덤 시드 고정, 가중치 확인, **동결(Freeze)**, 손으로 순전파 계산 |
| 오전 후반~오후 | **전이학습** | VGG16, `include_top=False`, 동결 vs 비동결, CIFAR10 적용 |
| 오후 | 랭체인 RAG 챗봇 업그레이드 | 검색 문서 출력, 유사도 임계값 retriever, 프롬프트 개선 |
| 마무리 | 팀/미니프로젝트 안내 | 팀 구성, 발표 기준, 라이선스 주의 |

---

## 1. 챗봇 히스토리(메모리) 복습

전날 내용 복습: Gradio 챗봇에 **대화 히스토리**를 넣어 메모리처럼 사용.

- 답변이 나올 때마다 히스토리에 계속 쌓으면 **토큰이 늘어 → 비용 부담** → 최근 N개만 유지.
- 메시지 역할 구분
  - `HumanMessage` (user, human) : 사람이 한 말
  - `AIMessage` (assistant, ai) : AI가 한 답변
- 개수 계산 주의
  - 히스토리를 10개로 제한 = (말한 것, 답한 것) 쌍이 5개
  - **내가 한 말만 10개** 유지하려면 20개(×2)로 잡아야 함
- 프롬프트에서는 `MessagesPlaceholder`(`chat_history`)로 히스토리를 끼워 넣음

```python
# 최근 N개만 유지하는 예시
chat_history = chat_history[-10:]
```

---

## 2. 랜덤 시드 고정 (재현성)

### 왜 필요한가
- 가중치 초기값이 **랜덤**이라 실행할 때마다 결과가 달라짐
- 초기값이 운 좋게 좋으면 빨리 수렴, 나쁘면 한참 헤맴 → "랜덤 맛집"
- 시드를 고정하면 **같은 코드 = 같은 초기 가중치 = 같은 결과**

### 코드

```python
import random
import numpy as np
import tensorflow as tf

random.seed(333)        # 파이썬 내장 random 고정
np.random.seed(333)     # 넘파이 고정
tf.random.set_seed(333) # 텐서플로 고정 → 초기 가중치 고정
```

- 난수표(암호표)를 하나 정해 놓고 그 표대로만 난수를 뽑는다고 이해
- 같은 시드여도 PC/환경(하드웨어 등)에 따라 값이 조금 다를 수 있음
- **그래도 텐서플로는 가끔 결과가 틀어질 수 있음** (완벽한 고정은 아님)

### 시드 값에 대한 주의
- `333`이 최적이라는 보장은 **없음** (예: 433으로 바꾸면 성능이 더 좋을 수도)
- 시드만 수천 번 바꿔 "최고 점수 시드"를 찾는 건 **요행**에 가까움 (특히 레이어가 얕을수록 영향이 큼)
- 시드 고정은 **비교 실험의 공정성**을 위한 것. 실험할 때는 항상 위 3줄을 써 두는 습관

---

## 3. 가중치 확인과 동결(Freeze)

### 3-1. 가중치 구조 확인

```python
print(model.weights)            # 전체 가중치(w, b 번갈아 나열)
print(model.trainable_weights)  # 훈련 가능한 가중치
print(len(model.weights), len(model.trainable_weights))  # 둘 다 6
```

- 예시 모델: 입력 1 → Dense(3) → Dense(2) → Dense(1)
  - w shape: `(1,3)`, `(3,2)`, `(2,1)` / b shape: `(3,)`, `(2,)`, `(1,)`
  - **리스트 원소 6개** = (w, b) × 3층
  - 파라미터 수: (1×3+3) + (3×2+2) + (2×1+1) = 6 + 8 + 3 = **17개**
- **bias 초기값은 항상 0**
- 동결 전에는 `weights`와 `trainable_weights`가 동일

### 3-2. 동결하기

```python
model.trainable = False   # 가중치 동결

print(model.trainable_weights)  # [] (빈 리스트)
print(model.weights)            # 가중치는 그대로 존재
model.summary()                 # Trainable params: 0 / Non-trainable params: 17
```

- 가중치는 **존재하지만 갱신되지 않는다** (= 최종 가중치로 고정)
- 동결된 모델로 `fit` 해도 **loss가 변하지 않음** (훈련이 안 되므로 초기 가중치로 예측만 함 → 결과가 엉망)
- 참고: 동결 설정은 `compile` 전에 하는 것이 안전 (바꾼 뒤에는 다시 compile)
- **추론 전용으로 쓰는 모델(GPT API, 허깅페이스 사전학습 모델 등)은 사실상 전부 동결 상태** → 가중치 갱신이 안 된다는 개념을 알고 있어야 함

### 3-3. 실습: 동결 vs 비동결로 [1,2,3,4,5] 맞추기

- 동결 안 한 모델로 `x = [1,2,3,4,5]`를 학습 → 예측이 거의 `1, 2, 3, 4, 5`에 가깝게 나와야 함
- 레이어 노드 수 예시: 20 → 20 → 20 → 10 → 1, 시드 고정, 에포크 수백 번
- 학습이 끝난 뒤 `model.weights`를 출력해 **손으로 순전파 계산**
  - 각 층: `H = X · W + b` (곱하기와 더하기만 반복)
  - 계산기를 써도 되고, 오차가 조금 있어도 OK → 결과가 약 1이 나오면 성공
- 포인트: **AI = 단순 곱셈과 덧셈의 반복**, 파라미터가 수억 개로 늘어난 것일 뿐

---

## 4. 전이학습 (Transfer Learning) — VGG16

### 4-1. 개념
- 남이 잘 만들어 학습시킨 모델(사전학습 모델, Pre-trained)의 **가중치를 가져와서 내 데이터에 활용**
- 핵심 구조: **위(입력) / 아래(출력) 부분만 내 데이터에 맞게 교체**, 가운데 특징 추출부(Conv)는 재사용
- 비슷한 도메인(이미지 → 이미지)일수록 잘 먹힐 확률이 높음

### 4-2. VGG16 구조
- ImageNet 대회(2014년경)에서 2등한 모델
- **Conv 13개 + Fully Connected(Dense) 3개 = 16층** → VGG16
- MaxPooling은 가중치(w, b)가 없으므로 **층 개수에 포함되지 않음**
- 원래 입력 `224×224×3`, 출력은 1000개 클래스(Softmax)
- 파라미터 약 **1억 3,800만 개** 중 1억 개 이상이 Flatten 뒤 Fully Connected 부분

### 4-3. 불러오기와 `include_top`

```python
from tensorflow.keras.applications import VGG16

# 원본 그대로 (224x224x3, 1000 클래스)
vgg16 = VGG16(weights='imagenet', include_top=True)

# 내 데이터용으로 커스터마이징
vgg16 = VGG16(weights='imagenet',
              include_top=False,          # Fully Connected 부분 제거
              input_shape=(32, 32, 3))    # 입력 크기 변경 가능
vgg16.summary()
```

| 옵션 | 의미 |
|---|---|
| `weights='imagenet'` | ImageNet으로 학습된 가중치 사용 (`None`이면 가중치 없는 깡 모델) |
| `include_top=True` | 원본 구조 유지 → 입력 크기 변경 불가(변경하면 에러) |
| `include_top=False` | Flatten 이후 Fully Connected 제거 → **입력 shape 변경 가능** + **출력부 커스터마이징 가능** |

- `top`은 CNN에서 관례적으로 **출력(분류) 쪽**을 부르는 이름 (위/아래 헷갈리기 쉬움)
- 파라미터: `include_top=False`, 32×32 입력 시 **약 1,400만 개** (약 10분의 1)
- 최소 입력 크기: MaxPooling이 5번 들어가 크기가 계속 반으로 줄어 소멸하므로, 너무 작으면 에러 → **32 이상**으로 설정 (에러 메시지에서 안내해 줌)

### 4-4. 내 모델에 붙이기 (CIFAR10 예시)

```python
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense

(x_train, y_train), (x_test, y_test) = cifar10.load_data()   # 32x32x3, 10 클래스

vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))
vgg16.trainable = False    # ← 동결 / True면 비동결(전체 재학습)

model = Sequential()
model.add(vgg16)                         # 사전학습 모델을 통째로 레이어처럼 추가
model.add(Flatten())                     # 직접 추가해야 함
model.add(Dense(100, activation='relu')) # 이 부분은 자유롭게 커스터마이징
model.add(Dense(100, activation='relu'))
model.add(Dense(10, activation='softmax'))  # 내 클래스 개수(10)로 출력
model.summary()
```

- **동결(`False`)**: Trainable params가 약 6만 2천 개 (내가 붙인 Dense 부분만 학습)
- **비동결(`True`)**: Trainable params가 약 1,400만 개 (전체 재학습)
- Flatten 대신 `GlobalAveragePooling2D` 등으로 커스터마이징도 가능

### 4-5. 동결 vs 비동결 — 실습 과제와 결과
- 비교 항목: **① 기존에 직접 만든 모델의 최고 성능 ② 동결 ③ 비동결 ④ 훈련 시간**
- 결과/교훈
  - 동결은 **훨씬 빠름** (13개 Conv를 다시 학습하지 않으므로), 비동결은 시간이 오래 걸림
  - 수업에서는 **CIFAR10(32×32)에서 성능이 오히려 떨어졌다는 학생이 많았음** (원래 224×224 이미지로 학습된 가중치라 작은 이미지와 맞지 않을 수 있음)
  - **"좋은 가중치라고 무조건 내 데이터에 바로 먹히지는 않는다 → 해봐야 안다"**
- 실전 순서
  1. 먼저 **동결**로 빠르게 시도
  2. 성능이 괜찮으면 그대로 사용
  3. 부족하면 **비동결(미세 조정, Fine-tuning)** 시도 (자원/시간 소모 큼)

---

## 5. 랭체인 RAG 챗봇 업그레이드 (RAG 22 시리즈)

### 전제 (기존 구성)
- 벡터 DB에는 이전 파일에서 만들어 둔 청크가 저장됨 (엔비디아 · 삼성전자 전망 텍스트 등)
- `create_stuff_documents_chain` + `create_retrieval_chain`을 합쳐 RAG 체인 구성
  - 이 두 체인은 **항상 함께 쓰이는 짝**
- Gradio에서 `response['answer']`를 반환하고, 히스토리를 `chat_history`에 누적

### 5-1. (22-1) 검색된 문서를 터미널에 출력하기
**문제**: 벡터 DB에서 **무엇을 검색했는지 모르는 채** 답변만 나옴 → 디버깅이 어려움

- `retriever`의 `k=2` → 검색 결과 문서가 **2개**
- `response['context']`는 **리스트(문서 객체)** → 반복문으로 출력
- `enumerate(..., start=1)`로 번호를 붙이면 편함 (기본은 0부터 시작)

```python
response = rag_chain.invoke({"input": message, "chat_history": chat_history})

# 검색된 문서 확인 (터미널에 출력)
for i, doc in enumerate(response["context"], start=1):
    print(f"[{i}] 출처: {doc.metadata['source']}")
    print(doc.page_content)

return response["answer"]
```

- `metadata`(딕셔너리) 안의 `source`(출처), `page_content`(본문)를 출력
- 장점: 답변이 이상할 때 **어떤 청크가 검색되어 들어갔는지** 확인 가능

### 5-2. (22-2) 유사도 임계값(Score Threshold) retriever
**문제**: "안녕", "내가 누구게?" 같은 질문에도 **관련 없는 문서까지 항상 k개 가져옴**

→ 일정 유사도 이하는 아예 가져오지 않도록 설정

```python
retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 2, "score_threshold": 0.7},
)
```

- `score_threshold=0.7`: 유사도 0.7 미만 문서는 **검색 결과에서 제외**
- 효과
  - 관련 없는 질문(인사 등) → 검색 결과 0개 → 컨텍스트 없이 답변 (**속도도 빨라짐**)
  - 벡터 DB 내용과 거의 같은 질문 → 유사도가 높아 정상적으로 2개 검색
- 기준값(0.5, 0.7, 0.8 등)은 **내 데이터에 맞게 직접 정함**
- 실습에서 확인한 현상: 엔비디아/삼성전자 질문은 잘 검색되지만, 벡터 DB에 없는 내용을 물으면 "주어진 정보로 답변할 수 없습니다"가 나옴

### 5-3. (22-3) 프롬프트 개선 (프롬프트 엔지니어링)
**문제**: 시스템 프롬프트에 "주어진 정보로 답변할 수 없으면 ~라고 답하라"만 있으면 **"안녕"이라고만 해도 "답할 수 없습니다"**가 나옴

**원인 분석 → 결론**
- 모델(OpenAI)과 retriever는 이미 정해져 있음 → **바꿀 수 있는 건 프롬프트뿐**
- 프롬프트를 신입 직원 교육하듯 **조건을 하나씩 구체적으로** 적어 줘야 함

**프롬프트 구성 예시**
```text
[페르소나] 당신은 강력한 AI 챗봇입니다. 다음 지침에 따라 답변하세요.
1. 이전 대화(chat_history)를 참고해 일관되게 답변합니다.
2. 컨텍스트에 질문과 관련된 내용이 있으면 컨텍스트만 근거로 답변합니다.
3. 인사 등 일반 대화는 자연스럽게 답변합니다.
4. 컨텍스트와 관련된 질문인데 컨텍스트에 답이 없을 때만 "주어진 정보로는 답변할 수 없습니다"라고 답합니다.
```

- 포인트: **"답할 수 없다"를 언제 말할지 조건을 명시** (관련 질문인데 근거가 없을 때만)
- 결과: 특별한 기교 없이 "문과 감성"의 평문 지시만으로도 훨씬 똑똑해짐
- 정리: RAG 체인에서 모델·retriever가 고정이면, **프롬프트를 어떻게 설계하느냐**가 품질을 좌우 → 히스토리·메모리도 최초 프롬프트 조건이 강력하면 그 아래 결과도 좋아짐
- 참고(강사 언급): 요즘 모델은 성능이 좋아져서 "영어는 please를 붙여라, 한국어는 존댓말을 써라" 같은 옛날 팁보다 **지시를 명확하게** 쓰는 것이 더 중요

### 5-4. 22 시리즈 한눈에 보기

| 파일 | 변경점 | 효과 |
|---|---|---|
| 22-1 | `enumerate`로 `context` 문서 터미널 출력 | 검색 결과 디버깅 |
| 22-2 | `similarity_score_threshold` + `score_threshold` | 관련 없는 문서 제외, 속도 향상 |
| 22-3 | 시스템 프롬프트에 페르소나·조건 명시 | "안녕"에도 "답할 수 없음" 나오는 문제 해결 |

---

## 6. 오늘 코드에서 챙길 파이썬/라이브러리 포인트

- `enumerate(iterable, start=1)` : 인덱스와 값을 같이 순회 (리스트 = 반복 가능한 객체)
- 딕셔너리 접근: `doc.metadata['source']`
- `model.trainable = False/True`, `model.weights`, `model.trainable_weights`, `model.summary()`
- `tf.keras.applications.VGG16(weights=..., include_top=..., input_shape=...)`
- Keras 버전이 바뀌며 `compile` 등 API/경로가 달라질 수 있음 → **에러 메시지를 끝까지 읽는 습관** (복붙해서 바로 GPT에 묻기보다 먼저 읽기)

---

## 7. 복습 체크리스트

- [ ] 시드 3종(`random`, `numpy`, `tensorflow`)을 모두 고정하는 이유를 설명할 수 있다
- [ ] `weights`와 `trainable_weights`의 차이를 설명할 수 있다
- [ ] 동결 시 `summary()`의 Trainable / Non-trainable이 어떻게 바뀌는지 안다
- [ ] 동결 모델에서 loss가 변하지 않는 이유를 안다
- [ ] VGG16이 Conv 13 + FC 3인 이유, MaxPooling이 층 수에 안 들어가는 이유를 안다
- [ ] `include_top=False`가 하는 일 두 가지(입력 shape 변경, FC 제거)를 말할 수 있다
- [ ] 동결 → 성능 부족 시 비동결(Fine-tuning) 순서로 실험한다
- [ ] `similarity_score_threshold` retriever를 직접 설정할 수 있다
- [ ] "안녕"에도 "답할 수 없음"이 나오는 문제를 프롬프트로 해결할 수 있다

---

## 특별 탭 : 수업 외 이야기 (AI 산업 · 팀 프로젝트)

### AI 산업/진로 관련 강사 이야기
- 지금 논문과 오픈소스 구현체 대부분이 **파이토치(PyTorch)** 기반 → 취업·연구를 생각하면 파이토치는 결국 할 줄 알아야 함
  - 수업이 텐서플로로 시작한 이유: 입문 난이도가 낮아서. 구조(개념)는 같고 문법만 다름
- AI 회사에서는 **최신 논문을 빨리 읽고 공유**하는 논문 스터디가 일상
- "AI 서비스 개발자"라는 직책은 불과 몇 년 전만 해도 거의 없었음
- (강사 개인 의견) 소버린 AI 등 국가 단위 AI 사업에 대한 견해 — 정치적 논쟁이 아니라 기술 의존성 관점에서의 이야기

### 팀/미니 프로젝트 안내
- 팀 구성: 4인 팀 둘 + 개인 프로젝트 2명, 나머지는 4~5인 팀
- 팀 아이디어 피드백에서 반복된 조언
  - **어떤 문제를 풀지, 어떤 데이터를 쓸지 방향부터 정하기** (데이터를 이것저것 끌어오기 전에)
  - 회귀/분류 모델 + RAG + Agent를 역할로 나누는 구성 (예: 여행 추천 에이전트, 주식·비트코인 예측)
  - LLM 호출은 비용이 들므로, 에이전트가 정해진 시간에 데이터를 수집해 **벡터 DB에 저장**해 두는 방식도 고려
  - 상용화를 염두에 둔다면 사용하는 **오픈소스/데이터의 라이선스**(MIT, Apache, GPL 등)를 먼저 확인
  - UI는 곁다리, **핵심은 기술력과 완성도** (포트폴리오로 보여줄 수 있는 수준)
- 발표 기준: 감성적인 이야기보다 **"어떤 공개 데이터로, 어떤 AI(에이전트/모델)를, 어떤 기술로 2주 안에 만드는가"**
  - 수상하지 못하더라도 **회사에 들이밀 수 있는 포트폴리오**가 되는 것이 목표
- 팀 프로젝트 마음가짐: 의견 충돌이 있어도 **화내지 않고 차분하게**
- 퇴실 체크 잊지 말기
