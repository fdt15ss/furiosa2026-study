# LLM 6일차 정리

> 녹취(음성 인식) 텍스트에서 **랭체인 · 임베딩 · 딥러닝(케라스) · 파이썬 실습** 위주로 정리했습니다.
> 녹취에 잡음과 오인식이 많아, 확실하지 않은 용어는 일반적으로 쓰는 표기로 보정했습니다. (보정한 부분은 `※` 표시)

---

## 0. 오늘의 흐름

| 순서 | 주제 | 키워드 |
|---|---|---|
| 1 | PDF 문서 불러오기 | `PyPDFLoader`, Document 객체, 페이지 단위 로드 |
| 2 | 허깅페이스 임베딩으로 교체 | `BAAI/bge-m3`, `sentence-transformers`, 오프라인 모델 |
| 3 | 환경/버전 이슈 | torch–CUDA 버전 불일치, 라이브러리 자동 설치 |
| 4 | 임베딩 모델 하나 더 | Qwen3 임베딩, 1024차원 확인 |
| 5 | 트랜스포머 아키텍처 | 분기·병합, Add & Norm, Multi-Head Attention |
| 6 | 케라스 함수형 API 심화 | 앙상블, 다중 입력, `Concatenate`, 다중 출력 |

---

## 1. PDF 문서 불러오기 (랭체인)

지난 시간에 `TextLoader`로 텍스트 파일을 불러왔다면, 이번에는 **PDF**를 불러왔습니다.

### 핵심 포인트
- PDF 로더는 `langchain_community`의 document loaders에 들어 있음 → **랭체인 커뮤니티 패키지가 설치돼 있어야 함**
- PDF 파일은 프로젝트의 `data` 폴더 안에 넣고 경로를 지정
- 로드 결과는 **리스트** 형태이고, 안의 요소는 **Document 객체**
- Document 객체 구조는 TextLoader와 동일
  - `page_content` : 본문 텍스트
  - `metadata` : 출처, 페이지 번호 등
- **15페이지짜리 PDF → 길이 15인 리스트** (PDF 한 페이지가 Document 하나)
- PDF 로더를 쓰려면 `pypdf` 설치가 필요 ※

```python
# 예시 흐름 ※ (수업 코드 재구성)
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("data/attention.pdf")   # 'Attention Is All You Need' 논문으로 실습
docs = loader.load()

print(type(docs), len(docs))     # list, 15
print(docs[0].page_content)      # 본문
print(docs[0].metadata)          # 메타데이터
```

### 다음 단계 (이어서 할 일)
PDF 로드 → **청크로 자르기(분할)** → **임베딩** → **벡터DB 저장** → 검색/RAG 로 연결
(텍스트로 했던 파이프라인과 로더만 다르고 나머지는 동일)

---

## 2. 허깅페이스 임베딩 모델로 교체하기

기존에는 OpenAI의 `text-embedding-3-small`(API 키 필요, 유료)을 썼는데, 이번에는 **허깅페이스의 오프라인(로컬) 임베딩 모델**로 바꿔 실습했습니다.

### 2-1. 설치

```bash
# 가상환경 활성화 후
pip install langchain-huggingface
pip install sentence-transformers      # 실행 시 안내 메시지가 나와서 추가 설치
```

- `langchain-huggingface` : 랭체인에서 허깅페이스를 지원하는 패키지 (`pip show`로 설치 확인 가능)
- 실행하면 *"sentence_transformers를 설치하라"* 는 에러가 나옴 → **에러 메시지를 꼭 읽고 시키는 대로 설치**
- 패키지 이름은 보통 하이픈(`-`)으로 쓰는 경우가 많음 (`langchain-huggingface`)

### 2-2. 코드 변경점 (OpenAI → HuggingFace)

바꾼 부분만 비교하면 됩니다.

```python
# 기존 (OpenAI)
from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# 변경 (HuggingFace) ※
from langchain_huggingface import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={"device": "cpu"},          # GPU면 "cuda"
    encode_kwargs={"normalize_embeddings": True},   # 녹취상 옵션 일부는 불명확 ※
)
```

- `model` → **`model_name`** 으로 파라미터 이름이 달라짐
- API 키 관련 설정은 **필요 없음** (로컬 모델이므로)
- `device` 는 `cpu` / `cuda` 선택 가능 → CPU만 있어도 동작
- **처음 한 번만 다운로드**, 이후에는 로컬에 저장된 모델을 사용 (두 번째부터 훨씬 빠름)
- 오프라인이므로 **데이터가 외부로 나가지 않음** → 보안 측면 장점

### 2-3. BGE-M3 모델 특징
- **BAAI(베이징 인공지능 연구원)** 에서 만든 모델
- **한국어 · 일본어 · 중국어 등 다국어에 강함** → 한국어 말귀를 잘 알아듣는 편
- **임베딩 차원: 1024** (OpenAI `text-embedding-3-small`은 **1536**)

### 2-4. 자주 보이는 경고
- 실행 시 *허깅페이스 토큰(키) 관련 경고* 가 뜰 수 있음
- 지금은 **무시해도 동작에 문제 없음**
- 나중에 허깅페이스 모델을 많이 받게 되면 무료 토큰을 발급받아 `.env`에 넣어 쓰면 됨

---

## 3. 환경·버전 이슈 (실무에서 가장 많이 겪는 문제)

### 3-1. 패키지는 "친구들"과 함께 설치된다
`sentence-transformers` 설치 시 아래 같은 의존 패키지가 **자동으로 같이 설치/변경**됨:
- `torch` (PyTorch), `scikit-learn`, `scipy` 등

→ 이 중 **기존 환경과 버전이 충돌하는 게 있으면 에러**가 날 수 있음
(텐서플로 설치할 때도 numpy 등이 같이 설치되는 것과 같은 원리)

### 3-2. torch ↔ CUDA 버전 불일치
- 실행 시 *"Torch not compiled with CUDA enabled"* 에러
- 원인: 새로 설치된 **torch 버전에 맞는 CUDA**가 없음
  - 현재 PC는 **텐서플로(TensorFlow)용 CUDA**가 맞춰져 있음
  - torch가 요구하는 CUDA 버전과 다르면 GPU 사용 불가
- 그렇다고 torch용 CUDA를 새로 깔면 → **텐서플로 환경이 깨질 수 있음**

### 3-3. 대응 방법
| 상황 | 해결 |
|---|---|
| 지금 수업 실습 | **`device="cpu"`** 로 돌리면 해결 (굳이 GPU 필요 없음) |
| 진짜 GPU로 쓰고 싶을 때 | torch 버전 + CUDA 버전을 맞춰 **별도 가상환경**에 설치 |

### 3-4. 선생님 조언
- 수업에서는 "잘 되는 조합"만 골라서 깔아줬기 때문에 쉽게 느껴지는 것
- 직접 하면 **환경 세팅만 며칠** 걸릴 수 있음
- 회사에 가면 "이 버전 목록대로 설치" 하는 리스트를 줌 → **버전 목록 관리가 중요**
- 가장 힘든 건 **최신 버전**을 쓰는 것
- 랭체인도 **버전 차이**로 코드가 안 돌아가는 경우가 많음 (코드 오류가 아니라 버전 문제인 경우가 대부분, 0.1 단위 차이도 영향)
- **프로젝트에서는 PyTorch 환경을 다룰 일이 생길 가능성이 높으므로 미리 해 볼 것**

---

## 4. Qwen3 임베딩 모델로 한 번 더 교체

- 모델 이름만 바꾸면 되는 구조 (나머지 코드는 동일)
- **Qwen 계열은 알리바바(중국)** 에서 만든 모델 ※
- 파라미터 크기별 버전이 있어 **크기를 조절해 성능/속도를 조절** 가능
- 이 모델도 **임베딩 차원 1024** 로 확인

### 차원 확인 방법
```python
vec = embeddings.embed_query("삼성전자의 창업주는 누구인가요?")
print(len(vec))    # 1024
```

### 헷갈리기 쉬운 포인트 (수업 중 질문)
- 질문 **한 문장 → 벡터 1개**, 그 벡터의 **길이(차원)가 1024**
- "벡터가 몇 개?" → **1개**, "차원이 몇 개?" → **1024개**
- 문장(청크) 하나당 벡터 하나가 만들어지고, 모델마다 차원 수가 다름 (모든 차원을 외울 필요는 없고 필요할 때 `len()`으로 찍어보면 됨)

---

## 5. ⚠️ 임베딩을 바꾸면 벡터DB도 다시 만들어야 한다

이 날의 핵심 함정입니다.

- 챗봇의 임베딩 모델을 바꿨는데 **기존 벡터DB는 이전 차원(예: 1536)으로 저장**돼 있음
- 새 모델(1024차원)로 질문을 임베딩해서 검색하면 → **차원 불일치 에러**
- **해결: 벡터DB를 삭제하고, 새 임베딩 모델로 다시 만들어 저장**

```
임베딩 모델 변경  →  기존 벡터DB 폐기  →  새 모델로 문서 재임베딩  →  벡터DB 재생성
```

### 비교 실험 (4가지 조합)
| 번호 | 조합 |
|---|---|
| 1 | 기존 챗봇 + BGE-M3 |
| 2 | 기존 챗봇 + Qwen3 |
| 3 | PDF(논문) + BGE-M3 |
| 4 | PDF(논문) + Qwen3 |

- 각 조합에서 **같은 질문**을 던져 답변이 어떻게 달라지는지 **직접 체감**하는 것이 목적
- 확인 질문 예: **"삼성전자의 창업주는 누구인가요?"**
  - 문서에 근거가 없다면 → **"내용이 없어서 답할 수 없습니다"** 로 답해야 정상 (환각 방지)
  - 모델별로 이렇게 제대로 답하는지 비교
- "원래 쓰던 OpenAI 임베딩과 비교했을 때 한국어 이해가 더 나은가?"도 같이 확인

---

## 6. 트랜스포머(Transformer) 아키텍처

"AI 서비스의 가장 기본"이며 **면접에서 가장 많이 나오는 질문**.
(아키텍처는 **아래에서 위로** 읽는다)

### 6-1. 구조를 읽는 법
- 순차(Sequential) 모델은 위→아래 한 줄이지만, 트랜스포머는 **화살표가 갈라졌다(분기) 다시 합쳐진다(병합)**
- 입력 → **임베딩 → 포지셔널 인코딩(Positional Encoding)** → 인코더 블록 → 디코더 블록 → **Linear → Softmax** → 출력

### 6-2. 인코더 블록
1. 입력이 **두 갈래**로 나뉨
   - 한쪽: **Multi-Head Attention**으로 들어감
   - 다른 쪽: 그대로 **잔차 연결(skip connection)** 로 우회
2. 둘이 만나서 **Add & Norm** (더하고 정규화)
3. 다시 분기 → **Feed Forward** 통과 / 우회 → 다시 **Add & Norm**
4. 이 블록이 **N번 반복**(Nx)

### 6-3. 디코더 블록
- 출력 임베딩 + 포지셔널 인코딩 후,
- **Masked Multi-Head Attention** → Add & Norm
- 인코더 출력과 만나는 **Multi-Head Attention (Cross Attention)** → Add & Norm
- **Feed Forward** → Add & Norm
- 마지막에 **Linear → Softmax**
- 디코더의 **출력이 다시 입력으로 들어가는 구조** (자기회귀, 한 토큰씩 생성)

### 6-4. 설명할 수 있어야 하는 용어
| 용어 | 정리 |
|---|---|
| Feed Forward | 어텐션 뒤에 붙는 완전연결 층 |
| Add & Norm | 잔차 연결(더하기) + 정규화(Layer Normalization) ※ |
| Multi-Head Attention | 여러 관점(헤드)에서 동시에 어텐션을 계산해 합침 |
| Masked Multi-Head Attention | 미래 토큰을 못 보게 가려서 계산 (디코더에서 사용) |
| Positional Encoding | 순서 정보를 벡터에 더해줌 |
| Softmax | 마지막에 확률 분포로 변환 |

> 면접 단골: **"트랜스포머 아키텍처를 설명해 보세요"**, **"Multi-Head Attention과 Masked Multi-Head Attention의 차이는?"**

### 6-5. 실무에서는?
- 허깅페이스에서 트랜스포머 모델을 **import → 불러오기 → 파라미터 넣기** 만 하면 **코드 5줄 정도**로 사용 가능
- 하지만 **인코더/디코더 구조를 바꾸고 싶다면** 직접 구현해서 수정할 줄 알아야 함
- 선생님 강조: **모델 아키텍처를 완벽히 이해 → 직접 구현 가능 → 추론/학습 모두 가능**
  - 학습 없는 추론은 의미가 없고, 두 방향 모두 필요
  - 두 방향 모두에서 가장 중요한 것은 **논문 읽기(논문 기반)**
- 구현 방법은 **함수형(Functional) API**로 분기·병합을 표현

### 6-6. 논문 읽기 팁 (수업 초반)
- `arXiv`에서 **Attention Is All You Need** 검색 → 파생 논문(예: *Attention is not all you need* 류)이 엄청 많음
- 논문 PDF를 직접 로더로 불러 RAG 실습 자료로 활용

---

## 7. 케라스 함수형 API 심화: 앙상블 & 다중 입출력

> 오늘 실습한 모델 구조를 이해하기 위한 예시 데이터

### 7-1. 예시 데이터 (재미용 가상 데이터)
| 변수 | 의미 | 값 |
|---|---|---|
| `x1` | 두 컬럼 (예: 삼성전자 종가, 하이닉스 종가) | `range(100)` / `range(301, 401)` → 100행 × 2열 |
| `x2` | 세 컬럼 (예: 원유가, 환율, 금시세) | 100행 × 3열 |
| `y` | 맞출 값 (예: 화성의 화씨 온도) | 100개 |

- 의미상 말이 안 되는 조합이어도 **모델링은 가능** (재미 목적, 데이터 형태에 집중)
- **중요한 관점:** `x`만으로 `y`를 맞추는 게 아니라, 학습 과정에서 **`y`도 역전파를 통해 가중치에 영향**을 줌
  - 예: "국어·영어 점수로 체육 점수를 맞춘다" → 체육 점수(y)도 훈련에 포함되어 있음
  - 점쟁이처럼 아무 정보 없이 맞추는 게 아니라, **학습된 데이터 안에서의 관계**를 찾는 것

### 7-2. 앙상블(Ensemble) 모델이란?
- **여러 개의 모델을 합친 모델**
- 랜덤 포레스트(나무가 모여 숲)도 앙상블의 대표 예
- 단일 모델보다 **조금이라도 성능이 좋아질 수 있음** (안 좋아질 수도 있음 → **직접 해봐야 앎**)
- 대회(컴피티션)에서는 아주 작은 성능 향상도 순위를 크게 가름
- 입력 데이터마다 다른 모델 구조가 필요할 수 있음
  - 예: 이미지는 CNN, 시계열은 LSTM 으로 각각 구성한 뒤 합치기

### 7-3. 모델 1 · 모델 2 만들기 (함수형)

```python
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

# ---- 모델 1 : 입력 2열 ----
input1  = Input(shape=(2,))
d1 = Dense(10, name='ibm1')(input1)
d2 = Dense(20, name='ibm2')(d1)
d3 = Dense(30, name='ibm3')(d2)
d4 = Dense(40, name='ibm4')(d3)
output1 = Dense(5, name='ibm5')(d4)

# ---- 모델 2 : 입력 3열 (구조를 다르게 해도 됨) ----
input11 = Input(shape=(3,))
d11 = Dense(100, name='ibm11')(input11)
...
output11 = Dense(5, name='ibm15')(d15)
```

> ※ 레이어 이름과 노드 수는 녹취가 불명확해 대략적인 예시입니다. 핵심은 **서로 다른 구조의 모델 2개**를 만든다는 것.

#### 주의할 점
- 두 모델의 **레이어 `name`이 중복되면 에러** → 이름을 서로 다르게 지정
- 두 모델의 **구조는 달라도 됨** (마지막 출력 노드 수만 맞출 필요는 없고, 합치면 그냥 이어붙을 뿐)
- 입력 데이터 형태가 다르면 모델 구조가 다른 것이 자연스러움

### 7-4. 모델 합치기 (Concatenate)

```python
from tensorflow.keras.layers import Concatenate, concatenate

# 클래스 방식 (대문자)
merge1 = Concatenate(name='mg1')([output1, output11])

# 함수 방식 (소문자)
merge1 = concatenate([output1, output11], name='mg1')
```

- **`Concatenate`(대문자) = 클래스**, **`concatenate`(소문자) = 함수(메서드)** → 사용법만 다르고 **결과는 같음**
- 이름의 뜻: **사슬처럼 이어 붙인다** → 넘파이에서 배열을 이어 붙이던 것과 같은 개념
- **레이어 수가 늘어나는 게 아니라** 노드가 **이어 붙을 뿐**
  - 예: 5개 + 3개 → **8개 노드**를 가진 레이어처럼 동작
- 합쳐지는 시점에서 파라미터 수가 정해짐
  - 예: 합쳐진 8개 → 다음 Dense 10개 → `8 × 10 + 10(bias) = 90` 파라미터 ※
- `Concatenate` 에는 **두 개 이상은 반드시 리스트로** 넘겨야 함

### 7-5. 합친 뒤 이어지는 레이어와 최종 모델

```python
m2 = Dense(10, name='mg2')(merge1)
m3 = Dense(10, name='mg3')(m2)
last_output = Dense(1, name='last')(m3)

model = Model(inputs=[input1, input11], outputs=last_output)
model.summary()
```

- **입력이 두 개 이상이면 리스트**로 전달 (`inputs=[...]`)
- 출력도 두 개 이상이면 리스트
- `summary()`로 구조 확인 → 두 입력이 각각 흐르다 **merge에서 합쳐지는 형태**가 보여야 정상

### 7-6. 컴파일 · 학습 · 평가 · 예측

```python
model.compile(loss='mse', optimizer='adam')
model.fit([x1_train, x2_train], y_train, epochs=..., batch_size=...)

loss = model.evaluate([x1_test, x2_test], y_test)
y_pred = model.predict([x1_test, x2_test])
```

- **X가 두 개 이상이면 리스트**로 `fit / evaluate / predict`에 넣음
- 위에서 `train_test_split`을 **안 했던 부분은 숙제**: 직접 분리해서 적용
- **`train_test_split`은 한 번에 여러 데이터를 분리 가능**
  - `x1, x2, y` 3개를 한꺼번에 넣고 `x1_train, x1_test, x2_train, x2_test, y_train, y_test` 로 받으면 됨
  - 변수마다 따로 나눌 필요 없음 (행 인덱스가 같이 섞여 대응이 유지됨)
- **파라미터 튜닝**으로 성능 개선

> 데이터가 아주 규칙적이라 대충 만든 모델도 잘 맞는 것이 정상.

### 7-7. 입력 3개로 확장 (`x3` 추가)
- 모델을 하나 더 복사해서 입력 데이터만 추가 (`x1, x2, x3`)
- `y`는 그대로, 합치는 `Concatenate` 에 **리스트 요소를 하나 더** 넣음
- 모델 3개를 병합하는 것까지 실습

### 7-8. 다중 출력(분기) 모델

> 지금까지는 **여러 입력 → 하나의 출력** 이었다면, 이번에는 **출력도 여러 개**.

- 병합(`Concatenate`) 후 **다시 분기**해서 출력을 두 개로
- `y1` (예: 온도), `y2` (예: 비트코인 가격) 처럼 **서로 다른 목표값 2개를 동시에 예측**
- 분기마다 레이어 구성을 자유롭게 (깊게 / 얕게 / 레이어 없이 바로 출력)

```python
# 병합 결과(merge)에서 분기
# 분기 1 : 레이어를 쌓은 뒤 출력
b1 = Dense(10, name='b1')(merge)
last_output1 = Dense(1, name='last1')(b1)

# 분기 2 : 레이어 없이 merge에서 바로 출력
last_output2 = Dense(1, name='last2')(merge)

model = Model(inputs=[input1, input11, input21],
              outputs=[last_output1, last_output2])
```

- 학습 시 **`y`도 리스트**로 전달: `model.fit([x1, x2, x3], [y1, y2], ...)`
- 평가/예측 결과도 출력 개수만큼 나옴 → 각각 확인
- 분기는 **여러 개 만들어도 상관없음**
- **장점:** 여러 입력/출력이 서로 영향을 주고받아 **단일 모델보다 좋아질 수 있음**. 성능이 나빠지면 빼도 됨
- 이런 구조(합침 · 분기)는 이후 **트랜스포머, 스킵 커넥션 등**을 구현할 때의 기초가 됨

### 7-9. 오늘의 과제 / 주말 숙제
- `train_test_split` 적용 후 훈련 / 평가 / 예측 완성
- 결과 값(예측)이 **실제 다음 값에 맞게 나오는지** 확인 (예: 301~ / 100~ 부근 값)
- **다중 출력(분기) 모델**을 직접 구성해 보기
- 위 과제를 맞춘 사람은 **팀 프로젝트 시간에 자유롭게** (보상)

---

## 8. 오늘 배운 것 한눈에 보기

```
[랭체인]
 PyPDFLoader → Document(page_content, metadata) → 청크 분할 → 임베딩 → 벡터DB → 검색/RAG

[임베딩 교체]
 OpenAI (1536차원, API 키)  →  HuggingFace BGE-M3 / Qwen3 (1024차원, 로컬)
 ※ 임베딩을 바꾸면 벡터DB도 반드시 재생성

[환경]
 torch ↔ CUDA 버전 · 패키지 자동 설치 · 버전 목록 관리 · 막히면 device="cpu"

[딥러닝]
 트랜스포머 = 분기 + 병합 + 잔차 연결 + Attention
 함수형 API = 다중 입력(리스트) + Concatenate + 다중 출력(리스트)
 앙상블 = 여러 모델 합치기 (좋아질 수도, 아닐 수도 → 실험)
```

---

## 9. 복습 체크리스트

- [ ] PDF 한 페이지가 Document 하나로 로드된다는 것을 설명할 수 있다
- [ ] `HuggingFaceEmbeddings`에서 OpenAI 대비 바뀌는 파라미터(`model_name`, `device`)를 안다
- [ ] 임베딩 모델을 바꾸면 벡터DB를 다시 만들어야 하는 이유를 설명할 수 있다
- [ ] "벡터 개수"와 "차원 수"를 구분한다 (문장 1개 → 벡터 1개 → 차원 1024)
- [ ] torch와 CUDA 버전이 맞지 않을 때의 대처법(CPU 사용 / 별도 환경)을 안다
- [ ] 트랜스포머 구조(Add & Norm, Multi-Head, Masked, Feed Forward)를 말로 설명할 수 있다
- [ ] 함수형 API로 다중 입력 → `Concatenate` → 다중 출력 모델을 만들 수 있다
- [ ] `train_test_split`으로 여러 데이터를 한 번에 분리할 수 있다

---

## 부록: 녹취 중 참고 내용 (비학습)

- **프로젝트/발표 관련:** 팀 프로젝트는 4~5명 구성, 주제 발표는 팀원 중 랜덤으로 한 명이 함. 발표 자료에는 **핵심 코드**가 들어가야 하고, 본인이 만든 부분을 **디테일하게 설명**할 수 있어야 함. 포트폴리오는 "라벨링만 했다"가 아니라 **본인이 주도한 구조와 역할**이 드러나야 함. 주제는 *회사에 제출해도 통할 만한 것*을 기준으로 선정.
- **허깅페이스 이야기:** 허깅페이스 / 텐서 관련 AI 생태계에 대한 업계 이야기(인수, 개발자 생태계 등)가 곁들여졌으나 실습 내용과는 무관하여 생략.
- 점심시간, 팀 구성 대화 등 잡담은 정리에서 제외.
