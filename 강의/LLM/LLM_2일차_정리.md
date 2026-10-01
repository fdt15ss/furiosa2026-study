# LLM 2일차 정리 — 임베딩(Embedding)과 랭체인 OpenAIEmbeddings

> 핵심 한 줄: **AI는 말을 이해하는 게 아니라, 텍스트를 숫자(벡터)로 바꿔서 연산하고 그 결과를 다시 글자로 보여주는 것.** 이 "숫자로 바꾸는 과정"의 기준이 임베딩이다.

---

## 0. 오늘의 흐름

```
텍스트 → 토크나이징(수치화) → 패딩(길이 맞추기) → 원핫 인코딩(문제점 발견)
      → 임베딩 레이어(벡터화) → 랭체인 OpenAIEmbeddings → 코사인 유사도
```

- 앞부분(토크나이저 ~ 원핫)은 **"왜 임베딩이 필요한가"** 를 이해하기 위한 빌드업
- 실무에서는 토크나이저/임베딩을 직접 만들 일이 거의 없고, **임베딩 모델 API를 호출**해서 쓴다
- 다음 시간(3일차)은 **벡터 스토어 / 벡터 DB**

---

## 1. 토큰(Token)

- 모델이 텍스트를 쪼개는 단위. 강의에서 정리한 감각은 **"음절 ~ 형태소 정도 크기의 말 덩어리"**
- 정확한 기준은 **제공사(모델)마다 다르다**. 특히 한국어는 더 제각각
  - 한국어 단위 복습: 음절(글자 하나) / 어절(띄어쓰기 단위) / 형태소(뜻을 가진 최소 단위)
  - 영어도 단어 단위와 글자 조각 단위 사이를 오간다
- "토큰 사용량이 많다" = 이 말 덩어리를 많이 썼다는 뜻 → **API 비용과 직결**
- 정확한 개수가 필요하면 제공사의 토크나이저로 직접 확인

---

## 2. Keras Tokenizer로 수치화하기 (연습용)

```python
from tensorflow.keras.preprocessing.text import Tokenizer

text = "나는 지금 진짜 진짜 매우 매우 맛있는 국밥을 엄청 마구 마구 마구 마구 먹었다."

tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])          # 사전 만들기 (여러 문장이면 리스트로)
print(tokenizer.word_index)             # {'마구': 1, '진짜': 2, '매우': 3, '나는': 4, ...}

seq = tokenizer.texts_to_sequences([text])
print(seq)                              # [[4, 5, 2, 2, 3, 3, 6, 7, 8, 1, 1, 1, 1, 9]]
```

### 핵심 포인트
| 개념 | 설명 |
|---|---|
| `fit_on_texts` | 텍스트로 단어 사전을 학습(`fit`). 인자는 **리스트** (문장이 1개 이상 들어갈 수 있으니까) |
| `word_index` | **빈도 높은 순**으로 번호 부여. 빈도가 같으면 **먼저 나온 단어가 앞 번호** |
| `texts_to_sequences` | 문장 → 숫자 리스트로 변환 |
| 번호는 1부터 | **0은 비워둠** → 나중에 패딩 값으로 사용 |

- 이 Tokenizer는 **어절(띄어쓰기) 단위**로 자른다. 연습용으로 충분
- 클래스를 `Tokenizer()`로 만드는 것 = **인스턴스화** ("토크나이저를 인스턴스화했다"), 함수는 값을 **리턴(반환)** 한다고 표현

---

## 3. 패딩(Padding) — 길이 맞추기

문장마다 길이가 달라서 그대로는 모델에 넣을 수 없다 → **길이를 통일**한다.

- 짧으면 **0으로 채우고**, 길면 **자른다** (오디세우스의 침대 비유)
- CNN에서 이미지 가장자리를 0으로 채우던 padding과 같은 개념

```python
from tensorflow.keras.preprocessing.sequence import pad_sequences

X = pad_sequences(sequences, maxlen=5, padding='pre', truncating='pre')
# X.shape → (문장 수, 5)   예) (15, 5)
```

| 옵션 | 의미 |
|---|---|
| `maxlen` | 맞출 길이 |
| `padding='pre'` | **앞**을 0으로 채움 (기본값) |
| `padding='post'` | **뒤**를 0으로 채움 |
| `truncating='pre'` | 길 때 **앞**을 자름 (기본값) |
| `truncating='post'` | 길 때 **뒤**를 자름 |

**어느 쪽을 자르고 채울까?**
→ "중요한 의미는 뒤에 있다(끝까지 들어봐야 안다)"면 **앞을 채우고(pre) 앞을 자른다.** 문맥상 앞이 중요하면 반대로.

---

## 4. 왜 원핫 인코딩으로는 부족한가

- 단어를 번호(1, 2, 3…)만 붙여 쓰면 **숫자 크기에 의미가 생긴다** (마구 2개 = 진짜?, 4 > 3 이라 더 중요?). `y = wx + b` 연산에서 값이 더해지면 엉뚱한 의미가 되어버림
- 그래서 번호를 **위치 값**으로 바꾸는 **원핫 인코딩** 사용 (`to_categorical`, `OneHotEncoder` 등)

```
단어 3  →  [0, 0, 0, 1, 0, 0, ...]
```

- (이전에 분류 문제에서 라벨에 쓰던 **소프트맥스 + 원핫**과 같은 원리를 입력 X에도 적용한 것)

### 원핫의 문제점
1. **차원 폭발**: 단어 사전이 30개면 벡터 길이 30, 옥스포드 사전 20만 단어면 **벡터 길이 20만**
2. 1은 하나뿐이고 **나머지는 전부 0** → 의미 없는 연산이 엄청나게 늘어남
3. 입력 shape가 `(15, 5)` → `(15, 5, 30)`처럼 커지고, **모델 파라미터/연산량 급증**
4. 패딩으로 넣은 0도 단어 1개로 취급되어 불필요한 칸이 하나 더 생김
5. (질문에서 나온 한계) 0 패딩을 앞뒤로 채우면 **시계열/순서 정보가 흐트러질 수 있음**

> 결론: **작은 차원의 "밀집(dense) 벡터"로 단어를 표현하자 → 임베딩**

---

## 5. 임베딩(Embedding) 개념

- **단어(토큰)를 의미를 담은 실수 벡터로 바꾸는 것**
- 원핫: `[0,0,0,1,0,...]` (희소, 차원 큼)  →  임베딩: `[0.12, -0.5, 0.33, ...]` (밀집, 차원 작음)
- 벡터 하나의 **값 개수 = 차원(dimension)**
  - 강의 포인트: "1536차원"은 **벡터 하나가 1536개의 숫자로 이루어져 있다**는 뜻 (가로로 1536칸). 스칼라가 1536개 모인 것
- 이 벡터들이 모인 공간을 **벡터 스페이스**, 저장하는 곳이 **벡터 스토어/벡터 DB** (3일차)
- 의미가 비슷한 단어/문장은 **벡터 공간에서 가깝게(비슷한 방향으로)** 위치한다

---

## 6. 텐서플로(Keras) 임베딩 레이어

원핫을 지우고, **패딩까지 한 X `(15, 5)`를 그대로** 임베딩 레이어에 넣는다.

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

model = Sequential()
model.add(Embedding(input_dim=31, output_dim=10))   # 단어사전 개수, 벡터 차원
model.add(LSTM(...))
model.add(Dense(1, activation='sigmoid'))
model.build(input_shape=(None, 5))
model.summary()
```

### 인자 정리
| 인자 | 의미 | 주의 |
|---|---|---|
| `input_dim` | **단어 사전의 개수** (토큰 종류 수) | 패딩 0 때문에 **+1** (30종류면 31). 실제 최댓값보다 **작게 잡으면 에러**, 크게 잡으면 동작은 하나 파라미터 낭비 |
| `output_dim` | **만들고 싶은 벡터 차원** | 우리가 정함 (강의에선 10). GPT는 1,500대 |
| `input_length` | 입력 길이(여기선 5) | 최신 Keras에서는 **생략 가능/제거**됨. 안 써도 알아서 맞춤. 틀린 값(6 등)을 주면 shape 에러 |

### shape 변화 (중요)
```
입력   (15, 5)         ← 정수 인덱스 2차원 (원핫 필요 없음!)
출력   (15, 5, 10)     ← 단어 하나하나가 10차원 벡터로 바뀌어 3차원이 됨
```
→ 그래서 **LSTM에 바로 넣을 수 있는 3차원 데이터**가 된다 (예전에는 원핫으로 3차원을 억지로 만들었음)

### 파라미터 개수 계산
```
파라미터 = input_dim × output_dim
         = 30 × 10 = 300     (31이면 310, 32면 320 ...)
```
- 임베딩 레이어는 `y = wx + b` 연산이 아니라 **번호에 해당하는 벡터를 꺼내오는 룩업 테이블(메모리 공간)**
- 하지만 이 값들도 **학습 가능한 가중치**라서 역전파로 갱신되고 `summary()`의 파라미터에 포함된다
- 차원을 100으로 키우면 → 30 × 100 = 3,000개

### 코드 읽는 요령
- `Embedding(30, 100)`처럼 쓰인 코드를 보면 → **앞 = 단어 사전 개수, 뒤 = 임베딩 차원** (출력 개수라고 헷갈리기 쉬움)
- 입력 데이터의 `(15, 5)`와 `Embedding`의 숫자는 **연관이 없다** (앞에 있는 건 입력 길이/개수가 아님)

---

## 7. 랭체인(LangChain) OpenAIEmbeddings ⭐ (오늘의 핵심)

> 실무에서는 위의 과정을 **직접 구현하지 않는다.** 임베딩 모델을 import해서 호출하면 끝.
> 포인트는 **"우리 데이터(문서, PDF 등)를 이 모델로 벡터화 → 벡터 DB에 저장"** 하는 흐름.

### 준비
```bash
pip install langchain-openai python-dotenv
```

`.env` 파일 (1일차에 배운 API 키 관리 방식 그대로)
```
OPENAI_API_KEY=sk-...
```

### 기본 사용법
```python
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

load_dotenv()   # .env의 API 키 로드

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# 1) 문장 하나 → 벡터 하나
vector = embeddings.embed_query("나는 밥을 먹었다")
print(len(vector))      # 차원 수 확인 (text-embedding-3-small 기본 1536)
print(vector[:5])       # 앞의 몇 개 값만 확인

# 2) 여러 문서 → 벡터 리스트
vectors = embeddings.embed_documents(["첫 번째 문서", "두 번째 문서"])
print(len(vectors), len(vectors[0]))   # (문서 수, 차원)
```

### 정리
| 항목 | 내용 |
|---|---|
| import | `from langchain_openai import OpenAIEmbeddings` (챗 모델 `ChatOpenAI` 대신 임베딩 모델을 import) |
| 모델 | `text-embedding-3-small` 사용 (강의: "그 이상 쓸 일은 없다") |
| `embed_query(text)` | **검색 질문 1개**를 벡터화 → 벡터 1개 |
| `embed_documents(list)` | **문서 여러 개**를 한 번에 벡터화 → 벡터 리스트 |
| 차원 | `len(vector)`로 확인. 강의에서 1,600이라 했다가 **1,500대로 정정** (공식 기본값은 1536) |
| 차원 조절 | `OpenAIEmbeddings(model=..., dimensions=256)`처럼 `dimensions`로 줄일 수 있음 (3-small/3-large 지원) |
| 비용 | 챗 모델처럼 **임베딩도 시간과 돈이 든다** (토큰 기준 과금) |

### 꼭 기억할 점
- **문서 하나(문장 하나) → 벡터 하나**가 나온다. (토큰마다 벡터가 따로 나오는 게 아님. 내부적으로는 토큰 단위로 처리해서 하나로 합쳐 줌)
- 내 이름이든 "코파일럿"이든 "밥을 먹었다"든 **입력 길이와 상관없이 항상 같은 차원(예: 1536개)** 의 벡터가 나온다
- 임베딩 모델마다 차원이 다르다. **임베딩할 때와 검색할 때는 반드시 같은 모델**을 써야 한다
- 나중에 RAG에서는: `문서 → embed → 벡터 DB 저장` / `질문 → embed_query → 유사한 벡터 검색`

---

## 8. 벡터 유사도 — 코사인 유사도 vs 유클리드 거리

벡터로 바뀐 텍스트끼리 **"얼마나 비슷한가"** 를 수치로 비교하는 방법.

### 코사인 유사도 (Cosine Similarity)
- **두 벡터 사이의 각도**(방향)로 유사도를 판단
- 범위 **-1 ~ 1**, **1에 가까울수록 비슷**
- 예: `왕` 과 `여왕`은 방향이 비슷 → 유사도 높음 (약 0.99 같은 값)

$$
\cos(\theta) = \frac{A \cdot B}{\|A\|\,\|B\|}
$$

```python
import numpy as np

def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

v1 = embeddings.embed_query("왕")
v2 = embeddings.embed_query("여왕")
v3 = embeddings.embed_query("국밥")

print(cosine_similarity(v1, v2))   # 높음
print(cosine_similarity(v1, v3))   # 상대적으로 낮음
```

### 유클리드 거리 (Euclidean Distance)
- 두 점 사이의 **직선 거리**. **작을수록 비슷**
- 코사인은 "각도", 유클리드는 "거리" → **개념이 다르다**

### 면접 대비 ⭐
- "코사인 유사도가 뭔가요?" → **"두 벡터가 이루는 각도(방향)를 이용해 유사도를 구하는 방법. 1에 가까울수록 의미가 비슷하다"** 정도로 간단히 설명할 수 있으면 충분
- 직접 계산식을 풀 일은 없다 (라이브러리/벡터 DB가 알아서 계산)
- 실무에서는 **코사인 유사도를 가장 많이 사용**. 벡터 DB 설정에서 `cosine / euclidean / dot product` 중 선택하는 정도

---

## 9. 오늘의 핵심 요약 (한눈에)

1. 컴퓨터는 글을 이해하지 못한다 → **수치화 → 연산 → 결과를 글자로 표시**
2. 토큰 = 음절~형태소 크기의 말 덩어리 (모델마다 다름, 비용 기준)
3. `Tokenizer`(`fit_on_texts` → `texts_to_sequences`) → `pad_sequences`로 길이 통일
4. 원핫은 **차원 폭발 + 0투성이**라 비효율 → **임베딩**으로 해결
5. `Embedding(input_dim=단어사전 개수(+1), output_dim=벡터 차원)` → 출력 shape `(N, 길이, 차원)`
6. 파라미터 수 = `input_dim × output_dim`
7. 실무: `OpenAIEmbeddings(model="text-embedding-3-small")` + `embed_query` / `embed_documents`
8. 벡터 간 유사도는 **코사인 유사도(각도)** 를 많이 사용
9. **다음 시간**: 벡터 스토어 / 벡터 DB에 저장하고 검색하기 (RAG로 연결)

---

## 10. 복습 체크리스트

- [ ] 토큰과 어절/형태소/음절의 차이를 설명할 수 있다
- [ ] `word_index`의 번호가 어떤 기준(빈도순)으로 매겨지는지 안다
- [ ] 패딩 `pre/post`, 자르기 `truncating`의 차이를 안다
- [ ] 원핫 인코딩이 왜 비효율적인지 말할 수 있다
- [ ] 임베딩 레이어 입력/출력 shape 변화를 말할 수 있다 `(15,5) → (15,5,10)`
- [ ] 임베딩 레이어 파라미터 수를 계산할 수 있다
- [ ] `embed_query` / `embed_documents` 차이를 안다
- [ ] 코사인 유사도를 한 문장으로 설명할 수 있다

> ※ 녹취 특성상 일부 용어(예: 차원 수, 파라미터 숫자)가 불명확해 보강해서 정리했습니다. 실습 코드는 환경(Keras 버전, langchain-openai 버전)에 따라 일부 인자가 다를 수 있으니 직접 실행해 확인하세요.
