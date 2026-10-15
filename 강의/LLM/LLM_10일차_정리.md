# LLM 10일차 정리

> 녹취(STT) 기반 정리입니다. 인식 오류가 많은 구간(예: 앙상블 → "인쌃불", 파일 번호 등)은 문맥으로 보정했고, 코드는 수업 흐름에 맞춰 **재구성한 예시**입니다. 실행 전 본인 노트북 코드와 대조하세요.

## 오늘의 핵심 한 줄

- **BM25(키워드 검색)** 리트리버를 배우고, **벡터 유사도 검색**과 묶어 **하이브리드(앙상블) 리트리버**를 완성했다.
- 오전에는 전이학습 마무리 설명, 오후에는 팀 프로젝트 기획 발표와 피드백이 있었다.

---

## 1. 딥러닝 — 전이학습 마무리

### 1-1. 전이학습 모델을 내 문제에 적용하려면
- **바꿔야 하는 곳은 위·아래 두 곳뿐**이다.
  - 입력부: 내 데이터에 맞게 `input_shape`(데이터 모양) 변경
  - 출력부: 내 **타겟**(클래스 수, 활성화 함수)에 맞게 출력층 변경
- 같은 모델도 타겟만 바꾸면 쓰임새가 달라진다. (반 친구 얼굴 찾기 → 엑스레이 사진에서 이상 유무 판단)
- 가중치(weights)를 그대로 가져오면서 위·아래도 안 바꾸고 제출하면 **같은 대회에서 똑같은 결과라 같이 탈락**한다.
- "가중치 가져온다 / 동결한다"가 무슨 뜻인지 이제 구분할 수 있어야 한다. (8일차 `trainable` 복습)

### 1-2. 잘라낸 모델 뒤에 레이어를 더 붙여도 되나?
- Flatten / Fully Connected 이후에 Conv를 붙이는 것도 **가능은 하다.**
- 안 하는 이유는 **확신이 안 서서**일 뿐이다.
- 은닉층(hidden layer) 뒤에 뭔가를 연결했을 때 성능이 좋아지는지 나빠지는지는 **아직 학계도 명확히 모른다.**
- 그래서 연구 방식은 이렇다.
  1. 연결해 본다
  2. 성능을 확인한다
  3. 재현했을 때도 같은 결과가 나오면 → 논문으로 등재
- 실제 사례: 모델 위에 CNN을 붙였더니 성능이 좋아졌다, 앙상블을 덮었더니 좋아졌다 → "좋은 걸 발견했다"가 논문의 결과.

### 1-2-1. Fully Connected 뒤에 Conv 붙이는 방법 (간략)

**문제:** `Dense`의 출력은 2차원 `(배치, 노드수)`인데, `Conv2D`는 4차원 `(배치, 높이, 너비, 채널)`, `Conv1D`는 3차원 `(배치, 길이, 채널)`을 입력으로 받는다. 그래서 **중간에 `Reshape`로 모양을 되돌려야** 붙일 수 있다.

**방법:** `Dense` → `Reshape` → `Conv` 순서로 연결한다. 단, `Reshape` 전후의 **원소 개수가 같아야** 한다. (예: `Dense(1024)` → `Reshape((32, 32, 1))` : 1024 = 32×32×1)

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Flatten, Dense, Reshape, Conv2D,
                                     MaxPooling2D, GlobalAveragePooling2D)

model = Sequential([
    base_model,                       # 전이학습 모델 (예: VGG16, include_top=False)
    Flatten(),                        # (N, 특성수)
    Dense(1024, activation='relu'),   # (N, 1024)        ← Fully Connected
    Reshape((32, 32, 1)),             # (N, 32, 32, 1)   ← 1024 = 32*32*1
    Conv2D(32, (3, 3), padding='same', activation='relu'),
    MaxPooling2D(),
    GlobalAveragePooling2D(),
    Dense(10, activation='softmax'),  # 내 타겟에 맞는 출력층
])
```

- 1차원 형태로 붙이고 싶으면 `Reshape((1024, 1))` → `Conv1D(32, 3, activation='relu')`를 쓴다.
- 함수형 API도 같은 원리다: `x = Dense(1024)(x)` → `x = Reshape((32, 32, 1))(x)` → `x = Conv2D(...)(x)`.
- `model.summary()`로 각 층의 출력 shape이 맞는지 확인한다.

**주의:** `Dense` 출력의 노드 순서에는 원래 "이웃" 관계(공간 구조)가 없다. 그래서 `Reshape`로 2차원처럼 만들어도 Conv가 보는 인접 관계가 **임의적**이다. 기술적으로는 붙일 수 있지만 성능이 좋아진다는 보장은 없고, **붙여 보고 성능을 비교해서** 판단해야 한다. (수업에서 "가능하지만 확신이 안 서서 안 한다"고 한 이유)

### 1-3. 모델 아키텍처 이야기
- 우리가 돌려 본 모델 중 **DenseNet(밀집 연결)** 이 가장 좋았다.
- 성공한 모델 중에는 **대충 만든 것처럼 보이는 구조**도 많다. 일단 해보고 잘 되면, 논문용으로 이유를 붙인다. (예: 역전파 때 기울기 소실이 줄었다 등)
- 가능한 조합을 컴퓨팅 파워로 다 돌려 본 경우도 많다.
- YOLO도 버전이 계속 올라간다. 11 버전부터는 Transformer 요소가 섞였다.
- **결론: 많이 보고 많이 겪어봐야 한다.**

---

## 2. LangChain — BM25 리트리버

### 2-1. BM25란?
- 지금까지: 문서를 임베딩 → 벡터DB → **유사도 검색(의미 기반)**
- 오늘: **BM25 = 키워드 기반 검색.** 단어(어절) 형태를 그대로 비교해서 **얼마나 많이 겹쳐 나왔는지**로 관련도를 계산한다. (TF-IDF 계열)
- 질문이 "삼성전자의 반도체 사업"이면 → 그 단어들이 많이 겹치는 문서가 상위로 올라온다.
- `BM25`, `K7`, `K9`처럼 숫자·영문 코드가 들어간 고유명사는 키워드 방식이 의미 기반보다 잘 잡는 경우가 있다.
- 키워드 방식이므로 **청크 단위가 아니라 문장 단위 텍스트**를 넣어도 된다.

### 2-2. 벡터 유사도 검색과 비교

| 구분 | 벡터 유사도 검색 | BM25 검색 |
|---|---|---|
| 기준 | 의미(임베딩 거리) | 단어 겹침(빈도) |
| 임베딩 | 필요 (API 비용 발생) | **불필요** |
| 속도 | 계산량이 많아 상대적으로 느림 | 빈도 계산이라 상대적으로 빠름 |
| 약점 | 고유명사·코드 매칭이 약할 수 있음 | 동의어·의미 유사성 못 잡음 |
| 실무 | 주력 | **보조**, 단독 사용은 드묾 |

### 2-3. 설치와 기본 사용
```bash
pip install rank_bm25
```

```python
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever

texts = [
    "삼성전자는 반도체 사업에서 메모리 중심의 ...",
    "...",
]
docs = [Document(page_content=t) for t in texts]

bm25_retriever = BM25Retriever.from_documents(docs)
bm25_retriever.k = 3            # 상위 3개 반환

result = bm25_retriever.invoke("삼성전자의 반도체 사업")
for r in result:
    print(r.page_content)
    print(r.metadata)
```

- `k=3`이면 결과는 **Document 3개짜리 리스트**다.
- `page_content`, `metadata`(페이지 번호 등) 같은 용어가 LangChain에서 거의 표준으로 통일되고 있다.
- 리트리버는 어떤 종류든 **`invoke(질문)` 로 부르는 방식이 통일**되어 있다. (예전엔 메서드 이름이 제각각이었음)

---

## 3. 벡터 검색 실습 — 기존 DB와 섞이는 문제

수업 흐름: BM25 코드(23-1)를 복사해서 벡터 검색 버전(23-2)을 만든다.

### 3-1. 필요한 코드만 가져오기
- 앞선 파일(22번대)에서 가져올 부분: `import os`, `load_dotenv`, 임베딩, 벡터스토어
- 필요 없는 부분: 주석 처리돼 있던 코드, 불필요한 import

### 3-2. 문서를 DB에 저장하고 검색
```python
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()
embeddings = OpenAIEmbeddings()

# 문서를 DB에 "저장"
vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory="./chroma_db",
    collection_name="my_collection",
)

vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
vector_docs = vector_retriever.invoke("삼성전자의 반도체 사업")
```

### 3-3. 실습에서 나온 포인트
- 임베딩만 끌어온다고 끝이 아니다. **DB에 문서를 넣는 단계(`from_documents`)가 있어야** 검색이 된다.
- 기존에 쓰던 DB(persist_directory)에 그대로 연결해서 넣었더니 **예전에 저장해 둔 문서까지 같이 검색 결과에 섞여 나왔다.**
  - BM25는 방금 넣은 텍스트만 대상 → 결과가 깔끔
  - 벡터DB는 기존 데이터까지 포함 → 결과가 더 많이/다르게 나옴
  - 큰 문제는 아니지만, **해당 문서에서만 답하게 하려면 DB 경로나 collection 이름을 바꿔야 한다.**
- 같은 `k=3`이어도 두 검색의 성격은 다르다. (키워드 겹침 vs 의미 유사도)

---

## 4. 하이브리드 검색 — EnsembleRetriever

### 4-1. 순서
1. 앞의 두 파일을 합친 새 파일을 만든다. (24-4번 근처)
2. 이미 저장된 DB는 **다시 저장하지 않고 불러온다.** (`from_documents`가 아니라 기존 컬렉션 로드)
3. BM25 리트리버와 벡터 리트리버, 2개를 준비한다.
4. `EnsembleRetriever`로 묶는다.

### 4-2. 코드
```python
from langchain_chroma import Chroma
from langchain_classic.retrievers import EnsembleRetriever
# LangChain 1.0 이전: from langchain.retrievers import EnsembleRetriever

# 저장된 DB 불러오기 (저장 X)
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings,
    collection_name="my_collection",
)
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

hybrid_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.5, 0.5],      # 검색 중요도(가중치)
)

hybrid_docs = hybrid_retriever.invoke("삼성전자의 반도체 사업")
```

### 4-3. 핵심 개념
- **`weights`** 가 각 검색의 중요도다. 합이 1이 되게 주는 것이 일반적이다.
  - 일단 반반(0.5 / 0.5)으로 시작
  - 벡터를 더 중시하면 `[0.1, 0.9]`, BM25를 더 중시하면 `[0.9, 0.1]` → 결과를 비교해서 판단
- 좋은지 나쁜지는 결국 **사람이 결과를 보고 판단**해야 한다.
- **앙상블 vs 하이브리드 용어**
  - **앙상블(Ensemble)** = 여러 리트리버를 묶는 큰 개념 (비슷한 성격끼리 묶어도 앙상블)
  - **하이브리드(Hybrid)** = 앙상블 중에서도 **성격이 다른 검색**(키워드 + 의미)을 묶은 경우
- 변수 이름도 `vector_docs` → `hybrid_docs` 처럼 구분해서 쓴다.

### 4-4. 장단점 / 실무 메모
- 장점: 키워드와 의미 검색의 약점을 서로 보완한다. 실무에서 많이 쓴다.
- 단점: 검색이 두 번 도니 속도가 느려질 수 있다. (요즘은 컴퓨팅 성능이 좋아 체감이 적음)
- 모델 성능이 워낙 좋아져서 "검색을 따로 지정하지 않으면 검색을 건너뛰는" 경우도 있다. 그래도 회사 코드에 이런 구조가 있으니 알아야 한다.

### 4-5. LangChain 버전 주의
- LangChain **1.0부터** 예전(0.x) 시절 클래스들이 **`langchain_classic`** 으로 옮겨졌다.
- 예전 책·블로그 코드를 따라 칠 때 import 에러가 나면 `langchain_classic`을 의심한다.
- 명령어/구조가 많이 바뀌었으므로 옛 자료는 조심해서 본다.

---

## 5. 용어 정리

| 용어 | 뜻 |
|---|---|
| BM25 | 단어 겹침 빈도 기반 키워드 검색 알고리즘 (TF-IDF 계열) |
| Retriever | 질문을 받아 관련 문서를 뽑아오는 객체. `invoke()`로 호출 |
| `k` | 가져올 문서 개수 |
| `page_content` / `metadata` | Document의 본문 / 부가정보(페이지 번호 등) |
| 유사도 검색 | 임베딩 벡터 간 거리로 의미가 가까운 문서 검색 |
| EnsembleRetriever | 여러 리트리버 결과를 가중치로 합치는 리트리버 |
| 하이브리드 검색 | 성격이 다른 검색(키워드 + 벡터)을 묶은 앙상블 |
| `weights` | 앙상블에서 각 리트리버의 비중 |
| `persist_directory` / `collection_name` | 벡터DB 저장 위치 / 컬렉션 이름 |
| `langchain_classic` | LangChain 1.0 이후 구버전 기능이 이동한 패키지 |

---

## 6. 복습 체크리스트

- [ ] BM25와 벡터 유사도 검색의 차이를 한 문장으로 설명할 수 있는가?
- [ ] `rank_bm25` 설치 후 `BM25Retriever`로 k=3 검색을 직접 돌려봤는가?
- [ ] `Chroma.from_documents`(저장)와 `Chroma(...)`(불러오기)의 차이를 구분하는가?
- [ ] 기존 DB에 연결하면 왜 결과가 섞이는지 설명할 수 있는가?
- [ ] `weights`를 `[0.5,0.5]`, `[0.9,0.1]`, `[0.1,0.9]`로 바꿔 결과를 비교해 봤는가?
- [ ] 앙상블과 하이브리드 용어를 구분하는가?
- [ ] 전이학습에서 바꿔야 하는 두 곳(입력 shape, 출력 타겟)을 말할 수 있는가?

---

## 특별 탭 — 팀 프로젝트 기획 발표 & 피드백

오후 발표(고용24 에이전트, 여행 추천, 워크플로우 개선, 지방의회 회의록 이슈 트래커, 보도 장애물 탐지, 컴파일러 오픈소스 기여)에서 나온 **기술 관점 피드백**만 추렸다.

### 공통 피드백
- **스펙 채우기 경계**: 머신러닝/딥러닝을 "스펙을 늘리려고" 넣지 말고, 문제에 필요한 기술을 선택한다. 효율(비용·속도)도 선택 기준이다.
- **개인정보/보안**: 외부 API(GPT 등)로 개인정보가 넘어가는 문제는 심사에서 반드시 지적될 수 있다. → **대책을 발표에 꼭 포함** (마스킹, 로컬 모델 등).
- **범위**: 2주 안에 두 개를 동시에 하기보다 **하나를 깊게** 만드는 편이 포트폴리오에 낫다. 다음 주쯤 체력이 떨어진다.
- **MVP**: 작게 시작하되, 어디까지 구현했는지 최종 발표에서 명확히 말한다.

### 주제별 요점
- **추천 시스템(여행)**: 사용자 데이터가 없는 **콜드 스타트** 문제가 핵심. 온보딩 설문, 부족한 정보를 되묻는 루프(→ 에이전트)를 고려. 경로 설계까지 가려면 그래프 계열 모델이 필요.
- **지방의회 회의록**: 핵심은 **임베딩 + RAG 검색**. 로컬 모델은 GPU(4060 8GB)가 부족하니 클라우드·외부 장비 확보 필요. 청킹 1,400~1,800자, 로지스틱 회귀 / LightGBM 예측 비교 계획.
- **보도 장애물 탐지(CV)**: 목적이 "길이 막혔는가"라면 모든 객체를 학습하기보다 **세그멘테이션으로 길 영역(ROI)을 잡고 침범/단절 여부**만 판단하는 방법도 있다. 실시간성 확보와 야간·우천 대응이 숙제. 프레임워크는 PyTorch/TensorFlow 제약 없음.
- **에이전트 워크플로우 개선**: AI 엔지니어링 역량으로 어필하려면 프롬프트·도구 활용을 넘어 직접 구성·구현하는 부분을 보여줘야 한다는 지적.

---

## 운영 공지 메모

- 발표 자료는 **PPT + PDF를 같이 제출**, **타임테이블 필수**. (PPT가 깨질 경우 대비)
- 발표는 **대본 없이** 진행. 팀원 발표도 직접 확인하기 전에는 믿지 말 것.
- 프로젝트는 다음 날부터 시작. 과정 중 코드를 대신 짜주지는 않고, **방향성·접근 방법 위주로 질문**하면 된다.
- 시험 안내가 있었음 (문항 수 배분은 녹취가 불분명해 확인 필요).
