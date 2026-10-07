# LLM 5일차 정리

> 주제: **벡터 DB(Chroma · FAISS) → 모델 연결 → RAG 체인 → Gradio 챗봇**
> 실습 파일 흐름: 12(Chroma 저장) → 13(모델 연결) → 14(RAG 체인) → 15(Gradio) → 16(프롬프트 제한 실험) → 17(FAISS) → 18(FAISS + Gradio 채점 과제)
>
> ※ 녹취(음성 인식) 품질이 낮아 일부 코드는 맥락으로 복원했습니다. 함수·파라미터 이름은 실행 전에 한 번 확인하세요.

---

## 1. 벡터와 벡터 DB 개념

| 용어 | 설명 |
|---|---|
| **벡터(Vector)** | 청크(문장 조각)를 임베딩하면 나오는 **숫자 1줄**. 스칼라 값이 차원 수만큼 들어 있음 |
| **차원(Dimension)** | 수업에서 쓴 OpenAI 임베딩은 **1536차원** = 스칼라 1536개 (모델마다 다름) |
| **벡터 스토어** | 임베딩된 벡터들을 모아 놓은 저장소 |
| **벡터 DB** | 벡터 스토어를 저장·검색할 수 있게 만든 DB (Chroma, FAISS 등) |

- 여기서 말하는 차원은 행렬·텐서의 2차원, 3차원 같은 **축의 개수가 아님**. 벡터 하나가 가진 **값의 개수**를 뜻함.
- 임베딩 모델이 달라지면 차원 수도 달라지므로, 저장할 때와 불러올 때 **같은 임베딩 모델**을 써야 함.

### 벡터 DB 저장 파이프라인 (외우기)

```
데이터 로드 → 청킹(자르기) → 임베딩 → DB에 저장
```

- 불러올 때는 **로드만** 하면 끝 (다시 청킹·임베딩 할 필요 없음).

---

## 2. Chroma DB 저장 / 불러오기 (12번 실습)

### 저장할 때 (처음 한 번만)

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
docs = splitter.split_documents(data)          # 청킹

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

db = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory="./chroma_db",           # 저장 경로
    collection_name="chroma_12",               # 컬렉션 이름
)
```

### 불러올 때 (저장 코드는 주석 처리 / 삭제)

```python
db = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings,             # 저장 때와 같은 임베딩
    collection_name="chroma_12",
)
```

**포인트**
- 로드에 필요한 것 3가지: **경로 · 임베딩 함수 · 컬렉션 이름**.
- 저장 코드를 지우지 않고 다시 실행하면 **같은 문서가 중복 저장**됨 → 불러오기 전용 파일에서는 반드시 주석 처리/삭제.
- `Document` 객체는 `page_content`(본문)와 `metadata`로 구성됨.

---

## 3. LLM 모델 연결 (13번 실습)

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="모델명",
    temperature=0,
    max_tokens=300,
)

response = llm.invoke("질문")
print(response.content)
```

| 파라미터 | 의미 |
|---|---|
| `temperature` | 0에 가까울수록 **있는 그대로·일관된 답변**, 1에 가까울수록 창의적(딴소리 가능). 강의처럼 **정보 전달용이면 0** |
| `max_tokens` | 출력 토큰 상한. 실습 중에는 **비용 때문에 제한**을 걸어 둠 (100~300 정도). 답이 잘리면 늘리면 됨 |

**주의할 점**
- 모델마다 **지원하지 않는 파라미터**가 있을 수 있음(예: temperature 고정 모델) → 에러가 나면 파라미터를 지우고 실행해 보기.
- 비용이 부담되면 **가격이 저렴한 소형 모델**을 사용 (수업에서도 비싼 모델 대신 저렴한 모델로 변경).
- `invoke` = **추론(inference)**. 이미 학습된 가중치로 답만 내는 **순전파**이고, 역전파·가중치 갱신(컴파일·훈련)은 없음. 딥러닝 수업의 `predict`와 같은 개념.
- 모델만 연결하면 **일반 지식으로 친절하게 답변**하고, 벡터 DB만 쓰면 **DB 안의 내용으로만** 답변함 → 둘을 합친 것이 RAG.

### 프롬프트에 변수 넣기 (f-string)

```python
query = "삼성전자의 창업자는 누구인가요?"
prompt = f"다음 질문에 답하세요: {query}"
```

- `f"..."` 안의 `{}` 에 변수 값이 문자열로 들어감.
- **대화 메모리**를 쓰면 이전 질문·답변이 매번 프롬프트에 함께 전달됨 → 대화가 길어질수록 **토큰 사용량이 계속 증가**. (토큰 관리 필요)

---

## 4. RAG 체인 (14번 실습) ⭐ 가장 중요

### 4-1. 구성 요소

```
질문(input) ─▶ [Retriever: 벡터DB 검색] ─▶ context ─┐
                                                    ├─▶ [Prompt] ─▶ [LLM] ─▶ answer
질문(input) ────────────────────────────────────────┘
```

| 구성 | 역할 |
|---|---|
| **Retriever** | 질문과 유사한 청크를 **벡터 DB에서** 검색 (웹 검색이 아님!) |
| **Prompt** | `{input}`(질문)과 `{context}`(검색 결과) 자리가 있는 템플릿 |
| **LLM** | 완성된 프롬프트를 받아 최종 답 생성 |
| **create_stuff_documents_chain** | **모델 + 프롬프트**를 연결 (검색된 문서를 프롬프트에 "채워 넣는" 체인) |
| **create_retrieval_chain** | **검색기 + 위 체인**을 연결해 RAG 완성 |

> 두 함수는 **항상 한 세트**로 같이 쓴다고 기억하기.

### 4-2. 코드

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# 1) 검색기: 유사한 청크 2개를 가져옴
retriever = db.as_retriever(search_kwargs={"k": 2})

# 2) 프롬프트: {context}, {input} 자리 필수
prompt = ChatPromptTemplate.from_template("""
당신은 주어진 context만 근거로 답변하는 도우미입니다.
context에 정보가 없으면 "주어진 정보로는 답할 수 없습니다."라고 답하세요.

[context]
{context}

[질문]
{input}
""")

# 3) 체인 구성
doc_chain = create_stuff_documents_chain(llm, prompt)          # 모델 + 프롬프트
rag_chain = create_retrieval_chain(retriever, doc_chain)       # 검색 + 위 체인

# 4) 실행: input만 직접 넣는다
result = rag_chain.invoke({"input": "삼성전자의 창업자는 누구인가요?"})

print(result["answer"])    # 최종 답변만 보고 싶을 때
```

### 4-3. 결과 구조 (dict)

| 키 | 내용 |
|---|---|
| `input` | 내가 넣은 질문 |
| `context` | 검색된 청크 리스트(`Document` 객체, 개수 = `k`) |
| `answer` | 최종 답변 |

- `result["context"][0].page_content` 처럼 **하나만 꺼내서** 확인할 수 있음.
- **`context`는 직접 넣지 않아도** 검색기가 자동으로 채워 줌 → 그래서 `invoke`에는 `input`만 넣으면 됨.
- 헷갈릴 땐 "프롬프트에 `{input}`과 `{context}` 두 개를 두고, input은 내가, context는 알아서 들어간다"로 기억.

### 4-4. 왜 "답할 수 없습니다"가 나왔나?

- 프롬프트에 **"정보가 없으면 모른다고 답해라"** 라고 제한을 걸었기 때문.
- 벡터 DB에 삼성전자 창업자 정보가 없으면, 검색은 되더라도 **관련 없는 청크**가 context로 들어가서 위 문구가 출력됨. → **정상 동작**.
- 유사도 검색은 관련 없는 질문이어도 **항상 "그나마 가장 비슷한" k개를 반환**함. (DB에 52개 청크가 있으면 52개와 비교해서 가까운 순으로 뽑음)
- 제한이 약하면 모델이 **DB 밖의 일반 지식으로 대답**해 버리기도 함 → 이럴 땐 프롬프트의 제약을 더 **강하게** 쓰기. (예: "절대 context 밖의 지식을 사용하지 말 것")

### 4-5. 속도 팁

- 실습 코드에 **테스트용 `invoke` / 검색 / `print`를 여기저기 남겨 두면** 실행할 때마다 API 호출·검색이 반복돼서 느려짐.
- 코드를 다 만든 뒤 **불필요한 `invoke`, 중복 검색, 유사도 검색 테스트 코드는 지우거나 주석 처리**.

---

## 5. Gradio로 챗봇 만들기 (15번 실습)

```python
import gradio as gr

def chat(message, history):
    result = rag_chain.invoke({"input": message})
    return result["answer"]

demo = gr.ChatInterface(
    fn=chat,
    title="나의 RAG 챗봇",
)

demo.launch()                 # 로컬 실행
# demo.launch(share=True)     # 외부에서 접속 가능한 임시 public URL 생성
```

- 함수는 **`(message, history)`** 를 받아 **답변 문자열**을 반환 → 앞에서 만든 RAG 체인 호출 코드를 그대로 함수에 넣으면 됨.
- `gr.ChatInterface(fn=함수, title=...)` 로 채팅 UI가 바로 생성됨.
- 실행하면 `http://127.0.0.1:7860` 같은 주소가 출력됨.
  - `127.0.0.1` = **내 컴퓨터 자신**(localhost). 외부에서 접속하는 게 아니므로 해킹 걱정 X.
  - 다른 사람이 접속해야 하면 `share=True` 사용.
- `import gradio as gr` 에 노란 줄이 뜨면 **설치 필요** → `pip install gradio`.
- 질문이 DB에 없는 내용이면 "답할 수 없습니다"가 나오므로, **DB에 있는 내용 위주로 질문**해서 테스트.
- 실제 서비스에서는 프롬프트·검색 방식을 고쳐 가며 성능을 개선하는 과정이 필요.

---

## 6. FAISS 벡터 DB (17번 실습)

### 6-1. 설치

```bash
pip install faiss-cpu
```

- FAISS는 **CPU / GPU 버전이 따로** 있음. 수업 환경(윈도우)은 GPU 버전을 지원하지 않고, 데이터 규모도 작아서 **CPU 버전**으로 충분.
- LangChain 쪽에서도 FAISS를 지원함 (`langchain_community.vectorstores.FAISS`).
- 대문자 `FAISS` = **LangChain이 제공하는 래퍼 클래스**, 소문자 `faiss` = **Meta(Facebook)의 원본 라이브러리**.

### 6-2. 거리 vs 유사도 (Chroma와의 차이)

| | Chroma(기본) | FAISS(`IndexFlatL2`) |
|---|---|---|
| 기준 | **코사인 유사도** — 두 벡터의 **각도** | **유클리드 거리(L2)** — 두 벡터 사이의 **직선 거리** |
| 값의 의미 | 클수록(각도가 작을수록) 비슷 | **작을수록** 비슷 |

**유클리드 거리 계산**

```
점 (1, 2), (4, 6) 이라면
거리 = √[(4-1)² + (6-2)²] = √(9 + 16) = √25 = 5
```

- 각 차원의 차이를 **빼고 → 제곱하고 → 모두 더한 뒤 → 제곱근**.
- 임베딩은 1536차원이므로 **1536개 차원 전체**에 대해 같은 계산을 하는 것.

### 6-3. 코드 (저장)

```python
import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# 인덱스: "내가 쓸 벡터의 차원 수"를 알려줘야 함
dim = len(embeddings.embed_query("hello world"))        # → 1536
index = faiss.IndexFlatL2(dim)

db = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=InMemoryDocstore(),       # 메모리 기반 문서 저장소
    index_to_docstore_id={},           # 인덱스 번호 ↔ 문서 ID 매핑 (처음엔 빈 dict)
)

print(db.index.ntotal)                 # 저장된 벡터 개수 → 처음엔 0

db.add_documents(docs)                 # 청크 추가
print(db.index.ntotal)                 # 추가 후 개수 확인

db.save_local(folder_path="./faiss_db", index_name="faiss_17")
```

- `embed_query("아무 문장")` 의 길이 = **임베딩 차원 수** (1536) → 인덱스 생성에 필요.
- `index.ntotal` : 인덱스에 들어 있는 **벡터 총 개수**. 처음엔 0, 문서를 넣으면 늘어남. (중복 실행하면 개수가 불필요하게 늘어나니 주의)
- 저장하면 폴더에 **두 개 파일**이 생김
  - `faiss_17.faiss` : 벡터 인덱스
  - `faiss_17.pkl` : **피클(pickle)** 파일 — 문서/메타데이터 등 파이썬 객체 저장용 (전통 머신러닝 모델·NumPy·Pandas 데이터 저장에도 많이 쓰임)

### 6-4. 코드 (불러오기)

```python
db = FAISS.load_local(
    folder_path="./faiss_db",
    embeddings=embeddings,
    index_name="faiss_17",
    allow_dangerous_deserialization=True,   # 피클 로드 허용 (최신 버전에서 필요)
)
```

- 불러올 때도 **저장 때와 같은 임베딩 모델 / 같은 `index_name`**.
- Chroma와 비교: 로드 파라미터가 `persist_directory/collection_name` → `folder_path/index_name` 으로 바뀐 정도의 차이.
- 이후 `db.as_retriever(...)` → RAG 체인 구성은 **Chroma와 동일**.

### 6-5. Chroma vs FAISS 어떤 걸 쓸까?

- **문서 수십~수백 개 수준**이면 속도·결과 **차이 거의 없음**.
- FAISS의 강점은 **대규모(수백만~수십억) 벡터 검색**. 그 정도 규모는 이번 수업에서 실습하기 어려움.
- 결론: **지원자가 가려는 회사가 쓰는 DB**를 따라가면 됨. (Chroma 쓰는 곳이면 Chroma, FAISS 쓰는 곳이면 FAISS)
- 라이브러리마다 버전/문법이 조금씩 달라서 **공식 문서 확인 습관**이 중요 (Keras, XGBoost 등도 마찬가지).

---

## 7. 18번 과제: Chroma/FAISS + Gradio 채점

- 파일명: `LLM_18_faiss_gradio_채점.py` (제목 정도의 이름)
- 목표: 오늘 배운 **벡터 DB(Chroma 또는 FAISS) + RAG 체인 + Gradio** 로 **직접 챗봇을 만들어 보기** (프로덕션 수준까지 가면 더 좋음)
- 방법: 앞 실습 코드를 복사해서 이어 붙이고, 막히면 바로 질문.
- 다음 시간 예고: 문서 **전처리 · 청킹 · 메타데이터 관리** 단계를 더 깊게 다룸.

---

## 8. (복습) 딥러닝 파트 — 입력 차원과 층 연결

수업 초반에 케라스 모델의 **입력 차원 정리**도 함께 했음.

| 데이터 차원 | 예시 | 주로 쓰는 층 |
|---|---|---|
| 2차원 (샘플, 특성) | California 주택, 정형 데이터 | `Dense` (필요 시 `Conv1D`) |
| 3차원 (샘플, 시간/길이, 특성) | 시계열, 임베딩 출력 | `Conv1D`, `LSTM`, `GRU` |
| 4차원 (샘플, 가로, 세로, 채널) | MNIST, Fashion MNIST, CIFAR | `Conv2D` |

- `input_shape`는 **샘플 수를 뺀** 나머지 차원. 예) 4차원 이미지 `(N, 28, 28, 1)` → `input_shape=(28, 28, 1)`.
- **Dense 계열로 넘어가기 전**에는 데이터가 2차원 이상이면 `Flatten` 또는 `Reshape`으로 **2차원으로 변환**해야 함. (출력층 직전)
- **Embedding 다음에 붙이는 정석**: `LSTM`/`GRU` 또는 `Conv1D`. (Conv2D도 억지로 붙일 수는 있지만 성능 보장 X)
- 신경 써야 할 것: **입력 shape**, 은닉층 **유닛 수**, 출력층 **출력 차원(분류면 클래스 수) + 활성화 함수**.
- `Conv1D`는 **2차원 데이터(시계열·정형)** 에도 사용 가능 → 아직 시도하지 않은 조합이니 실습해 볼 것.
  - `MaxPooling1D`, `GlobalAveragePooling1D` 등 풀링도 1D 버전이 있음 (기능은 2D와 동일).
- 5차원 이상 데이터도 이론상 가능하지만 흔하지 않고, 보통 2차원으로 줄여서 처리.
- 같은 2차원 데이터라도 **시계열인지 아닌지** 먼저 판단해서 층 구성을 결정.

---

## 9. 오늘의 핵심 요약 ✅

1. **벡터** = 청크를 임베딩한 숫자 배열(1536차원), **벡터 DB** = 벡터 저장·검색소.
2. 저장: **로드 → 청킹 → 임베딩 → 저장** / 불러오기: **DB 로드만**.
3. **RAG의 R(Retrieval)** = 웹이 아니라 **벡터 DB에서 유사 청크 검색**.
4. `create_stuff_documents_chain`(모델+프롬프트) + `create_retrieval_chain`(검색기+체인) 은 항상 한 세트.
5. 프롬프트에 **`{input}` / `{context}`** 필수. `invoke({"input": 질문})` → 결과는 `input / context / answer`.
6. 모델 호출(`invoke`)은 **추론**일 뿐 학습이 아님. `temperature`, `max_tokens`로 품질·비용 조절.
7. 환각·딴소리 방지 = **프롬프트 제약 강화** + **DB 내용 보강**.
8. **Gradio** `ChatInterface`로 몇 줄만에 챗봇 UI 완성 (`share=True`로 공유).
9. **FAISS** = 유클리드 거리(L2) 기반, `IndexFlatL2(차원)` + `save_local` / `load_local`. 소규모에서는 Chroma와 차이 없음.
10. 필요한 건 개념 이해 — 용어는 어려워 보여도 알고 보면 별거 아님.

---

## 📌 공지 / 기타

- **미니 프로젝트**: 2~3주 정도 진행 예정, 시작 전 **주제 발표 및 컨펌**. 수준이 너무 낮거나(재작업) 너무 높으면(기간 내 불가) 조정될 수 있음.
- 대회 준비용 프로젝트를 미니 프로젝트로 제출하는 것도 가능.
- 프로젝트 기간 중 필요한 **공부는 병행 가능**, 오전 수업·오후 프로젝트 진행.
- 수상 경력은 면접에서 "어떻게 수상했는지"를 물어보는 강력한 자산 → 대회 참여 권장.
- 취업용 프로젝트는 **발표 + 팀원 기여도**가 면접에서 확인될 수 있음.
