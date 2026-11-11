# LLM 20일차 정리

> 주제: **LLM 서빙 성능 측정 (퓨리오사 NPU 환경)** — 동시 요청(스레드), Continuous Batching, KV 캐시 / Prefix 캐시, TPS·RPS·지연시간 측정
> 실습 파일: `F23` ~ `F29` (대부분 앞서 배운 코드의 복습 + 응용)

---

## 0. 오늘의 큰 흐름

| 순서 | 파일 | 내용 |
|---|---|---|
| 1 | (스레드 실습) | `ThreadPoolExecutor`로 동시 요청 → Continuous Batching 개념 이해 |
| 2 | `F23_1`, `F23_2` | Prefix 캐시 적중률 측정 |
| 3 | `F24` | 단일 요청 (토큰 수 · 응답 시간) |
| 4 | `F25` | 순차 요청 (8번 연속) |
| 5 | `F26` | 동시 요청 (스레드, 워커 4개) |
| 6 | `F27` | TPS · RPS 계산 (16개 요청) |
| 7 | `F28` | 워커 수(1·2·4·8)별 성능 비교 ⭐ 오늘 가장 중요 |
| 8 | `F29` | 메트릭(`get_metrics`) 전체 목록 확인 |

---

## 1. 동시 작업: `concurrent.futures.ThreadPoolExecutor`

### 1-1. 기본 개념
- **스레드(Thread)**: 한 프로그램 안에서 여러 작업을 **동시에** 돌리는 실행 단위.
- **워커(worker)**: 스레드 풀 안에서 실제 일을 하는 스레드. `max_workers=4`면 **한 번에 4개**까지만 동시 처리.
- 5개 작업(A~E)을 워커 4개로 돌리면 → A·B·C·D가 먼저 시작, 하나가 끝나면 E가 그 자리에 들어감.
- 끝나는 순서는 **제출 순서와 다를 수 있음** (토큰이 짧은 요청이 먼저 끝남).

### 1-2. 코드 형태

```python
from concurrent.futures import ThreadPoolExecutor
import time

def ask(name, delay):
    time.sleep(delay)          # delay초 동안 쉼
    return f"{name} 완료"

# (1) submit 방식 — 함수와 인자를 넘김
with ThreadPoolExecutor(max_workers=4) as pool:
    futures = [pool.submit(ask, n, d) for n, d in [("A", 3), ("B", 1), ("C", 2)]]
    for f in futures:
        print(f.result())

# (2) map 방식 — 간결, 결과는 이터레이터
with ThreadPoolExecutor(max_workers=4) as pool:
    results = list(pool.map(ask_one, prompts))
```

### 1-3. 핵심 정리
- `submit(함수, 인자...)` → `Future` 객체 반환, `.result()`로 결과 꺼냄.
- `map(함수, 이터러블)` → 결과가 **이터레이터** 형태 → `list()`로 감싸거나 `for`문, `next()`로 접근.
- **`submit`은 `ThreadPoolExecutor`가 제공하는 함수**라서 문법 자체를 암기할 필요는 없음. "함수 + 그 함수에 들어갈 인자"를 받는다는 구조만 이해.

### 1-4. `with` 문 vs 직접 `shutdown`

```python
# with 사용: 블록이 끝나면 자동으로 닫힘
with ThreadPoolExecutor(max_workers=4) as pool:
    ...

# with 없이: 직접 닫아야 함
pool = ThreadPoolExecutor(max_workers=4)
...
pool.shutdown()   # 닫기 (파일의 close()와 같은 역할)
```
- `with` = **열어놓은 걸 깜빡하고 안 닫는 실수를 막아주는 장치**.
- `as pool`은 그 객체에 **별칭**을 붙이는 것.

### 1-5. 동기 vs 비동기, 병렬 처리 (면접 대비 개념)
- 동기/비동기는 주로 **CPU 처리·웹** 쪽에서 자주 나오는 이야기. GPU 위주의 AI 분야에서는 면접에서 자주 묻진 않지만 개념은 알아둘 것.
- 오늘 코드는 스레드 기반 **동시 처리**.

---

## 2. Continuous Batching (연속 배칭)

- 위 스레드 실험에서 본 동작 방식(끝난 자리에 새 요청이 바로 들어감)이 **NPU 서빙 엔진 내부에서 자동으로** 일어나는 것.
- **우리가 직접 설정하거나 건드릴 것은 없음** — NPU에 이미 자동 적용되어 있음.
- 비유: 자율주행차에 타서 자고 있어도 자율주행은 알아서 적용되어 있는 것과 같음.
- 오늘 실습은 "이런 방식으로 돌아간다"를 **눈으로 확인**시키는 것이 목적.

---

## 3. KV 캐시 vs Prefix 캐시

| 구분 | 설명 | 우리가 할 일 |
|---|---|---|
| **KV 캐시** | 이미 계산한 key/value를 저장해 재계산 방지. **사용량(%)** 을 모니터링 | 사용량 확인만 |
| **Prefix 캐시** | 프롬프트의 **앞부분이 동일**하면 그 부분의 계산을 **재사용** | 앞부분을 겹치게 프롬프트 설계 |

### 3-1. Prefix 캐시 포인트
- 적중률(hit rate)이 **높을수록** 재사용이 많이 됨 → 응답 시간이 조금씩 줄어듦.
- 프롬프트가 `[긴 공통 프리픽스] + [달라지는 질문]` 구조일 때 효과가 큼.
- 두 번째 요청부터 캐시 적용 → 첫 번째 요청보다 응답 시간이 짧아져야 정상.
- 질문이 길게 이어지는 대화는 앞의 내용이 모두 프리픽스로 묶이므로, **캐시 사용률을 보고 판단**하되 실무에선 **프롬프트 설계 / 에이전트 쪽에 더 신경 쓰는 게 낫다**는 조언.
- 만약 KV 캐시 사용률이 90%대로 치솟고 Prefix 캐시 적중이 0.1% 수준이면 → 프롬프트 구조를 고민해볼 시점.

### 3-2. 적중 횟수 계산 코드 (F23_2)

```python
before = get_cache_metrics()            # 요청 전 스냅샷
# ... LLM 요청 ...
after  = get_cache_metrics()            # 요청 후 스냅샷

hits    = after["hits"]    - before["hits"]      # 이번 요청에서 적중한 수
queries = after["queries"] - before["queries"]   # 이번 요청에서 조회한 수

if queries > 0:
    hit_rate = int(hits) / int(queries) * 100    # 적중률(%)
else:
    hit_rate = 0
```
- **After − Before** 순서 주의 (반대로 빼면 음수가 나옴).
- `queries`가 0이면 나눗셈 오류 → **0으로 처리**하는 방어 코드.
- `int()`는 **형변환**(동적 변환 아님).
- 루프를 돌 때마다 `hit_rate`는 새로 정의되므로 이전 반복과 무관.

---

## 4. 단일 · 순차 · 동시 요청 비교

### 4-1. 단일 요청 (`F24`)

```python
import time

start = time.time()
response = llm.invoke("프롬프트")
elapsed = time.time() - start

usage = response.usage_metadata
in_tok  = usage.get("input_tokens", 0)    # 없으면 0
out_tok = usage.get("output_tokens", 0)

print(response.content)
```
- `.get("키", 0)`: 키가 없을 때 **기본값 0** 반환 → 빈 딕셔너리가 들어와도 에러 방지.
- `usage_metadata`는 dict 구조 → LLM 다루다 보면 **딕셔너리 / 리스트가 중첩된 구조가 많으므로** 자주 쓰는 것은 손에 익혀둘 것.

### 4-2. 순차 요청 (`F25`)

```python
prompts = [f"{i}번 설명해줘" for i in range(8)]   # 리스트 컴프리헨션

start = time.time()
for p in prompts:
    llm.invoke(p)
elapsed = time.time() - start

print("전체 요청:", len(prompts))
print("전체 시간:", elapsed)
print("RPS:", len(prompts) / elapsed)
```
- `[식 for 변수 in 반복]` : 포문 **앞에 있는 건 결과값** → 한 줄로 리스트 생성.
- `range(8)` = 0~7 (8번 반복).

### 4-3. 동시 요청 (`F26`)

```python
def ask(prompt):
    return llm.invoke(prompt)

start = time.time()
with ThreadPoolExecutor(max_workers=4) as pool:
    results = list(pool.map(ask, prompts))
elapsed = time.time() - start
```
- 순차(25번)와 비교하면 **확연히 빨라짐** (예: 8개 요청이 동시 처리로 약 2.98초).
- 단, **"동시 처리가 무조건 좋다"는 뜻은 아님** — 상황에 맞춰 선택.

---

## 5. 성능 지표 용어

| 용어 | 풀이 | 계산 |
|---|---|---|
| **TPS** | Tokens Per Second (초당 토큰 수) | `총 토큰 수 / 전체 시간` |
| **RPS** | Requests Per Second (초당 요청 처리량) | `요청 수 / 전체 시간` |
| **TTFT** | Time To First Token (첫 토큰까지 걸린 시간) | — |
| **Latency(지연시간)** | 요청 1건의 응답까지 걸린 시간 | — |
| **Throughput(처리량)** | 단위 시간당 처리한 양 (TPS, RPS) | — |

> 수업 중 나온 용어 퀴즈 (TTFT, TPOT, TPS 등): 면접에서 **"그냥 알아들을 수는 있어야" 하는** 용어들.

### 5-1. TPS · RPS 계산 코드 (`F27`)

```python
def ask(prompt):
    resp = llm.invoke(prompt)
    return resp.usage_metadata.get("output_tokens", 0)   # 출력 토큰만 반환

start = time.time()
with ThreadPoolExecutor(max_workers=4) as pool:
    tokens = list(pool.map(ask, prompts))     # 16개 요청
elapsed = time.time() - start

total_tokens = sum(tokens)                    # 리스트 안 값 전부 합산
print("TPS:", round(total_tokens / elapsed, 2))      # 소수 둘째 자리
print("RPS:", round(len(prompts) / elapsed, 2))
```
- `sum(리스트)` → 리스트 안의 모든 숫자를 합산.
- `round(값, 2)` → 소수 **둘째 자리까지** 반올림.
- 요청 개수(`len(prompts)`)를 바꿔가며 실험.

---

## 6. 워커 수별 성능 비교 (`F28`) ⭐ 오늘의 핵심

### 6-1. 목표
워커를 **1 · 2 · 4 · 8개**로 바꿔가며 총 시간 · 지연시간 · RPS · TPS 비교.

### 6-2. 코드 구조

```python
def ask(prompt):
    start = time.time()
    resp = llm.invoke(prompt)
    elapsed = time.time() - start                         # 요청 1건 지연시간
    out_tok = resp.usage_metadata.get("output_tokens", 0)
    return elapsed, out_tok                               # 2개 반환 (튜플)

for workers in [1, 2, 4, 8]:
    start = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(ask, prompts))
    total_time = time.time() - start

    latency   = sum(r[0] for r in results) / len(results)  # 평균 지연시간
    total_tok = sum(r[1] for r in results)

    print(f"Worker {workers}")
    print("전체 시간:", round(total_time, 2))
    print("평균 지연:", round(latency, 2))
    print("RPS:", round(len(prompts) / total_time, 2))
    print("TPS:", round(total_tok / total_time, 2))
    print()
```

### 6-3. 결과 해석
- **워커 8개일 때 확연히 좋아짐** → 동시 작업의 효과.
- 1개 / 2개 / 4개 / 8개 중 무엇이 좋은지는 **결과를 보고 직접 판단**.

### 6-4. ⚠️ 강사가 헷갈렸던 포인트 (실수 포인트)
> "왜 `for`문을 썼지? 그냥 `next()` 쓰면 되잖아?" → **틀린 생각이었음!**

- `results` 안에는 요청 개수(16개)만큼의 값이 들어 있고,
- **각 값이 `(시간, 토큰)` 튜플 2개짜리**임.
- 첫 값 하나만 `next()`로 꺼내면 **나머지 15개 요청 결과를 놓침**.
- → 이런 경우 **모든 요소를 순회**해야 함 (`for` 또는 제너레이터 `sum(...)`).
- 교훈: **데이터 구조(리스트 안의 튜플)를 정확히 파악하고 코드를 작성할 것.**
- 추가 과제: 위 코드를 **더 간결하게 개선**해보기 (오늘 과제).

---

## 7. 토큰(Token) 정의에 대해

- 입력 토큰이 19개로 나온 이유? → 토큰은 **글자 ≠ 단어 ≠ 음절**.
- **BPE(Byte Pair Encoding)** 같은 **인코딩 알고리즘 단위**로 쪼개지므로 사람이 생각하는 기준과 다름.
- 모델마다(Claude, Qwen3, 등) **토크나이저가 달라** 같은 문장도 토큰 수가 다름. **한글은 특히 차이가 큼.**

| 용어 | 뜻 |
|---|---|
| 형태소 | 의미를 가지는 **최소 단위** |
| 어절 | 띄어쓰기로 구분되는 **덩어리** |
| 음절 | **한 글자** |

→ "토큰 ≠ 위 셋 중 어느 것도 아님." 그냥 **모델의 토크나이저가 정한 단위**로 이해.

---

## 8. `get_metrics` 전체 메트릭 확인 (`F29`)

```python
import requests

res = requests.get("http://서버주소/metrics")   # 서버 메트릭 엔드포인트
print(res.text)
```
- 퓨리오사(Furiosa)에서 제공하는 **메트릭 목록 전체**를 출력해 확인.
- 지금까지 주로 쓴 것은 **입력 토큰 / 출력 토큰 / KV 캐시 / Prefix 캐시** 정도.
- 그 외 다른 메트릭들도 있으니 **앞으로 조금씩 친해질 것**.
- `requests`는 웹 요청 라이브러리 (웹 개발 경험자에게 친숙).

---

## 9. 파이썬 문법 · 습관 복습 포인트

- **임포트 위치/오타 주의**: `Future`(t 하나), `concurrent.futures` 등 오타 → 에러 원인.
- **코드 재활용**: 이전 파일을 복사해서 이름만 바꿔 수정 (`F25 → F26 → F27 → F28`).
- **남이 만든 코드**: 고치기보다 **새로 만드는 게 빠른 경우**도 많음 (예: Tensor1 → Tensor2 변환 이야기).
- `print`로 중간값을 찍어보며 **자료구조 확인**하는 습관.
- 오늘 대부분의 코드는 **이미 했던 코드의 반복** → 복습이 안 되어 있으면 어려움.

---

## 10. 수업 중 기타 이야기 (참고)

### 면접 / 커리어 조언
- 질문: **"AI·에이전트·바이브 코딩을 빼고 본인이 할 수 있는 건 무엇인가?"** → 이에 답할 준비를 해둘 것.
- 폐쇄망(은행 전산실 등) 환경: 인터넷·휴대폰·외부 도구 제한 → **AI 도움 없이도 기본 코딩이 가능해야** 함. (요즘은 로컬 AI를 쓸 수 있어 조금 나아졌지만 제약이 있음)
- 괜찮은 회사들은 **논문을 직접 읽고 발표**하는 문화 (금요일 오전 논문 발표 등). 에이전트 시대에는 **에이전트 관련 논문**까지 봐야 함.
- 프로젝트도 **진도를 조금씩 나가며** 마음 편하게 진행할 것.

### 일정 안내
- 내일 **프로젝트 발표** (순서는 **먼저 메일 보낸 팀 순서**로, 발표자는 **내일 아침 랜덤 지정**).
- 제출 마감 시간 이후엔 **타임아웃** → 두 팀 동시 발표.
- **논문 스터디**는 모든 팀이 **동일 주제**로 진행 (물어볼 시간 확보를 위해).
- 시간이 남으면 **Halving Random Search → 활성화 함수(Activation) 댄스** 순으로 진행 예정.

---

## 11. 오늘의 핵심 요약 ✅

1. **스레드 동시 처리**: `ThreadPoolExecutor` + `submit` / `map` + `with` 문.
2. **Continuous Batching**은 NPU가 **자동 적용** — 개념만 이해.
3. **Prefix 캐시**는 앞부분이 겹칠수록 적중 ↑ → 적중 횟수 = `after − before`.
4. 성능 지표: **TPS = 토큰 / 시간**, **RPS = 요청 / 시간**, 지연시간(latency).
5. **워커 수를 늘리면(1→2→4→8) 처리량이 향상**됨을 직접 측정 (특히 8개).
6. 토큰은 **모델별 토크나이저가 정의**하는 단위 (한글은 차이가 큼).
7. `usage_metadata`, `results` 같은 **중첩 자료구조**를 정확히 파악하고 코드 작성할 것.

---

## 12. 복습 과제 / 체크리스트

- [ ] `F28` 코드를 `for`문 없이 **더 간결하게** 개선해보기
- [ ] 워커 수(1·2·4·8·16)를 바꿔 **TPS·RPS 그래프**로 비교해보기
- [ ] 프롬프트 앞부분을 길게 고정했을 때 **Prefix 캐시 적중률** 변화 관찰
- [ ] `submit`과 `map`의 차이를 **내 말로 설명**해보기
- [ ] "AI·에이전트·바이브 코딩 없이 내가 할 수 있는 것" **면접 답변 준비**
