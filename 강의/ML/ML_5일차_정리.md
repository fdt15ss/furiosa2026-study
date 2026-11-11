# ML 5일차 정리

> 녹취(STT) 기반 정리입니다. 음성 인식 오류가 많아 고유명사·용어는 문맥으로 보정했고, 불확실한 부분은 `(추정)`으로 표시했습니다.

---

## 0. 오늘의 큰 흐름

| 시간대 | 내용 |
|---|---|
| 오전 | 논문 스터디 (논문 달리기) |
| 오후 1 | LLM 서빙 성능 지표 실습 (응답시간, TTFC, Prefill/Decode, `usage_metadata`, TPS) |
| 오후 2 | ML — **HalvingGridSearchCV** (그리드/랜덤 서치 다음 단계) |
| 기타 | 퓨리오사(NPU) 서버 환경 구성, KV 캐시 모니터링, 학원 공지 |

---

## 1. LLM 응답 성능 측정 (LangChain + 로컬 서빙)

### 1-1. LLM 객체 가져오기 & 파라미터

- 직접 만든 헬퍼 모듈(수업에서 `get_llm()` 형태)을 `import`해서 사용 → 함수 안에서 **`return`을 빠뜨리지 않기**
- 서버(퓨리오사 LLM 서브)를 **먼저 켜 놓아야** 실행됨 — 서버가 꺼져 있으면 에러
- 주요 파라미터
  - `max_tokens` : 출력 토큰 상한 (안 주면 모델 기본값)
  - `temperature` : 무작위성
  - `top_p` : 누적 확률 샘플링
- 호출 방식은 사실상 **두 가지**: `invoke()`(한 번에 받기) / `stream()`(청크 단위로 받기)

#### 보충: `top_p` = 누적 확률 샘플링(Nucleus Sampling)이란?

LLM은 다음 토큰을 고를 때 **모든 후보 토큰에 확률**을 매깁니다. `top_p`는 "확률이 높은 순서대로 더해서 **누적 확률이 p에 도달할 때까지의 후보만** 남기고, 그 안에서만 뽑자"는 방식입니다.

**예시: 다음 단어 후보 확률 (`top_p=0.9`)**

| 순위 | 후보 토큰 | 확률 | 누적 확률 | 후보 포함 여부 |
|---|---|---|---|---|
| 1 | 인공지능 | 0.50 | 0.50 | O |
| 2 | 기술 | 0.25 | 0.75 | O |
| 3 | 머신러닝 | 0.15 | 0.90 | O (여기서 0.9 도달 → 마감) |
| 4 | 로봇 | 0.06 | 0.96 | X |
| 5 | 바나나 | 0.04 | 1.00 | X |

- 누적 확률이 0.9에 도달하는 상위 3개 토큰만 남기고, 이 3개의 확률을 다시 정규화해서 그 안에서 샘플링합니다.
- 꼬리 부분의 **엉뚱한 단어(바나나 등)를 잘라내는** 효과가 있습니다.

**값에 따른 차이**

| `top_p` | 후보 범위 | 결과 경향 |
|---|---|---|
| 낮음 (예: 0.1~0.5) | 확률 높은 소수 토큰만 | 안정적·일관적, 다양성 낮음 |
| 높음 (예: 0.9~1.0) | 거의 모든 후보 | 다양하고 창의적, 가끔 튀는 답 |
| 1.0 | 전체 후보 (제한 없음) | 필터링 없음 |

**`top_k`, `temperature`와 비교**

| 파라미터 | 방식 |
|---|---|
| `top_k` | 확률 상위 **k개 개수**만 남김 (예: 상위 40개) |
| `top_p` | 확률 **누적합이 p가 될 때까지**만 남김 → 상황에 따라 후보 수가 **자동으로 변함** (확신이 높으면 후보가 적고, 애매하면 많아짐) |
| `temperature` | 확률 분포 자체를 평평하게/뾰족하게 만듦 (높을수록 무작위) |

- 보통 `temperature`와 `top_p`는 **둘 중 하나 위주로 조절**하는 것을 권장합니다.
- 한 줄 요약: **"확률 높은 순으로 누적해서 p%가 될 때까지의 후보 안에서만 다음 단어를 뽑는다."**

### 1-2. 평균 응답시간 측정

```python
import time

times = []
for i in range(5):
    start = time.time()          # 또는 time.perf_counter()
    llm.invoke(prompt)
    end = time.time()
    times.append(end - start)
    print(f"{i+1}회: {end - start:.2f}초")   # 소수 둘째 자리

print("평균:", round(sum(times) / len(times), 2))
```

- `time.time()` vs `time.perf_counter()` : 편한 것 쓰되 **한 코드에서 섞어 쓰지 않기** (start/end는 같은 함수로)
- 결과 예시: `max_tokens=500`일 때 평균 약 5.3초대
- 프롬프트를 던진 시점 → 최종 출력까지 걸린 시간이 응답시간

### 1-3. 스트리밍 출력

```python
for chunk in llm.stream("인공지능의 발전 과정을 설명해줘"):
    print(chunk.content, end="", flush=True)
```

- `end=""` : 줄바꿈 하지 않기 (기본 `print`는 줄바꿈됨)
- `flush=True` : 버퍼를 바로 비워서 **타자 치듯 실시간 출력**. 안 쓰면 한꺼번에 나오는 것처럼 보일 수 있음
- 한글까지 제대로 보려면 `max_tokens`를 1000~2000 정도로 충분히 주기
- 스트림에서 나오는 각 조각 = **chunk**

---

## 2. 성능 지표 용어 (면접에서 물어볼 수 있는 것들)

> 강사 말: "알고 보면 아무것도 아니니까 용어 나오면 겁먹지 말고 찾아보거나 물어봐라."

| 용어 | 풀네임 | 의미 |
|---|---|---|
| **TTFC** | Time To First Chunk | 요청을 보낸 뒤 **첫 번째 청크**가 나올 때까지 걸린 시간 |
| **TTFT** | Time To First Token | 첫 번째 **토큰**이 나올 때까지의 시간 (TTFC와 거의 비슷) |
| **Prefill** | — | 입력 프롬프트를 처리하는 단계 ≈ TTFT 구간 |
| **Decode** | — | 토큰을 하나씩 생성하는 단계 (수업에서는 "전체 완료까지" 정도로 이해) |
| **TPS** | Tokens Per Second | 초당 생성 토큰 수 = `출력 토큰 수 / 걸린 시간(초)` |

### 2-1. TTFC 측정 — 같은 결과를 내는 3가지 코드

**① 첫 청크에서 시간을 기록 (기본형)**

```python
start = time.time()
first_time = None
for chunk in llm.stream(prompt):
    if chunk.content:
        if first_time is None:
            first_time = time.time() - start
        print(chunk.content, end="", flush=True)
print("\nTTFC:", round(first_time, 2))
```

**② 첫 청크에서 `break`** — 끝까지 돌 필요가 없으므로 출력 없이 중단

```python
start = time.time()
for chunk in llm.stream(prompt):
    if chunk.content:
        ttfc = time.time() - start
        break
```

**③ `next(iter(...))`** — for문 자체가 필요 없음

```python
start = time.time()
chunk = next(iter(llm.stream(prompt)))   # 첫 번째 청크만 꺼냄
ttfc = time.time() - start
```

- 세 방식 모두 약 **0.03~0.04초**. "첫 번째가 나올 때까지"라 순식간이 정상
- ②, ③은 **파이썬 코드를 간결하게** 만든 것일 뿐 결과 차이는 없음

### 2-2. Prefill / Decode 측정

- 프롬프트를 리스트로 여러 개 돌려서 각각 **TTFT(= prefill)** 와 **전체 시간(= decode 포함)** 을 비교
- 강사 정리: "정확한 값을 따지기보다 **prefill ≈ TTFT**, **decode ≈ 전체 생성 시간** 정도의 개념으로 이해하면 충분"
- 관찰: 입력이 길다고 항상 느려지는 건 아님. **출력 토큰 수가 많을수록** 전체 시간이 길어진다 (예: 2000토큰 꽉 채운 응답은 약 21초)

---

## 3. 응답 메타데이터 `usage_metadata`

```python
response = llm.invoke(prompt)
print(response.content)            # 답변 본문
print(response.usage_metadata)     # 토큰 사용량
```

- 응답 객체(`response`)는 `content` 말고도 **딕셔너리/클래스 형태의 많은 속성**이 들어 있음 → 하나씩 익혀 두기
- `usage_metadata` 주요 키

| 키 | 의미 | 예시 |
|---|---|---|
| `input_tokens` | 입력 토큰 수 | 14 |
| `output_tokens` | 출력 토큰 수 | 813 |
| `total_tokens` | 입력 + 출력 | 827 |
| `input_token_details` | 오디오 토큰, **캐시 읽기(cache_read)** 등 | audio 0, cache_read 13 |
| `output_token_details` | 출력 쪽 세부 | — |

- `max_tokens=500`을 주면 `output_tokens`가 **500에서 잘림** → 토큰 길이 조절 확인에 활용
- **TPS 계산**

```python
elapsed = end - start
out_tokens = response.usage_metadata["output_tokens"]
tps = round(out_tokens / elapsed, 2)
print("TPS:", tps)     # 수업 예: 약 22.04 (1000토큰 기준)
```

- 활용: 로컬 모델 vs 다른 서비스(코덱스 등)의 **성능 비교 지표**로 TPS 사용
- 프롬프트 3종 비교 실습: 짧은 질문("NPU란?") → 출력 약 1100~1200토큰, "활용 분야를 더 자세히" → 출력 2000토큰 상한까지 채움(21.65초)
- 비동기(async) 처리는 아직 안 나왔고 **나중에 비교** 예정 ("차이는 크지 않을 것")

---

## 4. 파이썬 기초 포인트 (코드 짜면서 복습한 것)

- **이터레이터(Iterator)**: `for`문은 내부적으로 이터레이터에서 값을 하나씩 꺼냄
  - 옛날 방식 `.next()` 는 deprecated → **`next(iterator)`** 내장 함수 사용
  - `next(iter(x))` → 첫 번째 값만 꺼냄
- `if chunk.content:` → 내용이 있을 때만 실행 (빈 문자열은 거짓)
- `if first_time is None:` → **처음 한 번만** 실행시키는 플래그 패턴
- `break` → 필요한 만큼만 돌고 종료
- 파일명 규칙을 번호로 관리 (`f14_1_ttfc.py`, `f14_3_ttfc.py` …) — 앞 코드를 복사해 조금씩 고쳐 가며 실습
- 함수에서 `return` 빼먹지 않기 / 오타(`print`, `import`)는 에러 로그부터 읽기
- 리스트에 `append` → `sum()/len()` 으로 평균

---

## 5. 머신러닝: HalvingGridSearchCV (반땅 그리드 서치)

### 5-1. 개념

- **GridSearchCV**: 모든 하이퍼파라미터 조합을 전체 데이터로 전부 학습 (느림)
- **RandomizedSearchCV**: 무작위로 일부 조합만 학습
- **HalvingGridSearchCV**: 적은 데이터로 **모든 후보를 먼저 가볍게 학습 → 상위 후보만 남기고 → 데이터(리소스)를 늘려가며 반복**
  - 토너먼트(승자 진출) 방식, 마지막엔 가장 좋은 파라미터 하나만 남아 `refit`
- 이름은 "halving"이지만 실제 기본값은 **`factor=3` → 1/3씩** 줄어듦
- **HalvingRandomSearchCV** 도 있음 (후보를 랜덤으로 뽑는 버전)

### 5-2. 임포트 (버전 이슈 주의)

```python
# scikit-learn 버전에 따라 아래 줄이 필요할 수 있음 (실험적 기능 활성화)
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import HalvingGridSearchCV
```

- 정식 기능이 아닌 것은 `sklearn.experimental` 안의 **`enable_...`** 를 먼저 import 해야 쓸 수 있음
- "안 되면 찾아보면 다 나온다" — 버전/ImportError는 검색으로 해결
- `random_state`, `n_iter` 같은 랜덤 서치용 파라미터는 **그리드 계열에는 없음** (제거)

### 5-3. 실습 설정과 결과 해석

- 데이터: `load_digits` (10개 클래스), 학습 샘플 **1437개**, `cv=5`, 파라미터 조합 **42개**, `verbose=1`

**factor=3 (기본) 일 때**

| 반복(iter) | n_candidates | n_resources | 학습 횟수 |
|---|---|---|---|
| 0 | 42 | 100 | 42 × 5 = **210** |
| 1 | 14 (= 42/3) | 300 (= 100×3) | 14 × 5 = **70** |
| 2 | 5 | 900 | 5 × 5 = **25** |

- 총 약 305회 + `refit` 1회 ≈ 306회 → 그리드 서치(전체 42×5=210회를 **전체 데이터**로)보다 훨씬 가벼움
- `max_resources = 1437` : 학습 데이터 행 수 (`X_train.shape[0]`)
- `n_resources × factor`가 `max_resources`를 넘으면 더 이상 반복하지 않음 (900 × 3 = 2700 > 1437 → 종료)

**`min_resources`가 100인 이유 (수업 설명)**
- 기본값(`'smallest'`)이 대략 **`2 × cv × 클래스 수`** = 2 × 5 × 10 = **100**
- 알고리즘상 의미 있는 값이라기보다 "최소한 이 정도는 있어야 교차검증이 돌아간다"는 최소치

**factor=2로 바꾼 실험**
- 리소스: 100 → 200 → 400 → 800 (1600은 한도 초과) → **iteration 4번**
- 후보 수: 42 → 21 → 11 → 6 내외로 감소
- `min_resources=50`으로 줄이면 반복이 5번으로 늘어남

### 5-4. 장단점

| | 내용 |
|---|---|
| 장점 | 데이터가 **클 때(예: 10만 건)** 그리드 서치가 사실상 불가능한 상황에서 매우 강력. 상위 후보에만 자원을 집중 |
| 단점 | 데이터가 **작을 때** 초반 소량 데이터로 학습하므로 과적합/불안정 → 좋은 후보를 초기에 탈락시킬 수 있음 |
| 대처 | `min_resources`를 크게 잡거나, 데이터가 작으면 일반 그리드 서치 사용 |

- 어떤 서치가 좋은지는 **상황(데이터 크기·시간)을 보고 선택** — 정답은 없음

### 5-5. 파일/환경

- 이전 `M09` (RandomizedSearchCV) 코드를 복사해 `M10_halving_grid_search_cv` 로 작성
- 가상환경 `ML311` 사용
- 랜덤 서치 → 그리드 서치로 이름만 바꾸고 파라미터 일부 제거하여 실행

---

## 6. 리눅스 / 서버 / 도구

- **SSH로 퓨리오사 서버 접속** 후 작업 (CPU/NPU 환경, 가상환경 활성화 후 LLM 서브 실행)
- **터미널 모니터링**: NPU 상태(status) 창을 켜 두고 메모리/사용량 확인
- **KV 캐시 (Key-Value Cache)**
  - 모델 서버가 사용하는 캐시 메모리. 수업 화면에서 약 **7.8%** 사용 중
  - 갑자기 70~90%까지 치솟으면 → 메모리 터질 위험 → **`Ctrl + C`** 로 중단
  - 연결 개념: 트랜스포머의 **Query / Key / Value** — "트랜스포머 논문 발표 때 Q·K·V를 설명하라는 질문은 100% 나온다"
- **서버에 직접 Claude 같은 에이전트를 설치하는 것**: 제공받은 리눅스 서버 자원만 쓰게 되어 NPU를 쓰는 의미가 약해짐. 대안으로 **Hermes(에이전트 도구) + 유료 API를 메인, 무료 API(또는 퓨리오사 서브 모델)를 폴백**으로 구성 → 비용이 떨어지면 자동으로 넘어가게 하는 방식 `(추정, 오인식 구간)`
- 24시간 쓸 수 있는 장비가 지급되었으므로 **수업 외 시간에 자유롭게 실습** (LLM에서 한 것들 다시 해 보기, 에이전트 2~3개 연결해 오케스트레이션 등)

---

## 7. 학습 로드맵 (강사 설명)

- 현재 **3트랙**으로 진행: ① 퓨리오사(NPU) ② 논문 ③ 정규 수업(ML/DL)
- 오전: 논문 / 오후: 퓨리오사 NPU (오늘 예정: **KV 캐시·배칭**, **서빙 환경 구성**)
- 딥러닝(TF/Keras → PyTorch): Keras 문법은 PyTorch에도 거의 그대로 대응되어 **PyTorch는 시간이 덜 걸림**
- 논문 주차별 흐름
  1. 1주차: CNN 계열 — **One-stage / Two-stage 모델** 개념 이해
  2. 2주차: **Transformer** (필수, "어디에 가든 무조건 나온다")
  3. 3주차 이후: 세부 내용은 다음 주에 안내, 최신 논문은 5·6주차에 결정
- 남은 기간 약 35일 안에 마무리

---

## 8. 학원 공지 / 기타

- **출석**: 9시까지 전원 착석 → 오전 논문 스터디. **지각자가 한 명이라도 있으면 진도를 당김** (최대 4시까지)
- **발표**: 목요일 발표 예정
- **취업 지원**: 이메일 접수 마감은 당일 **23:59**
- **면접 대비**: 12월 초부터 면접 본격화, **정장 준비** (다음 주 수요일 면접 예정자 있음)
- 면접 팁: 직전 회사에서 끝을 좋게 마무리해 평판 관리. 같은 회사에 동기와 함께 지원해도 무방

---

## 9. 복습 체크리스트

- [ ] `invoke` vs `stream` 차이를 설명할 수 있는가
- [ ] `print(end="", flush=True)` 가 하는 일
- [ ] TTFC / TTFT / Prefill / Decode / TPS 정의
- [ ] `usage_metadata` 의 input/output/total 토큰 확인 및 TPS 계산
- [ ] `next(iter(...))` 로 첫 청크만 꺼내기
- [ ] HalvingGridSearchCV 동작 원리 (후보 수 ÷ factor, 리소스 × factor)
- [ ] `min_resources` 기본값 계산 (`2 × cv × 클래스 수`)와 `max_resources`
- [ ] KV 캐시란 무엇이고 왜 모니터링하는가
- [ ] 트랜스포머 Q / K / V 설명 준비
- [ ] 내일 아침 9시 출석!
