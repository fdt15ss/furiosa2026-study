# ML 6일차 정리

> 오늘의 흐름
> 1. 논문 발표 — *Attention Is All You Need* (Transformer) + 발표 피드백
> 2. Batch Normalization vs Layer Normalization, **BN과 ReLU의 순서**
> 3. 실습 ① Keras CNN에 `BatchNormalization` 넣고 3가지 방식 성능 비교
> 4. POC 기업 안내 및 팀빌딩 일정 (간단 요약)
> 5. 실습 ② 퍼셉트론 / LinearSVC로 AND·OR·XOR → **XOR 문제와 AI의 겨울**
> 6. 실습 ③ TensorFlow로 단층 → 다층 퍼셉트론(MLP)으로 XOR 해결 시도, Decision Tree
> 7. AI 역사 정리(1·2차 겨울) + 다음 논문 예고 (ReAct / YOLO)

---

## 1. Transformer (*Attention Is All You Need*) 핵심 정리

### 1-1. 등장 배경 — Seq2Seq의 한계
| 구조 | 설명 |
|---|---|
| **인코더** | 원문을 **순차적으로** 계산해 고정 길이 벡터 `V`(context vector)에 압축 |
| **디코더** | 고정 벡터 `V`를 받아 **순차적으로** 번역 결과 생성 |

**단점**
1. **느리다** — RNN 기반이라 앞 단계 계산이 끝나야 다음 단계 계산 가능 → 병렬화 불가
2. **장기 의존성 문제** — 문장이 길어지면 앞쪽 단어 정보가 희미해짐 (고정 벡터 병목)

→ **Attention**이 2번 문제를 해결: 출력 단어마다 **원문 전체**를 참조하고, 관련 있는 부분에 더 큰 가중치 부여
→ 그래도 여전히 RNN이 남아 있어 느림 → **"RNN 없이 Attention만으로 가능한가?"** → Transformer

### 1-2. 전체 구조
- 인코더 / 디코더 각각 **6층** 스택
- **인코더 레이어** = 2개의 서브레이어
  1. Multi-Head Self-Attention
  2. Position-wise Feed Forward
  - 각 서브레이어 뒤에 **Add & Norm** (Residual connection + Layer Normalization)
- 인코더의 최종 출력(**Key, Value**)이 디코더의 **모든 층**에 전달됨

### 1-3. Q, K, V와 Scaled Dot-Product Attention

```
Attention(Q, K, V) = softmax( Q·Kᵀ / √d_k ) · V
```

| 요소 | 역할 |
|---|---|
| **Query** | "내가 찾고 싶은 것" |
| **Key** | "각 단어가 가진 꼬리표" — Q와 내적해 **적합도(관계 점수)** 계산 |
| **Value** | 실제 정보 — 가중치로 **가중합** |

**왜 √d_k로 나누나? (면접 단골)**
- 논문에서 `d_k = 64` → 내적값의 표준편차가 약 `√64 = 8`로 커짐
- 값이 크면 softmax 출력이 한 칸에 거의 1로 몰림 → **기울기가 0에 수렴 (gradient vanishing)** → 학습이 안 됨
- `√d_k`로 나눠 **분산을 1 근처로 고정** → 학습 가능한 상태 유지

### 1-4. Multi-Head Attention
- `d_model = 512` 차원을 **64차원 × 8개 헤드**로 분할
- 각 헤드에서 독립적으로 attention 계산 → **Concat** → 출력 가중치 `W^O` 곱
- 헤드별 차원을 줄였기 때문에 **전체 계산량은 single-head와 거의 비슷**
- 논문 실험상 **헤드 8개**가 가장 좋았음

### 1-5. Transformer의 3가지 Attention
| 종류 | Query | Key / Value | 특징 |
|---|---|---|---|
| **Encoder Self-Attention** | 인코더 | 인코더 | 원문 단어끼리 서로 참조 |
| **Masked Decoder Self-Attention** | 디코더 | 디코더 | **현재 위치까지만** 참조 |
| **Encoder-Decoder Attention** | 디코더 | 인코더 최종 출력 | 번역 시 원문 참조 |

### 1-6. 디코더 마스킹 (Masking)
- 학습 시 정답 문장을 한 번에 넣음 (teacher forcing)
- 그러면 "나는"이 뒤의 "밥을"을 미리 볼 수 있음 → **치팅**
- 해결: softmax **전에** 미래 위치에 `-∞`를 넣어 → softmax 후 0이 되게 함

```python
import numpy as np
seq_len = 4
mask = np.triu(np.ones((seq_len, seq_len)), k=1)   # 대각선 위쪽 = 미래
scores = np.random.randn(seq_len, seq_len)
scores = np.where(mask == 1, -np.inf, scores)       # 미래 위치 -inf
weights = np.exp(scores) / np.exp(scores).sum(axis=-1, keepdims=True)
print(weights.round(2))   # 위 삼각형이 전부 0
```

### 1-7. 발표 피드백 (면접에도 그대로 적용)
- **"어", "에", "음" 같은 습관어 금지** — 실력 외적인 걸로 감점당하지 말 것
- 자기소개는 **짧고 간결하게**, 불필요한 형용사/관사 빼기
- 지적 한 번 받고 머리가 하얘지는 상황 → **평정심 유지 연습** 필요
- 팀원끼리 리허설하며 서로의 습관을 잡아줄 것
- 논문 용어가 틀렸어도 → "**논문에 이렇게 명시되어 있습니다**"로 근거를 대면 됨

---

## 2. Batch Normalization vs Layer Normalization

### 2-1. 용어 정리 — "Normalization인데 사실은 Standardization"
| 용어 | 변환 | 예 |
|---|---|---|
| **Normalization (정규화)** | 0 ~ 1 사이로 | `MinMaxScaler` |
| **Standardization (표준화)** | **평균 0, 분산 1** | `StandardScaler` |

- Batch Normalization은 이름은 Normalization이지만 실제로는 **평균 0, 분산 1 = Standardization**을 함
- 면접 답변 팁: *"원래 normalization은 0~1 정규화를 뜻하지만, BN 논문(2015)에서는 평균 0·분산 1로 바꾸는 것을 normalization이라고 명시하고 있습니다."*

### 2-2. BN vs LN
| 구분 | Batch Normalization | Layer Normalization |
|---|---|---|
| 정규화 단위 | **배치 단위** (같은 피처를 배치 전체에 걸쳐) | **레이어(샘플) 단위** (한 샘플의 모든 피처) |
| 배치 크기 영향 | 배치가 **작으면 불안정**, 클수록 잘 먹힘 | 배치 크기와 무관 |
| 주로 쓰는 곳 | **CNN** (원래 conv용으로 나옴) | **RNN, Transformer** (Add & Norm) |

```python
import numpy as np
x = np.random.randn(32, 10)   # (batch, features)

# Batch Norm: 피처별로 배치 축(axis=0) 기준
bn = (x - x.mean(axis=0)) / np.sqrt(x.var(axis=0) + 1e-5)

# Layer Norm: 샘플별로 피처 축(axis=1) 기준
ln = (x - x.mean(axis=1, keepdims=True)) / np.sqrt(x.var(axis=1, keepdims=True) + 1e-5)
```

### 2-3. 왜 강력한가?
- 일반적인 전처리(스케일링)는 **#1 데이터 단계**에서 1번만 함
- BN은 **#2 모델 구간 안에서**, 훈련 배치마다 **은닉층 출력**을 계속 표준화
- 그래서 **Dropout급으로 강력한 기법** — 학습 안정화, 수렴 가속

### 2-4. BN과 ReLU의 순서 (면접 단골 ★)
- **2015년 BN 원논문**: `Conv/Dense → BatchNorm → ReLU` (활성화 함수 **전**에 BN)
  - 이유: 활성화 함수로 들어가는 입력 분포를 조정 → 0 근처에 모인 데이터를 ReLU가 절반 자름
- **실무 / Keras 흔한 코드**: `Conv(activation='relu') → BatchNorm` (활성화 **후**에 BN)
- 결론
  - 현재 정답은 **"둘 다 가능, 해보고 좋은 걸 쓴다"** = **하이퍼파라미터 튜닝의 일부**
  - CNN에서는 원논문 방식이 조금 더 좋다는 의견이 다수지만 명확하진 않음
- BN은 **여러 레이어에 넣어도 됨** (Dense 뒤에도 OK)
- BN + Dropout을 한 레이어에 같이 쓰는 건 가능하지만 **권장하지 않음** → 레이어당 하나 정도

---

## 3. 실습 ① Keras CNN + BatchNormalization 3가지 비교

### 3-1. 방식 1 — BN 없음 (baseline)
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))
```

### 3-2. 방식 2 — Keras 흔한 방식 (ReLU → BN)
```python
from tensorflow.keras.layers import BatchNormalization

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(BatchNormalization())          # 활성화 '후'에 BN
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(10, activation='softmax'))
```

### 3-3. 방식 3 — 원논문 방식 (BN → ReLU)
```python
from tensorflow.keras.layers import BatchNormalization, Activation

model = Sequential()
model.add(Conv2D(32, (3, 3), activation=None, input_shape=(28, 28, 1)))  # 활성화 X (linear)
model.add(BatchNormalization())
model.add(Activation('relu'))            # BN '후'에 ReLU
model.add(Conv2D(64, (3, 3), activation=None))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(64))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(10, activation='softmax'))
```

포인트
- `activation=None`(= `'linear'`) 으로 두고, `Activation('relu')` **레이어를 따로 추가**
- 실행 결과: BN을 넣으니 **순식간에 acc 0.8 → 0.9대**로 상승, 원논문 방식은 약간 느림
- 과제: 세 가지를 **50 epoch**로 돌려 성능 비교 → 반 의견도 갈림(ReLU 다음 / 원논문 / BN 없음) → 결국 **데이터·모델마다 다름 → 튜닝 대상**

### 3-4. (참고) 실행 시간 재는 데코레이터
```python
import time
from functools import wraps

def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print(f"{func.__name__} 실행시간: {time.time() - start:.2f}초")
        return result
    return wrapper

@timer
def train(model, x, y):
    return model.fit(x, y, epochs=50, batch_size=32, validation_split=0.2, verbose=0)
```

---

## 4. POC 기업 / 팀빌딩 (요약)

| 기업 | 주제 | 특이사항 |
|---|---|---|
| 휴데이터스 | NPU 기반 온프레미스 LLM 광고 콘텐츠 분석·변환 자동화 | 기술 멘토링 어려움, 기획 위주, 채용 의사 있음 |
| 시너지어 | 자유 주제 (제조 공정 데이터 AI 모델/에이전트) | 멘토링 적음, 자율 프로젝트 원하는 팀 추천 |
| Potential Labs | 상용 앱(강아지 일기) 고도화 | 기획 중심, 오프라인 멘토링 긍정적 |
| 보물AI | 데이터 기반 LLM 학습 분석 리포트 | 외국인 멘토 → **영어** 가능자 필요 |
| 유니움 | 시니어 생애 인터뷰 AI 에이전트 · 자서전 자동 생성 | 시나리오 구체적, 개발 비중 높음, 100% 온라인 예상 |
| 로그베이스 | B2B 몰 운영용 AI 오퍼레이션 에이전트 | 멘토링 우호적(주 1회), **NPU 있는/없는 버전 성능 비교 필수**, API 키 GitHub 업로드 금지 |

- 구글폼으로 **다음 주 월요일까지** 희망 기업 1~6순위 제출 → 11/24~25 조정 → 목요일 확정, **11/27 협약서 작성**, 이후 팀별 자리 배치
- 결과물 권리는 **교육생 귀속**, 기업은 비독점적 사용권. 기업 데이터·자료·상표는 기업 소유
- 종료 후 **1년간 경쟁사 제공·동일 코드 창업 금지**
- 포트폴리오 활용 OK, 단 화면 캡처·시연 영상·소스 공개 범위는 **기업과 협의**
- 프로젝트 일지 **매일**, 회의록 **주 1회**
- 프로젝트 기간 GPU: 팀당 **2장(파드 개념)** → KV 캐시를 한쪽으로 몰아 큰 모델 돌려보기 실험 권장

---

## 5. 실습 ② 퍼셉트론 / LinearSVC — AND, OR, XOR

### 5-1. AND 게이트 (Perceptron)
```python
import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

# 1. 데이터
x_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])   # shape (4, 2)
y_data = np.array([0, 0, 0, 1])                       # AND

# 2. 모델
model = Perceptron()          # 또는 LinearSVC()

# 3. 훈련
model.fit(x_data, y_data)

# 4. 평가, 예측
y_pred = model.predict(x_data)
print("score :", model.score(x_data, y_data))
print("acc   :", accuracy_score(y_data, y_pred))     # 1.0
```
- 리스트로 넣어도 sklearn이 자동으로 인식하지만 `np.array`로 맞추는 습관 추천

### 5-2. OR 게이트
```python
y_data = np.array([0, 1, 1, 1])   # 나머지 코드 동일 → acc 1.0
```

### 5-3. XOR 게이트 — 여기서 막힌다
```python
y_data = np.array([0, 1, 1, 0])   # 같으면 0, 다르면 1
```
- Perceptron, LinearSVC 모두 **0.5 ~ 0.75**가 한계, 아무리 해도 **1.0 불가**
- 이유: AND·OR은 **직선 하나로 분리 가능(선형 분리)** 하지만, XOR은 **직선 하나로 절대 분리 불가**
- 해결 아이디어: 종이를 **접는다** = 공간을 변형한다 = **층을 쌓는다** → 다층 퍼셉트론

```
AND / OR              XOR
 1 | o   x            1 | x   o
 0 | o   o            0 | o   x
   +------              +------
     0   1                0   1
 직선 1개로 분리 OK      직선 1개로 분리 불가
```

### 5-4. AI의 1차 겨울
- 1969년 민스키 & 페퍼트 『Perceptrons』 — **단층 퍼셉트론은 XOR을 못 푼다** 증명 → AI 연구 침체 (1차 겨울)
- 해결: **다층 퍼셉트론(MLP) + 역전파(Backpropagation)** — 1986년 럼멜하트·**제프리 힌튼** 등

---

## 6. 실습 ③ TensorFlow로 XOR 해결

### 6-1. 단층 퍼셉트론 (실패 버전)
```python
import numpy as np
import tensorflow as tf
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 시드 고정
SEED = 333
tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

x_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_data = np.array([0, 1, 1, 0])

# 단층: 히든 레이어 없음
model = Sequential()
model.add(Dense(1, input_dim=2, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_data, y_data, epochs=100, verbose=0)

result = model.evaluate(x_data, y_data)        # [loss, acc]
print("loss :", result[0])
print("acc  :", result[1])

y_pred = np.round(model.predict(x_data))       # sigmoid 출력 → 0/1
print(y_pred.ravel())
```
- `evaluate()`의 반환값 = `[loss, metrics...]` → `result[1]`이 acc
- `predict()`는 확률(0~1)이므로 `np.round()`로 0/1 변환
- 결과: acc ≈ **0.5 ~ 0.75**. epoch↑, EarlyStopping, BN, batch_size 조절해도 **1.0 불가** (히든층이 없으니까)

### 6-2. 다층 퍼셉트론 (MLP)
```python
model = Sequential()
model.add(Dense(10, input_dim=2, activation='relu'))   # 히든층 ★ 활성화 함수 필수
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_data, y_data, epochs=500, verbose=0)
print(np.round(model.predict(x_data)).ravel())         # [0. 1. 1. 0.] 기대
```

> 💡 **수업 중 막혔던 부분 보충**
> 수업에서는 층을 쌓아도 1.0이 안 나와서 TensorFlow 버전 문제(2.8 이후 변화)로 추정하고 넘어갔음.
> 그런데 체크할 점: `Dense(10)`처럼 **activation을 안 주면 기본값이 `None`(linear)** 이다.
> 선형 층을 아무리 쌓아도 합성하면 결국 **선형 1개와 같음** → XOR을 못 푼다.
> 수업 중 "렐루 넣어볼게요 / 시그모이드 넣어볼게요"가 나온 이유가 바로 이것 → **히든층에 비선형 활성화 함수(relu, sigmoid, tanh)** 를 넣고, epoch을 충분히(수백 회) 주면 보통 해결된다.
> 데이터가 4개뿐이라 시드에 따라 local minimum에 빠질 수 있으니 seed·노드 수·learning rate도 바꿔볼 것.

### 6-3. Decision Tree — 바로 해결
```python
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier()
model.fit(x_data, y_data)
print(model.score(x_data, y_data))   # 1.0
```
- 트리는 **축에 평행한 분할을 여러 번** 하므로 비선형 경계 표현 가능 → XOR 해결

### 6-4. 환경 메모
- TF 버전이 올라가며 예전 코드가 deprecated 되는 경우 많음 → **가상환경을 버전별로 분리**해 두기
  ```bash
  conda create -n tf212 python=3.10 -y
  conda activate tf212
  pip install tensorflow==2.12
  python -c "import tensorflow as tf; print(tf.__version__)"
  ```

---

## 7. AI 역사 — 겨울과 해결

| 시기 | 문제 | 해결 |
|---|---|---|
| **1차 겨울** | 단층 퍼셉트론으로 **XOR 문제** 해결 불가 | **다층 퍼셉트론 + 역전파** (힌튼 등) |
| **2차 겨울** | 층을 깊게 쌓으면 **기울기 소실(vanishing) / 폭주(exploding)**, 연산량 부족 | ReLU·초기화 기법 등 + **GPU 활용** (앤드류 응 등, 2009년경 GPU로 딥러닝 가속 입증) → **컴퓨팅 파워(장비빨)** |
| 이후 | 2019~20년 "3차 겨울" 우려 | 생성형 AI 붐, NVIDIA GPU·메모리 수요 폭증 → 당분간 겨울은 없을 듯 |

- AI 4대 천왕: 제프리 힌튼, 얀 르쿤, 요슈아 벤지오, 앤드류 응

---

## 8. 다음 논문 / 일정 예고

- **다음 주 목요일 논문**: **ReAct**, Reflection, CoT
  - ReAct 안에 CoT가 포함됨
  - Reflection은 요즘 잘 안 쓰지만 개념은 알아둘 것
  - 핵심은 ReAct = 실질적으로 **Tool Calling** 개념 → 비교적 쉬운 회차
- **그다음 논문(프로젝트 시작 후)**: **YOLO**
  - **YOLOv1의 Loss Function은 반드시 명확히 이해** (v2~v5는 "무엇이 바뀌었나" 정도만)
  - 이후 버전 흐름: ... v11 → v12 → v26 (이후 연도 표기), 일부 버전엔 Transformer 도입
  - 구현은 **YOLO v26**을 Hugging Face에서 가중치 받아 지난번 데이터·소스 재활용
- **11/27부터 팀 프로젝트(POC) 시작** → 논문은 일단 여기서 마무리, 더 하고 싶은 팀은 별도 신청

---

## ✅ 오늘의 면접 대비 체크리스트
- [ ] Seq2Seq의 두 가지 한계와 Attention / Transformer가 각각 무엇을 해결했나
- [ ] Q·K·V의 역할, **√d_k로 나누는 이유**
- [ ] Multi-Head Attention의 차원 분할(512 → 64×8)과 계산량
- [ ] 디코더 마스킹에서 `-∞`를 softmax **전**에 넣는 이유
- [ ] Batch Norm vs Layer Norm 차이, BN이 사실 Standardization인 점
- [ ] **BN과 ReLU 순서** — 원논문(BN→ReLU) vs 실무, 결론은 튜닝 대상
- [ ] XOR이 단층 퍼셉트론으로 안 풀리는 이유 (선형 분리 불가)
- [ ] 히든층에 **비선형 활성화 함수**가 없으면 층을 쌓아도 의미 없는 이유
- [ ] AI 1차·2차 겨울의 원인과 해결책
