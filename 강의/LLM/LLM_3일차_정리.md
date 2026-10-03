# LLM 3일차 정리 — Embedding + LSTM 텍스트 분류, sparse 라벨, LSTM/Reshape 활용

> 핵심 한 줄: **2일차에 배운 "임베딩"을 실제 모델에 붙여서 뉴스·리뷰를 분류해 봤고, LSTM·Reshape 층으로 "데이터 차원(shape)을 자유롭게 바꿔 끼우는 법"을 익혔다.**

---

## 0. 오늘의 흐름

```
Reuters(다중분류) → IMDB(이진분류) → sparse 라벨(원핫 생략)
        → LSTM으로 이미지(MNIST) 읽기 → Reshape 층으로 CNN ↔ LSTM 연결
```

- 앞의 두 개(keras62)는 **텍스트 + Embedding + LSTM** (2일차 이론의 실습판)
- 가운데(keras63)는 **라벨 처리 방식** 정리 (원핫 vs sparse)
- 뒤의 두 개(keras64, 65)는 **LSTM / Reshape의 shape 감각** 익히기
- 공통 주제: **"각 층을 지나면 shape가 어떻게 바뀌는가"** 와 **"파라미터 수 계산"**

---

## 1. 텍스트 데이터 불러오기 — Reuters / IMDB

```python
from tensorflow.keras.datasets import reuters, imdb

(x_train, y_train), (x_test, y_test) = reuters.load_data(
    num_words=1000,     # 빈도 높은 단어 1,000개만 사용
    test_split=0.2,
)
```

| 데이터 | 내용 | 클래스 | 학습/테스트 |
|---|---|---|---|
| Reuters | 뉴스 기사 주제 분류 | **46개** (다중분류) | 8,982 / 2,246 |
| IMDB | 영화 리뷰 긍정/부정 | **2개** (이진분류) | 25,000 / 25,000 |

### 알게 된 점
- 텍스트는 이미 **단어 → 번호**로 바뀐 상태로 들어온다 (2일차의 `Tokenizer` 결과와 같은 형태)
- `num_words=1000` → 단어 번호는 **0 ~ 999**만 쓰인다
- `x_train`은 `numpy.ndarray`지만 안쪽 원소는 **길이가 제각각인 `list`** → 그대로는 모델에 못 넣음
- 길이 분포를 먼저 확인하는 습관

| | 최대 | 최소 | 평균 |
|---|---|---|---|
| Reuters | 2376 | 13 | 약 145.5 |
| IMDB | 2494 | 11 | 약 238.7 |

---

## 2. pad_sequences로 길이 맞추기 (2일차 복습 + 적용)

```python
from tensorflow.keras.preprocessing.sequence import pad_sequences

x_train = pad_sequences(x_train, maxlen=200)   # Reuters: 평균 145 → 200
x_train = pad_sequences(x_train, maxlen=250)   # IMDB: 평균 238 → 250
```

- **maxlen은 평균 길이보다 조금 넉넉하게** 잡아서 대부분의 문장을 담는다
- 기본값은 `padding='pre'`, `truncating='pre'` → **앞을 0으로 채우고, 긴 건 앞을 자름**
- 왜 pre가 유리한가?
  → LSTM은 **마지막 시점의 상태**를 출력으로 쓴다. 실제 단어가 뒤쪽에 붙어 있어야 마지막 상태에 의미가 많이 남는다
- 결과 shape: Reuters `(8982, 200)`, IMDB `(25000, 250)` → **정수 인덱스 2차원**

---

## 3. 라벨(y) 처리 — 다중분류 vs 이진분류

| | Reuters (다중) | IMDB (이진) |
|---|---|---|
| y 처리 | `to_categorical` → `(N, 46)` 원핫 | **그대로 0/1** (원핫 안 함) |
| 출력층 | `Dense(46, activation='softmax')` | `Dense(1, activation='sigmoid')` |
| loss | `categorical_crossentropy` | `binary_crossentropy` |
| 예측 변환 | `np.argmax(y_predict, axis=1)` | `np.round(y_predict)` (0.5 기준) |
| 정답 변환 | `y_test`도 원핫이라 `np.argmax(y_test, axis=1)` | 이미 0/1이라 변환 없음 |

- 2일차의 "원핫은 비효율"은 **입력 X** 이야기. **라벨 y의 원핫**은 분류 문제의 기본 도구로 계속 쓴다
- `np.unique(y_train)`으로 클래스 개수 확인 → 출력층 노드 수 결정 (0~45 → 46개)

---

## 4. 모델 구성 — Embedding + LSTM

### Reuters (다중분류)
```python
model = Sequential()
model.add(Embedding(1000, 200))              # (N, 200) → (N, 200, 200)
model.add(LSTM(128))                         # (N, 200, 200) → (N, 128)
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(46, activation='softmax'))
```

### IMDB (이진분류)
```python
model = Sequential()
model.add(Embedding(input_dim=1000, output_dim=100, input_length=250))
model.add(LSTM(64))
model.add(Dropout(0.3))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
```

### shape 흐름 (Reuters 기준)
```
입력        (N, 200)          ← 정수 인덱스
Embedding   (N, 200, 200)     ← 단어 하나가 200차원 벡터로 → 3차원 (LSTM 입력 OK)
LSTM(128)   (N, 128)          ← 마지막 시점 출력만 → 다시 2차원
Dense ...   (N, 64) → (N, 46)
```

### 핵심 포인트
- **Embedding이 정수 2차원 → 실수 3차원**으로 바꿔 줘서 LSTM에 바로 연결된다 (2일차 내용 그대로)
- LSTM은 기본적으로 **마지막 시점 출력만** 내보낸다 → 뒤의 `Dense`는 일반 2차원 연결
- `Embedding(1000, 200)`의 `input_dim=1000`
  - 2일차에는 "패딩 0 때문에 +1"이라고 배웠지만, 여기서는 `num_words=1000`이 **번호 0 ~ 999** (0 포함)라서 **1000이면 충분**
  - 기준은 "**가장 큰 단어 번호 + 1**"
- Embedding 파라미터 = `input_dim × output_dim`
  - Reuters: 1000 × 200 = **200,000**
  - IMDB: 1000 × 100 = **100,000**
- `input_length`는 최신 Keras에서는 안 써도 됨 (2일차 정리 참고)

---

## 5. 콜백 — EarlyStopping + ModelCheckpoint, 저장

```python
es = EarlyStopping(monitor='val_loss', mode='min', patience=10, restore_best_weights=True)

mcp = ModelCheckpoint(
    monitor='val_loss', mode='min',
    save_best_only=True,                       # 가장 좋았던 epoch 모델만 덮어쓰며 저장
    filepath='./_save/keras62/keras62_1_mcp.keras',
    verbose=1,
)

model.fit(x_train, y_train, epochs=100, batch_size=64,
          validation_split=0.2, callbacks=[es, mcp])

model.save_weights('./_save/keras62/keras62_1_save.weights.h5')
```

| 항목 | 설명 |
|---|---|
| `EarlyStopping` | `val_loss`가 `patience` 동안 안 좋아지면 중단 |
| `restore_best_weights=True` | 중단 시점이 아니라 **가장 좋았던 epoch의 가중치로 되돌림** |
| `ModelCheckpoint` | 훈련 중 **가장 좋은 모델을 파일로 저장** (`save_best_only=True`) |
| `.keras` 저장 | **모델 구조 + 가중치** 전체 |
| `save_weights` | **가중치만** 저장 → 불러올 땐 **같은 구조의 모델을 먼저 만들고** `load_weights` |

- `restore_best_weights=True` 덕분에 `save_weights`로 저장되는 가중치 = mcp의 가중치 (최고 epoch)
- `.weights.h5` 확장자 규칙 (최신 Keras)

---

## 6. 실습 결과

| | 목표 | 결과 acc | 걸린 시간 |
|---|---|---|---|
| Reuters | 0.67 이상 | **0.7418** | 약 65초 |
| IMDB | 0.6 이상 | **0.8619** | 약 36초 |

- 평가 방법 두 가지: `model.evaluate` (loss, acc) / `predict` 후 `accuracy_score` → **값이 같게 나오는지 교차 확인**

---

## 7. sparse 라벨 — 원핫 없이 분류하기 (CIFAR100)

> 클래스가 많을수록(100개) 원핫은 메모리를 많이 쓴다 → **정답 숫자 그대로** 넣자

```python
# 원핫 코드 삭제 → y는 (50000, 1), 숫자 0 ~ 99 그대로

model.add(Dense(100, activation='softmax'))        # 출력층은 그대로 100개 + softmax
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])
```

| | 원핫 | sparse |
|---|---|---|
| y shape | `(50000, 100)` | `(50000, 1)` |
| loss | `categorical_crossentropy` | `sparse_categorical_crossentropy` |
| 출력층 | `Dense(100, softmax)` | **똑같음** |
| loss 값 | 같음 | 같음 (내부에서 원핫처럼 계산) |

### 평가할 때 주의 ⚠️
```python
y_predict = np.argmax(model.predict(x_test), axis=1)   # 예측은 확률(10000, 100) → argmax 필요
# y_test는 이미 정답 숫자 → argmax 하면 안 됨!
acc_score = accuracy_score(y_test, y_predict)
```
- `y_test`가 `(10000, 1)`일 때 `argmax(axis=1)` → 칸이 1개뿐이라 **전부 0** → **에러 없이 acc만 틀리게 나옴** (제일 위험한 실수)
- `y_test`가 `(10000,)` 1차원일 때 `argmax(axis=1)` → `AxisError`
- 둘 다 `axis=0`으로 바꾸면 → `ValueError: inconsistent numbers of samples`
- 결론: **sparse면 y_test는 그대로, 예측만 argmax**

### 이미지 스케일링 `(x - 127.5) / 127.5`
- 0~255 → **-1 ~ 1** 범위로 맞추는 스케일링

---

## 8. LSTM으로 이미지(MNIST) 읽기

> 이미지도 "줄 단위 시퀀스"로 보면 LSTM에 넣을 수 있다.

```python
x_train = x_train.reshape(-1, 28, 28)       # (N, timesteps=28, feature=28)

input1 = Input(shape=(28, 28))
lstm1 = LSTM(128)(input1)                   # (N, 128)
dense1 = Dense(512, activation='relu')(lstm1)
...
```

### 핵심 포인트
- **가로줄 28개 = 시점 28개**, 한 시점마다 그 줄의 **픽셀 28개를 한 번에** 읽는다
  - `(N, 784, 1)`처럼 픽셀을 하나씩 읽으면 시점이 784개라 **훨씬 느리고 앞부분을 잊기 쉽다**
- **스케일링을 꼭 켠다**
  - LSTM 내부의 `tanh` / `sigmoid`는 입력이 0~255처럼 크면 값이 끝에 몰려(**포화**) 학습이 거의 안 됨
- LSTM 출력이 2차원이라 뒤의 `Dense`는 기존 그대로 연결
- 함수형 모델: `Input → LSTM → Dense ... → Model(inputs, outputs)`

### LSTM 파라미터 계산
```
LSTM param = 4 × (units × (feature + units) + units)

LSTM(128), feature=28:
  4 × (128 × (28 + 128) + 128) = 80,384
```
- **4**: LSTM 내부 게이트(입력/망각/출력 + 후보) 4세트 → 각각 가중치 + bias
- **feature + units**: 현재 입력(feature) + 이전 시점 출력(units)이 함께 들어감
- 효과: 첫 Dense 파라미터가 `784×512+512 = 401,920` → `128×512+512 = 66,048`로 줄었다

---

## 9. Reshape 층 — 모델 안에서 shape 바꾸기

### (1) Dense는 "마지막 축"에만 적용된다
```python
model.add(Dense(280, input_shape=(28, 28)))      # (N, 28, 28) → (N, 28, 280)
# param = 28 × 280 + 280 = 8,120
```
- 가로줄 28개마다 **같은 가중치**로 280칸을 만든다
- 3차원 입력을 Dense에 넣어도 에러가 아님 (마지막 축만 변환)

### (2) Reshape로 CNN에 연결
```python
model.add(Reshape(target_shape=(28, 28, 10)))    # (N, 28, 280) → (N, 28, 28, 10)
model.add(Conv2D(32, (3,3), activation='relu'))
```
- Conv2D는 **4차원**이 필요 → `Reshape`로 맞춘다
- **값의 개수가 같아야** 바꿀 수 있다: `28 × 280 = 28 × 28 × 10 = 7,840`
- **순서만 바꾸는 층 → 파라미터 0**
- Conv2D 파라미터 (채널이 10): `(3×3×10 + 1) × 32 = 2,912`

### (3) 데이터 reshape 대신 모델 안에서 4차원 만들기
```python
model.add(Reshape(target_shape=(28, 28, 1), input_shape=(28, 28)))   # (N,28,28) → (N,28,28,1)
```
- 데이터에 `reshape(-1, 28, 28, 1)`을 하는 대신, **모델 첫 층에서** 변환 (param 0)

### (4) CNN → LSTM → CNN 연결 (shape 왕복)
```python
model.add(Conv2D(32, (5,5), activation='relu'))           # (N, 20, 20, 32)
model.add(Reshape(target_shape=(20*20, 32)))              # 4차원 → 3차원 (N, 400, 32)
model.add(LSTM(10, return_sequences=True))                # (N, 400, 10)
model.add(Reshape(target_shape=(20, 20, 10)))             # 3차원 → 4차원 (N, 20, 20, 10)
model.add(Conv2D(64, (3,3), activation='relu'))
```

| 단계 | shape | 의미 |
|---|---|---|
| Conv2D 출력 | (N, 20, 20, 32) | 이미지 |
| Reshape | (N, 400, 32) | 픽셀 400개 = **시점 400개**, 채널 32 = **feature** |
| LSTM(`return_sequences=True`) | (N, 400, 10) | 시점마다 출력 유지 → **3차원 유지** |
| Reshape | (N, 20, 20, 10) | 다시 Conv2D용 4차원 (400 = 20×20) |

- **`return_sequences=True`**: 모든 시점의 출력을 내보내 3차원 유지 (기본값 False는 마지막 시점만 → 2차원)
- LSTM 출력(3차원)을 바로 Conv2D에 넣으면:
  `ValueError: ... expected min_ndim=4, found ndim=3` → **Reshape로 되돌려야** 한다
- 파라미터 계산 (feature = 앞 Conv2D의 **채널 수 32**):
  - LSTM: `4 × (10 × (32 + 10) + 10) = 1,720`
  - 뒤 Conv2D: `(3×3×10 + 1) × 64 = 5,824` (입력 채널이 32 → 10으로 바뀜)

---

## 10. 오늘의 핵심 요약 (한눈에)

1. 텍스트 분류 흐름: **데이터 로드 → 길이 확인 → `pad_sequences` → Embedding → LSTM → Dense**
2. 길이는 **평균보다 약간 크게** `maxlen`, 기본 pre(앞 채움·앞 자름)가 LSTM에 유리
3. **Embedding이 정수 2차원 → 3차원**으로 바꿔 LSTM 입력을 만들어 준다
4. `Embedding(input_dim, output_dim)` 파라미터 = `input_dim × output_dim`, `input_dim`은 **최대 번호 + 1**
5. 다중분류: **softmax + 원핫 + categorical_crossentropy** / 이진분류: **sigmoid + 0/1 + binary_crossentropy**
6. 평가 변환: 다중은 `argmax`, 이진은 `round`
7. `EarlyStopping(restore_best_weights)` + `ModelCheckpoint(save_best_only)`로 최고 모델 확보, `save_weights`는 **가중치만**
8. 클래스가 많으면 **sparse_categorical_crossentropy**로 원핫 생략 → **`y_test`에 argmax 하지 말 것**
9. LSTM 입력은 **(N, timesteps, feature)**, 이미지는 "가로줄 = 시점"으로 읽을 수 있고 **스케일링 필수**
10. `LSTM param = 4 × (units × (feature + units) + units)`
11. `Reshape`는 **값 개수가 같을 때만**, **param 0**, CNN(4차원) ↔ LSTM(3차원) 사이를 이어 준다
12. LSTM 뒤에 이어 붙일 층의 차원 요구에 따라 `return_sequences`를 정한다

---

## 11. 복습 체크리스트

- [ ] Reuters와 IMDB의 분류 종류(다중/이진)와 출력층·loss 차이를 말할 수 있다
- [ ] `num_words`와 `Embedding`의 `input_dim` 관계를 설명할 수 있다
- [ ] `pad_sequences`의 기본값(pre)이 LSTM에 왜 유리한지 설명할 수 있다
- [ ] Embedding → LSTM → Dense 사이의 shape 변화를 말할 수 있다
- [ ] Embedding 파라미터 수를 계산할 수 있다
- [ ] `ModelCheckpoint`와 `save_weights`의 차이를 안다
- [ ] sparse 라벨을 쓸 때 평가 코드에서 무엇을 바꿔야 하는지 안다 (`y_test` argmax 금지)
- [ ] LSTM 입력 3차원이 `(N, timesteps, feature)`임을 안다
- [ ] LSTM 파라미터 수를 계산할 수 있다
- [ ] `Reshape`의 조건(값 개수 동일)과 `return_sequences=True`의 의미를 설명할 수 있다

> ※ 업로드한 파이썬 파일(keras62~65)의 코드와 주석을 바탕으로 정리한 내용입니다. 실제 강의에서 강조된 부분과 다를 수 있으니, 수업 때 들은 내용에 맞게 보완해 주세요.
