# LLM 8일차 정리

> 녹취(음성 인식 결과)와 코드(`VGG16` 전이학습 + 동결)를 바탕으로 **파이썬 공부 위주**로 정리한 노트입니다.
> 녹취에 잡음이 많아, 문맥상 뚜렷한 부분만 정리했고 불확실한 부분은 `(추정)`으로 표시했습니다.

## 오늘의 큰 흐름

| 구분 | 내용 |
|---|---|
| 랭체인 / RAG | 유사도 임계값(threshold) 조정, RAG의 3요소(모델·프롬프트·리트리버), 하이브리드 검색 |
| 딥러닝 | 가중치 동결(freeze), 전이학습(Transfer Learning), 사전학습 모델(Pre-trained), `include_top=False` |
| 실습 코드 | VGG16 동결 후 Dense 층 붙이기, 레이어 구조 확인, 부분 동결, 여러 모델 반복 실험 |
| 면접 포인트 | Batch Norm vs Layer Norm, Attention vs Multi-Head Attention, ReLU/BN 순서 |
| 공지 | 내일 아침 시험, 프로젝트 발표 양식, 타임테이블, 데이터 수집 근거 |

---

## 1. 랭체인 / RAG 복습

### 1-1. 검색이 약하면 "답변할 수 없습니다"가 나온다
- 리트리버가 문서를 잘 못 가져오면 LLM은 컨텍스트가 없으니 **"답변할 수 없습니다"**를 반복한다.
- 원인: 벡터에 넣은 정보가 너무 기초적이어서 유사도가 낮게 나옴.
- 먼저 확인할 것: 문서의 **`metadata`** 와 **`page_content`** 가 제대로 들어갔는지.

### 1-2. 유사도 임계값(score_threshold) 조정
- 임계값 `0.7` → 너무 높아서 아무 문서도 안 걸림.
- `0.5`, `0.3` 까지 낮춰서 확인. 데이터가 적을 때는 결과가 비슷하게 나올 수 있음.

```python
# 유사도 임계값을 건 리트리버 (예시)
retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"score_threshold": 0.5, "k": 4},
)
docs = retriever.invoke("질문")
for d in docs:
    print(d.metadata, d.page_content[:100])
```

### 1-3. RAG에 필요한 것은 딱 3개
```
모델(LLM)  +  프롬프트(Prompt)  +  리트리버(Retriever)
```
- 항상 이 3가지가 핵심이다.
- 과거에는 프롬프트 엔지니어링을 극한으로 했지만, **모델 버전이 올라갈수록 자연어로 편하게 지시해도 잘 알아듣는다.**
- 프롬프트를 "어떻게 넣을까"보다, 모델이 잘 알아듣도록 **자연스럽게 지시**하는 게 점점 더 강력해지는 중.

### 1-4. 하이브리드 검색 (웹 검색 + LLM)
- 웹 검색 도구(DuckDuckGo 등 `(추정)`)를 연결해서 검색 결과를 끌어오는 방식 vs 검색 없이 모델 자체 지식만 쓰는 방식을 비교.
- 검색을 붙이면 오히려 **시간이 늘어나고**, 모델 자체 지식이 더 나은 경우도 있어서 애매한 상황.
- 결론: 검색이 꼭 필요한 질문인지 판단해서 붙이는 게 좋다.

---

## 2. 딥러닝 핵심 개념 — 가중치 동결과 전이학습

### 2-1. 모든 것은 가중치에서 시작
- **학습** = 가중치를 갱신하는 것 (순전파 → 손실 계산 → **역전파**로 가중치 업데이트)
- **추론** = 가중치를 **동결(freeze)** 한 상태에서 **순전파만** 하는 것
- 파라미터가 160억 개쯤 되면 순전파도 느리다 → 그래서 순전파를 빠르게 하는 기술(양자화 등)이 나옴.
- **양자화(Quantization)**: 가중치를 압축하듯 줄이는 것. 예) 8GB 메모리 기기(온디바이스)에 모델이 안 들어갈 때 양자화해서 넣음.

### 2-2. `trainable` 이란?
- `trainable=True` : 역전파로 **갱신되는** 가중치
- `trainable=False` : **동결** → 갱신 안 됨, 역전파 안 됨
- 남이 잘 만든 모델의 가중치를 가져다 쓸 때 동결한다.

### 2-3. 동결, 언제 좋고 언제 나쁜가?
- 가중치가 **내 데이터에 잘 맞을 때** → 동결하면 시간 절약 + 성능 좋음
- 안 맞을 때 → 동결하지 않고 **재학습(fine-tuning)** 하는 게 더 나을 수 있음
- 수업 중 실험 결과 두 경우가 모두 나왔다:
  - "동결한 게 내 모델보다 성능이 좋다" → 내가 힘들게 만든 것보다 가져다 쓴 게 나은 경우가 실제로 있음
  - "동결하지 않고 재학습한 게 더 좋다" → 이것도 있음
- **비교 기준은 항상 "내가 만든 최고 점수"**. VGG16이 그것을 넘는지, 동결 시/비동결 시 각각 어떤지 확인.

### 2-4. 용어 정리 (현업 커뮤니케이션용)

| 용어 | 의미 |
|---|---|
| **Transfer Learning (전이학습)** | 이미 학습된 모델을 가져와 내 문제에 맞게 사용 |
| **Pre-trained Model (사전학습 모델)** | 대규모 데이터로 미리 학습된 모델 (예: ImageNet으로 학습된 VGG16) |
| **Fine-tuning (미세조정)** | 가져온 모델을 내 데이터로 **추가 학습**하는 것 (넓게는 재학습 전체를 가리키기도 함) |
| **Feature Extraction (특징 추출)** | 사전학습 모델의 가중치를 **그대로 가져다 쓰는 것** (동결해서 특징만 뽑음) |

- 실무에서는 "피처 익스트랙션으로 가져다 써" = 가중치 그대로 쓴다는 뜻으로 통하는 경우가 많다.
- 용어는 시험용이 아니라 **선임/동료와 소통할 때 쓰는 말**이라고 생각하자.

### 2-5. `include_top=False` — 위/아래를 내 데이터에 맞게 교체

VGG16 원형 구조: **Conv 13개 + Pooling + Fully Connected(Dense) 3개**, 입력 `224x224x3`, 출력 1000 클래스(ImageNet)

내 데이터가 `100x100` 이미지의 **2클래스 분류**(예: 말/사람)라면?
- **위(입력) shape**를 내 이미지 크기로 바꿔야 한다 → `input_shape=(...)`
- **아래(출력) Dense**를 내 클래스 수로 바꿔야 한다 → `Dense(2)`
- 즉, **가운데 특징 추출부(Conv 블록)만 가져오고** 위/아래는 직접 구성한다. → `include_top=False`
- Keras에서는 FC 부분은 `include_top=False`로 빠지지만, **다른 곳에서 받은 가중치 파일(.h5 등)은 직접 잘라야** 한다. (실무 포인트)
- 트랜스포머(Hugging Face) 모델도 마찬가지로 **입력부(Input)와 출력부(Output)만 내 문제에 맞게 커스터마이징**한다.
  - 예) 로또 번호 예측: 입력은 지금까지의 데이터, 출력은 번호 6개 + 보너스 = 7개

### 2-6. VGG16 최소 입력 크기와 정보 소실
- VGG16이 받을 수 있는 최소 입력은 **32x32x3**.
- Conv 블록의 Pooling을 5번 거치면 `32 → 16 → 8 → 4 → 2 → 1`, 마지막에는 **1x1** 로 줄어든다.
- 1x1까지 줄어들면 **정보가 거의 소실** → 분별력이 떨어질 수 있다.
- MNIST, Fashion MNIST(28x28 흑백)는 이 최소 조건에 안 맞아서 VGG16 실습에서 제외했다.
- 모델마다 최소 입력 크기가 다르다 (참고, 대략): VGG/ResNet/MobileNetV2/DenseNet ≈ 32, Xception ≈ 71, InceptionV3 ≈ 75.
  - 그래서 작은 이미지(CIFAR 등)에서는 **Xception, InceptionV3 가 안 돌아간다** (수업에서도 제외).

---

## 3. 오늘의 코드 — VGG16 동결 + Dense 붙이기

```python
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.applications import VGG16

vgg16 = VGG16(
    weights='imagenet',      # ImageNet으로 사전학습된 가중치 사용
    include_top=False,       # 위(분류기, FC 3개) 제거 → 내가 직접 구성
    input_shape=(32,32,3),   # 내 이미지 크기 (VGG16 최소 32x32x3)
)
vgg16.trainable = False      # 동결: VGG16 전체 가중치 학습 안 함

model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(10, activation='softmax'))   # 10클래스 분류

print(len(model.weights))            # 30
print(len(model.trainable_weights))  # 4
```

### 3-1. 가중치 개수가 왜 30개일까?
```
VGG16 Conv 13개 × 2 (w, b)  = 26
Dense 2개       × 2 (w, b)  =  4
                              ----
                               30   ← model.weights
```
- `Flatten`은 가중치가 없다.
- `trainable_weights`는 **동결 안 된 것만** → Dense 2개 × 2 = **4개**
- `weights - trainable_weights = 26` → 동결된 VGG16 부분

### 3-2. 레이어 구조를 DataFrame으로 예쁘게 출력

```python
import pandas as pd
pd.set_option('max_colwidth', None)

layers = [
   (type(layer).__name__, layer.name, layer.trainable)
   for layer in model.layers
]
results = pd.DataFrame(
    layers, columns=['Layer Type', 'Layer Name', 'Layer Trainable']
)
print(results)
```

```
   Layer Type Layer Name  Layer Trainable
0  Functional      vgg16            False
1     Flatten    flatten             True
2       Dense      dense             True
3       Dense    dense_1             True
```

- `model.layers` 는 **리스트**다 → 리스트니까 `for` 문으로 돌 수 있다.
- `vgg16` 은 모델 안에 모델이 들어간 형태라 **Functional 한 덩어리(레이어 1개)** 로 보인다.
- 리스트 컴프리헨션 + `pd.DataFrame(..., columns=[...])` 로 표 만들기 패턴을 익혀두자.

### 3-3. 예상 파라미터 수 (참고 — `model.summary()` 로 확인)
```
VGG16 (Conv 부분)     14,714,688   (동결 → Non-trainable)
Flatten 출력          1x1x512 = 512
Dense(100)            512*100 + 100 = 51,300
Dense(10)             100*10 + 10   = 1,010
-> Trainable params   52,310
```
- 입력이 32x32라 Flatten 결과가 512가 된다. 입력이 커지면 Flatten 크기도 커져서 Dense 파라미터가 확 늘어난다.

### 3-4. 코드 개선 팁
- `Dense(100)` 에 `activation` 이 없으면 선형이라 `Dense(10)` 과 합쳐도 사실상 한 층과 같은 효과 → **`activation='relu'`** 를 넣는 게 보통이다. (수업에서도 "식성에 따라 ReLU 넣어도 된다"고 함)
- `Flatten` 대신 **`GlobalAveragePooling2D`** 를 써도 된다 (파라미터 수가 줄어듦).
- 라벨이 정수(0~9)면 `loss='sparse_categorical_crossentropy'`, 원-핫이면 `'categorical_crossentropy'`. 예측 시에는 `argmax` 를 취한다.

---

## 4. 특정 레이어만 동결 (부분 동결)

### 4-1. 인덱스로 접근
```python
# model.layers 의 인덱스는 0부터 시작! (Flatten, Dense 등 모든 층 포함)
model.layers[2].trainable = False      # 3번째 층만 동결
model.summary()
```
- 인덱스는 **0부터** 시작한다는 것 — 수업에서 몇 번 헷갈린 부분.
- 동결할 층 번호를 바꿔가며 어디가 동결되는지 `summary()` 나 위 DataFrame으로 확인해 보기.

### 4-2. 범위로 동결
```python
for layer in vgg16.layers[:-4]:   # 마지막 블록(block5)만 빼고 모두 동결
    layer.trainable = False
```

### 4-3. 어느 부분을 동결하나?
- **입력에 가까운 앞쪽(저수준 특징: 선, 모서리 등)** → 범용적이라 동결해도 잘 맞는 경우가 많음.
- **뒤쪽 블록** → 내 데이터에 특화된 특징이라 **재학습**하는 경우가 많음.
- 수업 말씀: "임의로 층을 골라서 동결했는데 성능이 좋은 경우는 별로 못 봤다." 정답은 없으니 실험으로 확인해야 한다.
- 가중치가 좋다고 **판단될 때** 동결하고, 확신이 없으면 **해 보고 안 되면 재학습**.

### 4-4. 층을 붙이는 두 가지 방식
```python
# (A) 단순 붙이기 : 레이어를 그대로 쌓음 → 각 레이어를 개별로 동결
# (B) 모델 통째로 넣기 : VGG16을 하나의 모델처럼 add → vgg16.trainable 한 줄로 동결
model.add(vgg16)
vgg16.trainable = False
```
- (B)가 훨씬 간단하다. 쓰다 보면 편한 방식으로 쓰면 된다.
- 참고: `compile()` **전에** `trainable` 을 바꿔야 적용된다. 바꾼 후에는 다시 `compile()`.

---

## 5. 여러 모델 반복 실험 (for 문 + 리스트)

모델 이름을 **리스트**에 담아서 `for` 로 돌리면, 한 번에 여러 사전학습 모델을 비교할 수 있다.

```python
import warnings
warnings.filterwarnings('ignore')     # 경고 메시지 숨기기

from keras.applications import VGG16, VGG19, ResNet50, MobileNetV2, DenseNet121
from keras.layers import GlobalAveragePooling2D

model_list = [VGG16, VGG19, ResNet50, MobileNetV2, DenseNet121]

for ModelClass in model_list:
    base = ModelClass(weights='imagenet', include_top=False, input_shape=(32,32,3))
    base.trainable = False                       # 동결 (False) / 재학습 (True) 비교 실험

    model = Sequential([
        base,
        GlobalAveragePooling2D(),
        Dense(100, activation='relu'),
        Dense(10, activation='softmax'),
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=5, batch_size=128,
              validation_split=0.2, verbose=0)

    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f'{base.name}: loss={loss:.4f}, acc={acc:.4f}')
```

### 핵심 포인트
- **`include_top=False` 를 모델마다 넣어야** 한다. 안 넣으면 위(shape), 아래(Dense 1000) 때문에 shape 에러가 난다.
  - 포문 안에서 `ModelClass(...)` 로 호출하면 한 번에 해결 (각각 따로 쓰지 않아도 됨).
- `evaluate()` 의 반환값은 `[loss, accuracy]` 라서 `[0]` 은 loss, `[1]` 은 accuracy.
- `verbose=0` 으로 두면 훈련 로그가 안 나오고, 마지막 결과만 깔끔하게 출력된다.
- 이미지 값은 **스케일링**(`/255.`)하거나 모델별 `preprocess_input` 사용.
- 같은 가중치를 받는 과정은 **처음 한 번만 다운로드**하고, 이후에는 로컬에 저장된 걸 쓴다.
- 데이터 크기가 큰 문제(개/고양이, 남/여 등)는 메모리 터짐에 주의 → 배치 크기 줄이기.
- 실험은 **동결(`False`) → 비동결(`True`)** 순서로 비교하고, **내 모델의 최고 점수와 비교**한다.
  - 수업 중 확인된 사실: CIFAR 같은 작은 이미지에서는 **VGG16이 가장 약한 편**이었고, 요즘은 대부분 ResNet 계열 이상을 쓴다.
  - 작은 입력 이미지로 인해 정보가 소실된 영향도 있다.

### 경고(warning) 끄기
```python
import warnings
warnings.filterwarnings('ignore')
```
- 대부분의 경고가 사라지지만 **전부 막히지는 않는다** (예: `tf.placeholder` 경고는 TF1 문법에서 오는 것).
- 에러보다 경고가 더 무서운 경우도 있으니, 경고는 **원인을 해결**하는 게 나은 경우가 많다.

### `numpy` 는 왜 import 하나?
- 지금 코드에서 직접 안 써도 pandas 와 TensorFlow 가 내부적으로 numpy 기반이고 변환이 잦아서, 보통 같이 넣어둔다.

---

## 6. 모델 구조 관련 지식 & 면접 질문

### 6-1. 모델 계보
- 2017년 전후까지는 이미지 분야가 거의 **Convolution 기반** (VGG, ResNet 등) → 이후 트랜스포머 등장.
- **ResNet**: Skip Connection(점프)이 핵심 → 트랜스포머도 이 아이디어를 가져다 썼다.
- 요즘은 **Global Average Pooling** 을 많이 쓴다.

### 6-2. 면접에서 자주 나오는 질문
1. **Batch Normalization vs Layer Normalization** 의 차이 (거의 필수 질문)
   - Batch Norm: 배치 방향으로 정규화 (CNN에서 주로)
   - Layer Norm: 한 샘플의 특징 방향으로 정규화 (트랜스포머에서 주로)
2. **Attention vs Multi-Head Attention** 의 차이
3. **BN 다음에 ReLU vs ReLU 다음에 BN**
   - 원 논문은 `Conv → BN → ReLU`. ReLU 뒤에 BN을 쓰는 건 엄밀히는 논문과 순서가 다른데, 이 순서를 왜 선택했는지 물어보는 회사도 있다.
   - 이런 걸 물어보는 회사는 **실력이 있는 회사**라고 보면 된다.
- BN, Layer Norm 은 쓰기만 해도 **성능이 확 올라가는** 강력한 도구라서 꼭 이해해 두자.

### 6-3. Batch Normalization(BN)이란?

**한 줄 요약:** 각 층의 출력을 **미니배치 단위로 평균 0, 분산 1에 가깝게 맞춰 주는 층**. 층을 지날 때마다 값의 분포가 들쭉날쭉해지는 문제를 줄여서 학습을 안정시킨다.

#### 동작 방식
1. 배치 안에서 특징(채널)별로 평균 μ 와 분산 σ² 를 구한다.
2. 정규화한다: `x̂ = (x - μ) / √(σ² + ε)`
3. 학습 가능한 파라미터 γ(스케일), β(이동)로 다시 조정한다: `y = γ·x̂ + β`

- 3번 덕분에 정규화가 모델의 표현력을 해치지 않는다 (필요하면 γ, β 가 원래 분포로 되돌릴 수도 있음).

#### 효과
- 학습이 안정되고 수렴이 빨라진다.
- 학습률을 더 크게 줘도 된다.
- 약한 규제(overfitting 완화) 효과도 있다.

#### 왜 보통 ReLU 앞에 오나? (`Conv → BN → ReLU`)
- 정규화된 값(0 주변 분포)이 ReLU 에 들어가야 활성화가 고르게 일어난다는 아이디어.
- 다만 BN 을 ReLU **뒤**에 두는 구성도 실제로 쓰이고, 성능 차이는 모델/데이터마다 다르다.
- 수업에서 말한 "논문과 순서가 다르다"는 면접 질문에는 **"논문 원형은 이 순서인데, 저는 이런 이유로 이렇게 썼다"** 고 답할 수 있으면 충분하다.
- (녹취의 BN/ReLU 순서 설명은 인식이 불분명해서 위 내용으로 보충함)

#### Keras 코드
```python
from keras.layers import Conv2D, BatchNormalization, Activation

x = Conv2D(64, (3,3), padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
```

Sequential 로 쓰는 경우:
```python
model.add(Conv2D(64, (3,3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
```

#### 알아둘 점
- **학습 시**에는 현재 배치의 평균/분산을 쓰고, **추론 시**에는 학습 중 쌓아 둔 **이동평균**을 쓴다. 그래서 배치 크기가 너무 작으면 불안정하다.
- **Layer Normalization** 은 배치가 아니라 **한 샘플 안의 특징 방향**으로 정규화한다. 배치 크기에 영향을 받지 않아서 트랜스포머에서 주로 쓴다. → 면접 단골 질문이 바로 이 차이.

| 구분 | Batch Norm | Layer Norm |
|---|---|---|
| 정규화 기준 | 배치 방향 (같은 채널의 배치 전체) | 한 샘플의 특징 전체 |
| 배치 크기 의존 | 있음 (작으면 불안정) | 없음 |
| 주로 쓰는 곳 | CNN | 트랜스포머, RNN 계열 |
| 추론 시 | 학습 때 쌓은 이동평균 사용 | 샘플별로 그대로 계산 |

---

## 7. 진로 / 학습 방향 (수업 중 강조된 내용)

- **"기승전 PyTorch"**: 앞으로 AI 쪽은 PyTorch 가 핵심이다. TensorFlow 문법을 알면 PyTorch 는 **차이점 위주**로 금방 배운다 (오전 보강으로 진행 예정).
- 반도체/CUDA/C++ 쪽 수요가 많다. GPU 병렬 처리(CUDA)를 요구하는 채용도 있다.
- 면접에서 모르는 기술을 물어보면 "**기간을 주시면 그 기간 안에 익히겠다**" 식으로 정리해 두면 좋다.
- 바이브 코딩이 늘어난 지금은, **원천 기술(직접 구현·모델 이해)** 을 아는 사람과 그렇지 않은 사람의 차이가 점점 커진다.
- **연봉 기준선**을 미리 정해서 지원할 것 (수업에서는 최소 기준선 이야기가 있었음).

---

## 8. 프로젝트 / 공지 메모

- **내일 아침 시험** 있음 → 앞부분(동결, 전이학습, 레이어 구조 출력, 반복문 모델 비교) 복습.
- **프로젝트 아이디어 발표** (내일 목요일 `(추정)`):
  - 팀별 **연장자 1명**이 메일로 제출.
  - 형식은 자유: 어떤 **데이터**로, 어떤 **모델**을, 어떻게 만들어서, 어떤 **결과/타겟 기업**을 노릴지.
  - 반드시 **타임테이블**(언제부터 언제까지 무엇을 할지) 포함.
  - 아이디어에는 **근거**가 있어야 한다 (왜 이 문제인지, 데이터를 어떻게 모을지).
- 최종 발표 때는 **핵심 기술 파트**(핵심 클래스/함수, 아키텍처)를 정확히 설명할 수 있어야 한다.
- **산출물은 팀원 수만큼, 각자 이름이 맨 앞에 들어간 것**으로 준비 (본인 취업용 포트폴리오).
- 데이터를 **어렵게 모았다는 과정(수집 노력)** 자체가 가점 요소.
- 평가 포인트: ① 대회/프로젝트 성과 ② 데이터 수집의 타당성 ③ 모델·아키텍처 ④ 핵심 기술 설명.
- 팀 프로젝트 중 **서로 싸우지 말 것.**

---

## 9. 오늘의 체크리스트 (복습)

- [ ] `vgg16.trainable = False` 의 의미와 `model.weights` / `model.trainable_weights` 개수 계산을 설명할 수 있다
- [ ] `include_top=False` 를 쓰는 이유(입력 shape, 출력 Dense 교체)를 말할 수 있다
- [ ] `model.layers` 로 레이어 이름/타입/`trainable` 을 DataFrame 으로 출력할 수 있다
- [ ] 레이어 인덱스(0부터)로 부분 동결을 할 수 있다
- [ ] 모델 리스트 + `for` 문으로 여러 사전학습 모델을 비교하는 코드를 짤 수 있다
- [ ] Transfer Learning / Pre-trained / Fine-tuning / Feature Extraction 의 차이를 말할 수 있다
- [ ] Batch Norm 이 무엇이고(평균·분산 정규화 + γ, β) 왜 `Conv → BN → ReLU` 순서로 쓰는지 설명할 수 있다
- [ ] Batch Norm vs Layer Norm 의 차이를 설명할 수 있다
- [ ] RAG 에서 `score_threshold` 를 낮추는 이유와 방법을 말할 수 있다
- [ ] 모델·프롬프트·리트리버 3요소를 설명할 수 있다
