# LLM 22일차 정리 — 트랜스포머 논문 발표 (Attention Is All You Need)

> 녹취 기반 정리. 잡담·출석·간식 얘기는 빼고 **트랜스포머 이론, 발표 Q&A, 퓨리오사 NPU(RNGD) 실습 공유, 면접 대비 포인트** 위주로 정리함.
> 녹취 인식이 부정확한 고유명사(모델명·툴명)는 `(녹취 불명확)`으로 표시.

---

## 0. 오늘의 흐름

| 순서 | 내용 |
|---|---|
| 1 | 면접 후기 공유 (AI 연구소 신입 면접) |
| 2 | 트랜스포머 발표 ① 팀 브레인 — T5-small 파인튜닝 로또 예측 |
| 3 | 특별 발표 — 퓨리오사 RNGD 1장에서 비공식 모델 컴파일·서빙 |
| 4 | 트랜스포머 발표 ② 팀 피지컬 — Add & Norm 질문에서 리젝 |
| 5 | 트랜스포머 발표 ③ 오이너스 팀 — 숫자 뒤집기 + 로또, LayerNorm vs BatchNorm에서 리젝 |
| 6 | 트랜스포머 발표 ④ 묘안 팀 — T5 영→독 번역, SOTA 질문에서 리젝 |
| 7 | 트랜스포머 발표 ⑤ 로보틱스 팀 — 인코더 전용 + Pre-LN + Sigmoid/BCE 로또 |
| 8 | 다음 주 과제 안내 (논문 3개, Reflection 등 에이전트 논문) |

**핵심 메시지:** *"AI 하려면 트랜스포머는 당연히 하는 거다."* 면접에서 트랜스포머 아키텍처 그림의 **모든 박스를 하나씩 물어본다**고 생각하고 준비할 것.

---

## 1. 기계 번역의 발전 흐름

```
RNN  →  Seq2Seq (고정 컨텍스트 벡터)  →  Seq2Seq + Attention (Bahdanau, 2014)  →  Transformer (2017, Google)
```

### 1-1. Seq2Seq의 한계 — 정보 병목(bottleneck)
- 인코더가 입력 문장을 순서대로 읽어 마지막 hidden state를 **고정 크기 컨텍스트 벡터 하나**로 압축 → 디코더에 전달
- 문장이 짧든 길든 벡터 크기는 같음 → **긴 문장일수록 정보 손실**
- 순차적으로 갱신하므로 **앞쪽 단어 정보가 희미해짐** (장기 의존성 문제)

> **Q. 컨텍스트 벡터가 고정 길이면 왜 문제인가?**
> 입력 길이와 무관하게 하나의 고정 벡터로 압축해야 해서 정보 손실이 생기고, 순차 갱신 과정에서 앞 단어 정보가 점점 흐려진다.

### 1-2. Attention (RNN + Attention)
- 인코더의 **최종 출력만이 아니라 모든 단계의 hidden state**를 사용
- 디코더가 단어를 하나 생성할 때마다
  1. 디코더 이전 hidden state와 인코더 각 hidden state를 비교 → **정렬 점수(alignment / energy)**
  2. softmax → **어텐션 가중치 α**
  3. α × 각 hidden state를 가중합 → **그 시점 전용 컨텍스트 벡터**
- 한계: 여전히 RNN 기반 → **앞 단어 계산이 끝나야 다음 계산** (순차 처리, 병렬화 불가)

### 1-3. Transformer
- RNN·CNN을 완전히 제거하고 **어텐션만으로** 문맥 학습
- 모든 단어가 다른 모든 단어와의 관계를 **한 번에** 계산 → **병렬 처리**로 학습 속도·성능 향상
- GPU 발전과 맞물려 각광받음

> **Q. Bahdanau 어텐션과 트랜스포머의 가장 큰 차이는?**
> 앞은 RNN 구조 위에 어텐션을 *보조 장치*로 붙인 것이고, 트랜스포머는 RNN을 제거하고 어텐션을 *유일한 시퀀스 혼합 수단*으로 쓴 것.

---

## 2. 트랜스포머 전체 구조

```
          [Encoder] × 6                         [Decoder] × 6
 Input Embedding + Positional Encoding    Output Embedding + Positional Encoding
        │                                        │
  Multi-Head Self-Attention              Masked Multi-Head Self-Attention
        │  Add & Norm                            │  Add & Norm
  Feed Forward (FFN)                     Cross-Attention (Q: 디코더, K·V: 인코더 출력)
        │  Add & Norm                            │  Add & Norm
        └──────── 인코더 최종 출력 ──────────→  Feed Forward
                                                 │  Add & Norm
                                           Linear → Softmax → 다음 단어 확률
```

- **인코더 레이어 = 서브레이어 2개**: Multi-Head Self-Attention, FFN
- **디코더 레이어 = 서브레이어 3개**: Masked Self-Attention, Cross-Attention(Encoder-Decoder Attention), FFN
- 논문은 각각 **N = 6층** 쌓음
- 모든 서브레이어의 입출력 차원 **d_model = 512**로 동일 (잔차 연결을 위해 필요)
- 인코더는 원문을 **한 번만 계산**해두고, 디코더가 단어를 생성할 때마다 그 출력을 **재사용**

---

## 3. Positional Encoding (위치 인코딩)

**왜 필요한가?** 한 번에 병렬로 보기 때문에 → 단어 **순서 정보를 모른다**. RNN은 순서대로 들어오니 위치를 자연히 알지만 트랜스포머는 따로 넣어줘야 함.

- 입력 임베딩 = 단어 **의미** 정보, 위치 인코딩 = 단어 **위치** 정보 → 둘을 **더해서** 어텐션 입력으로 사용
- 논문 방식: **짝수 차원 sin, 홀수 차원 cos**, 차원마다 주기가 다름

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right),\quad PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

- 학습형(learned) 위치 임베딩도 실험했는데 **성능 차이가 거의 없었음**
- 그럼에도 sin/cos를 택한 이유: **주기함수라서 학습 때보다 긴 문장이 들어와도 그대로 사용 가능**

```python
import torch, math

def positional_encoding(max_len: int, d_model: int) -> torch.Tensor:
    pe = torch.zeros(max_len, d_model)
    pos = torch.arange(0, max_len).unsqueeze(1).float()              # (max_len, 1)
    div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(pos * div)   # 짝수 차원
    pe[:, 1::2] = torch.cos(pos * div)   # 홀수 차원
    return pe                            # (max_len, d_model)

# x = token_embedding(ids) + positional_encoding(seq_len, d_model)
```

> 실험 결과(오이너스 팀): **숫자 뒤집기 과제(1,2,3 → 3,2,1)** 에서 위치 인코딩을 뺐더니 순서를 제대로 출력하지 못함 → 위치 인코딩이 순서 처리에 필수라는 걸 직접 확인.

---

## 4. Scaled Dot-Product Attention

$$\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

| 요소 | 의미 |
|---|---|
| **Query (Q)** | 현재 단어가 **무엇을 찾고 싶은지** (질문) |
| **Key (K)** | 다른 단어들과의 **관련성을 판단하는 기준** |
| **Value (V)** | 실제로 **가져올 정보** |

**계산 순서**
1. 입력 X에 서로 다른 가중치 W_Q, W_K, W_V를 곱해 Q, K, V 생성
2. Q · Kᵀ (내적) → **Attention Score** (단어 간 연관도)
3. √d_k로 나눔 (**스케일링**)
4. (디코더라면 마스킹)
5. softmax → **Attention 가중치**
6. 가중치 × V → 가중합 → 문맥이 반영된 새 표현(Z)

### 왜 √d_k로 나누나? (면접 단골)
- d_k가 커지면 내적값의 **분산이 커짐** → softmax 입력값이 커져 **한쪽으로 쏠림(포화)**
- 포화되면 **기울기가 0에 가까워져 학습이 안 됨** → √d_k로 나눠 분산을 안정화

### 왜 V(Value)가 따로 필요한가? (오늘 선생님 질문)
- Q·K만으로는 "**얼마나 볼지(가중치)**"만 나옴. 실제로 **가져올 내용**은 따로 있어야 함
- "관계를 판단하는 표현(K)"과 "전달할 정보(V)"를 **분리**해서 각자 다른 선형 변환으로 학습 → 표현력이 커짐
- 가중치를 V에 곱해 "이 단어 정보를 이만큼 반영한다"를 구현하는 것

> ⚠️ 발표 시 용어 주의: "벡터 행렬과 곱한다" ❌ → "**Value 행렬**과 곱한다" ⭕ (그냥 '벡터'라고 하면 오해 소지)

```python
import torch
import torch.nn.functional as F

def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.size(-1)
    scores = Q @ K.transpose(-2, -1) / d_k ** 0.5           # (B, h, L_q, L_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))  # 소프트맥스 직전에 마스킹
    attn = F.softmax(scores, dim=-1)                        # 행 합 = 1
    return attn @ V, attn
```

---

## 5. Multi-Head Attention

- 어텐션을 한 번만 하지 않고 **h개를 병렬로** 수행 (논문 h = 8, 각 head 차원 = 512/8 = 64)
- 각 head마다 **서로 다른 W_Q, W_K, W_V** → 서로 다른 관점(문법 관계, 의미 관계 등)으로 학습
- 비유: 문제 하나를 **전문가 여러 명이 같이 보고 논의**하는 것
- 모든 head 출력을 **Concat → W_O(선형 변환)** → 최종 출력

> **Q. 왜 헤드를 8개로?** 논문에서 여러 개수를 실험했는데 8과 16일 때가 가장 좋았고, 논문은 8을 채택. (head가 너무 많으면 head당 차원이 작아져 오히려 성능 하락)
> **Q. 왜 하나만 안 쓰나?** 하나로 하면 여러 관계가 **평균**되어 의미가 흐려짐.

```python
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=512, n_heads=8):
        super().__init__()
        self.h, self.d_k = n_heads, d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        B = q.size(0)
        split = lambda x: x.view(B, -1, self.h, self.d_k).transpose(1, 2)  # (B, h, L, d_k)
        Q, K, V = split(self.W_q(q)), split(self.W_k(k)), split(self.W_v(v))
        out, _ = scaled_dot_product_attention(Q, K, V, mask)
        out = out.transpose(1, 2).contiguous().view(B, -1, self.h * self.d_k)  # Concat
        return self.W_o(out)
```

---

## 6. 트랜스포머 안의 어텐션 3종류

| 종류 | 위치 | Q | K, V | 역할 |
|---|---|---|---|---|
| **Encoder Self-Attention** | 인코더 | 인코더 입력 | 인코더 입력 | 입력 문장 단어끼리 서로 참고해 문맥 파악 |
| **Masked Decoder Self-Attention** | 디코더 1번째 | 디코더 입력 | 디코더 입력 | 미래 단어를 못 보게 막고 지금까지의 출력 문맥만 사용 |
| **Cross-Attention** (Encoder-Decoder) | 디코더 2번째 | **디코더** | **인코더 출력** | 다음 단어 생성 시 원문(인코더 출력)을 참고 |

### Masking — 왜 −∞ 인가? (오늘 집중 질문)
- 학습 시 미래 단어를 보면 **정답지를 보고 문제 푸는 것**과 같음 → 미래 위치를 가림
- **소프트맥스 직전**에 적용: 마스킹하려면 먼저 연관도 점수가 계산되어야 하고, 소프트맥스는 상대적 확률을 만들기 때문
- softmax(−∞) = e^(−∞) / Σ = **정확히 0** → 미래 단어 가중치가 완전히 0
- −1 같은 유한값을 넣으면 e^(−1) ≈ 0.37 → **여전히 확률이 남아 미래 정보가 새어 나감**

```python
L = 5
causal_mask = torch.tril(torch.ones(L, L))   # 하삼각 = 볼 수 있는 위치
# [[1,0,0,0,0],
#  [1,1,0,0,0], ...]  0인 곳을 -inf로 채움
```

> 📌 **softmax는 "정규화(normalization)" 함수다** — 출력의 총합을 1로 만들어 확률분포로 바꿈. 선생님이 "소프트맥스 뭐라고 해?" 했을 때 정답이 **정규화**였음. (활성화 함수로 분류되기도 하지만 핵심 키워드는 '정규화')

---

## 7. Add & Norm, FFN

### 7-1. Add = 잔차 연결 (Residual Connection)
- 서브레이어 입력 x를 출력에 **그대로 더함**: `x + Sublayer(x)`
- 이유: 6층 × 서브레이어 여러 개로 깊어짐 → **역전파 시 기울기 소실 방지**, 학습 안정화, 원래 정보(과거 정보) 보존
- 그래서 모든 서브레이어의 차원이 512로 같아야 함 (더하려면 shape이 같아야)
- **ResNet**에서 처음 접한 개념 → 면접 단골. 아키텍처 그림에서 **입력에서 갈라져 서브레이어를 우회해 Add로 들어가는 화살표**가 residual

### 7-2. Norm = Layer Normalization ⭐ (오늘 두 팀 탈락 원인)

$$\text{LayerNorm}(x) = \gamma \cdot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$$

- 한 샘플(한 토큰) 안의 **feature 차원 방향**으로 평균 0, 분산 1로 맞춘 뒤 γ, β로 스케일/시프트 학습

### LayerNorm vs BatchNorm — "100% 나오는 면접 질문"

| | **Batch Normalization** | **Layer Normalization** |
|---|---|---|
| 정규화 축 | **배치 방향** (같은 feature를 여러 샘플에 걸쳐) | **feature 방향** (한 샘플 내부의 모든 feature) |
| 배치 크기 의존 | 의존함 (작은 배치에서 불안정) | 의존 안 함 |
| 학습/추론 차이 | 있음 (추론 시 이동평균 통계 사용) | 없음 (항상 같은 계산) |
| 가변 길이 시퀀스 | 패딩·길이 차이 때문에 다루기 어려움 | 토큰마다 독립 계산이라 자연스러움 |
| 주 사용처 | CNN (이미지) | RNN, **Transformer** (NLP) |

```python
import torch, torch.nn as nn
x = torch.randn(32, 10, 512)                 # (batch, seq_len, d_model)

ln = nn.LayerNorm(512)                       # 마지막 차원(512) 기준
y_ln = ln(x)                                 # 각 토큰 벡터마다 평균0/분산1

bn = nn.BatchNorm1d(512)                     # (N, C, L) 형태를 기대
y_bn = bn(x.transpose(1, 2)).transpose(1, 2) # feature별로 배치·시퀀스 전체 통계 사용

print(y_ln[0, 0].mean().item(), y_ln[0, 0].std().item())  # ≈ 0, ≈ 1
```

> 🧭 선생님 예고: 토치 들어가기 직전쯤 **Normalization / Standardization / Regularization** 용어를 한 번 정리할 예정. 번역도 제각각(정규화·표준화·정칙화…)이라 **0~1 스케일링(min-max)** 인지, **평균 0·표준편차 1(표준화)** 인지 개념으로 구분할 수 있어야 함.

### 7-3. Post-LN vs Pre-LN
- 논문(원본): **Post-LN** → `LayerNorm(x + Sublayer(x))`
- 요즘 트렌드(GPT 등): **Pre-LN** → `x + Sublayer(LayerNorm(x))` — 깊은 모델에서 학습이 더 안정적
- 로보틱스 팀은 Pre-LN을 명시적으로 구현함

### 7-4. FFN (Position-wise Feed Forward)
- `FFN(x) = max(0, xW₁ + b₁)W₂ + b₂` → **선형 → ReLU → 선형** (512 → 2048 → 512)
- 어텐션은 본질적으로 **가중합(선형 결합)** 이라 비선형성이 약함 → FFN이 **비선형 변환** 담당
- 토큰마다 독립적으로(같은 가중치로) 적용 → "position-wise"
- ⚠️ 발표에서 "선형 변환 두 번"이라고만 하면 지적받음 → **중간에 ReLU가 있다**고 꼭 말할 것

```python
class EncoderLayer(nn.Module):
    def __init__(self, d_model=512, n_heads=8, d_ff=2048, p=0.1):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, n_heads)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model))
        self.ln1, self.ln2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.drop = nn.Dropout(p)

    def forward(self, x, mask=None):            # Post-LN (논문)
        x = self.ln1(x + self.drop(self.mha(x, x, x, mask)))   # Add & Norm
        x = self.ln2(x + self.drop(self.ffn(x)))               # Add & Norm
        return x
```

---

## 8. 트랜스포머의 장점과 한계

### 논문이 말하는 장점 3가지
1. 레이어당 **계산량**이 적음 (n < d인 일반적인 경우)
2. **순차 연산 수 O(1)** → 병렬화에 유리 (RNN은 O(n))
3. **최대 경로 길이가 짧음** (O(1)) → 장거리 의존성 학습에 유리

### 성능 (WMT 2014)
- 영→독: Base 27.3 BLEU, Big **28.4 BLEU** (기존 최고 대비 2 BLEU 이상 향상)
- 영→불: Big **41.x BLEU** (단일 모델 SOTA)
- 평가지표 **BLEU**(블루): 기계 번역 결과와 정답 번역의 n-gram 일치도

> 📌 **SOTA = State-Of-The-Art** (현 시점 최고 성능). "단일 모델 SOTA" = 앙상블 없이 단일 모델 기준 최고 기록. → 이걸 모델명/협회라고 답해서 리젝됨. 슬라이드에 쓴 용어는 **전부 설명할 수 있어야 함**.

### 한계
- Self-Attention 연산량이 시퀀스 길이에 대해 **O(n²)** → 문장이 길면 연산·메모리 급증
- 그래서 **트랜스포머가 항상 RNN보다 빠른 건 아님** (n이 d보다 매우 크면 오히려 느려질 수 있음). 다만 보통 n < d라서 통상적으로 빠름
- 디코더는 여전히 **한 단어씩 순차 생성(autoregressive)** → 출력을 한 번에 못 만듦

---

## 9. 팀별 실습 — 트랜스포머로 로또 예측

| 팀 | 방법 | 결과 |
|---|---|---|
| 브레인 | 1~1250회차, 지난 K회 → 다음 회를 **seq2seq(text-to-text)** 로 만들어 **T5-small 파인튜닝** | 평균 0.79~0.88개 적중 vs 무작위 0.8개 → **95% 기준 유의미한 차이 없음**. K=1,5,10,20 바꿔도 차이 없음 |
| 오이너스 | 트랜스포머 **직접 구현**, 지난 10회 입력(A) vs 정보 없음(B) | 둘 다 train loss↓ val loss↑ **과적합**. A가 오히려 loss 더 높음 → 과거 정보 효과 없음 |
| 묘안 | (로또 대신) Hugging Face **T5로 영→독 번역**. T5는 `"translate English to German: ..."` 같은 **작업 지시문(prefix)** 필요 | 일부 단어 오역 |
| 로보틱스 | **인코더만** 사용, 회차를 **45차원 multi-hot** (0/1), d_model 512→**128** 축소, **학습형 위치 임베딩**, **Pre-LN**, 출력 **Sigmoid + BCE loss**, 활성화 **GELU** | lookback 10~1198 비교, 무작위(0.8)와 대부분 오차범위 내. 200회 참조 시 1.255(테스트 8회뿐) |

**학습 포인트**
- 로또는 매 회 **독립 시행** → 이론상 학습할 패턴이 없음. 결과가 무작위와 통계적으로 같다는 결론이 "정답"
- **T5-small 선택 이유**: 인코더-디코더를 모두 가진 대표적 범용 text-to-text 모델, 데이터가 적어 작은 모델이 적합, 입력 길이 512
- **인코더만 쓴 이유**: 독립 회차라 "미래를 가리는" 디코더 마스킹이 의미 없음 → 분류 문제로 접근
- **Softmax 대신 Sigmoid**: 45개 번호 각각이 뽑힐지 말지 **독립적인 이진분류(multi-label)** 이므로 Sigmoid + BCE. (Softmax는 하나를 고르는 multi-class용)
- **시계열 분할 vs 랜덤 분할** 질문 → 시계열 데이터는 원칙적으로 시간 순 분할(미래 데이터 누설 방지)이지만, 회차가 독립이라고 가정하면 상관없다는 논리

> **Q. GELU를 쓴 이유가 뭐예요? (논문엔 ReLU)** — 하이퍼파라미터 선택 질문엔 이렇게 답한다:
> "ReLU, GELU, SELU, LeakyReLU를 비교 실험했고 그중 GELU가 가장 좋았습니다. 지면 관계상 GELU만 명시했습니다."
> → "그냥 써봤더니 좋았어요"가 아니라 **비교 근거**를 말할 것.

---

## 10. 특별 발표 — 퓨리오사 RNGD 1장에서 비공식 모델 컴파일·서빙

### 배경
- 퓨리오사 공식 지원 모델 중 RNGD 1~2장용은 11개 정도인데 **오래되었거나 작은 모델** 위주
- RNGD 1장 메모리 **48GB**를 다 못 쓰는 아쉬움 → 포럼 문의 결과 "27B급은 2장으로 컴파일해서 서빙하라"는 권장
- 1장으로 하려면 **직접 컴파일**해야 함

### 두 가지 컴파일·서빙 방식

| | ① SDK 빌드 + `furiosa-llm` 서빙 | ② Furiosa OPT + **vLLM 플러그인** |
|---|---|---|
| 수업에서 | 사용한 방식 | 새로 시도 중 |
| 모델 제한 | SDK 안의 **허용 모델 목록**에 있어야 컴파일 가능 | 오픈소스 스택 → 사용자가 원하는 모델 추가 가능 (단, 직접 모델 구현 코드 작성 필요) |
| 산출물 | 컴파일 아티팩트(EDF 등) | `.so` 파일 |
| 서빙 | `furiosa-llm serve` | vLLM이 요청 처리, 계산은 NPU가 담당 |
| 사용 언어 | Python 기반 | **Rust** 사용 |

### Qwen3-8B(녹취: "QN 3.8") 1장 컴파일 과정
1. 허용 모델 목록에 있는지 확인 → 있음
2. SDK 안의 **커널 생성 코드**를 찾음 → 기본이 **4장 기준**으로 작성되어 있음
3. 장수 설정값 4 → **1로 수정**, 에러를 하나씩 고쳐가며 컴파일
4. 1장에선 한 번에 컴파일하면 메모리가 터짐 → **커널을 하나씩(200여 개) 생성**
5. 서버에 설치된 정식 SDK는 건드리지 않고 **별도 환경에 SDK 설치 후 수정**
6. 시간: 멀티모달 모델은 이미지까지 하면 하루 이상 → 텍스트만 테스트

### Q&A에서 나온 개념
- **8B 올려도 메모리가 꽉 차는 이유**: 가중치는 얼마 안 차지만 남는 메모리를 **KV Cache 풀**로 채워 응답 속도를 높이기 때문. 27B(8bit 양자화 시 ~30GB)면 48GB 중 남은 ~18GB만 KV Cache로 사용
- **커널**: NPU가 읽을 수 있는 실행 파일 안에서 "어떤 계산을 할지" 정해주는 단위 (직관적 이해)
- **버킷(bucket)**: 기본 제공 버킷 안에 토큰을 넣고 나머지는 **패딩**으로 채움 → 버킷 개수·크기 조정이 최적화 포인트
- 퓨리오사가 11월 말~내년 초 일부를 **오픈소스화**해서 생태계를 키울 계획이라고 밋업에서 언급
- 서버의 NPU는 총 12장, 팀당 2장 → 2장 묶어 쓸 땐 **팀원에게 미리 공지**하고 작업

### 실습 도움 도구
- **tmux** (녹취: "Teamworks"로 추정): SSH 연결이 끊겨도 세션이 유지되고, 재접속 시 그 세션에 다시 붙어 작업 계속
  ```bash
  tmux new -s compile        # 세션 생성
  # (작업 실행 후) Ctrl+b → d  : 세션에서 빠져나오기(detach), 작업은 계속 돌아감
  tmux ls                    # 세션 목록
  tmux attach -t compile     # 다시 붙기
  ```
- **오픈소스 AI 에이전트 앱** (녹취: "DeepSick Harness", 녹취 불명확): 서빙한 모델의 **API 엔드포인트**를 붙여 채팅·파일 읽기·플러그인 테스트 가능

---

## 11. 면접 대비 — 오늘 나온 질문 모음 ✅

**트랜스포머 이론**
- [ ] Seq2Seq에서 컨텍스트 벡터가 고정 길이면 왜 문제인가?
- [ ] 어텐션(Bahdanau)과 트랜스포머의 가장 큰 차이는?
- [ ] 트랜스포머의 한계/병목은? (O(n²), 순차 디코딩)
- [ ] 트랜스포머가 항상 RNN보다 빠른가?
- [ ] 위치 인코딩이 왜 필요하고, 어떻게 계산하나? 왜 sin/cos인가?
- [ ] Q, K, V 각각의 의미는? **Q·K만 있어도 되는데 V를 따로 쓰는 이유는?**
- [ ] 왜 √d_k로 나누나?
- [ ] 마스킹에 왜 −1이 아니라 **−∞** 를 넣나? 어느 단계에서 하나?
- [ ] **softmax를 한 단어로 뭐라고 하나?** → 정규화(총합 1인 확률분포)
- [ ] 멀티헤드는 왜 쓰고, 왜 8개인가?
- [ ] **Add & Norm에서 Norm은 뭔가? LayerNorm vs BatchNorm 차이는?** ⭐⭐⭐
- [ ] **Add(잔차 연결)는 왜 하나? Residual block을 그림으로 그려보라** ⭐⭐
- [ ] FFN의 구조와 역할은? (선형-ReLU-선형, 비선형성)
- [ ] 평가지표 BLEU란? **SOTA란?**

**실습/선택 근거**
- [ ] 왜 T5-small을 골랐나?
- [ ] 왜 Sigmoid를 썼나? (multi-label 이진분류)
- [ ] 왜 GELU를 썼나? (→ 비교 실험 근거로 답하기)
- [ ] 데이터를 시계열로 나눴나, 랜덤으로 나눴나?

---

## 12. 선생님 조언 (발표·면접·취업)

**발표 태도**
- 질문자는 **나의 평가자**. 최대한 **공손하게**, 싸우지 말 것. 질문자가 틀린 걸 물어봐도 최악의 경우엔 맞다고 해줘야 함
- 질문이 애매하면 되묻지 말고 "네" 하고 명확히 답하기 (예: "Value 벡터"라고 정정해주면 그냥 수용)
- **대본은 외우지 말 것** → 슬라이드에 핵심 키워드만 두고 부연 설명
- 분량은 **10~15페이지**로 함축 (시간 초과 주의)
- 처음 앞에 서면 머리가 하얘짐 → **많이 해봐야 함**. 혼자 벽 보고 연습하는 것과 실제로 나와서 하는 건 차이가 큼

**발표 규칙 (내일부터 리젝 기준)**
- **PPT + PDF 둘 다** 제출, 메일에 **팀원 참조** + 맹 선생님 참조
- 발표자(팀장) 이름은 항상 **맨 왼쪽(팀장 위치)** 에
- 각 팀은 **다른 팀 발표를 듣고 질문할 사람을 1명 지정** (5,4,3,2,1 카운트 안에 질문 없으면 그 팀 재발표)
- 트랜스포머 발표를 한 번 더 하고 싶은 사람은 요청하면 6시 이후 개인 발표 가능

**면접 후기 (AI 연구소 신입)**
- 면접관 4명, 기술보다 **인성·태도 질문** 비중이 큼
- 질문 예: 팀 내 갈등 해결, AI로 만든 프로젝트 검증 방법, **기술/아키텍처 선택 이유**, 데이터 처리 방법, 왜 팀장을 했는지
- 포트폴리오 PDF를 넘겨가며 질문 → **포트폴리오에 쓴 건 전부 설명 가능해야 함**
- 신입 AI 직무는 GPU 서버(H100 등) **장비 세팅·관리**까지 할 줄 알아야 하는 경우 많음
- 면접 다녀오면 **시간 순서대로 후기 공유**하기 (문 열고 들어간 분위기 → 인원 → 질문 순서)

**취업 준비물 정리**
- 현재 준비된 것: **미니 프로젝트 1개, 논문 발표 2개**, 공모전(예선 탈락이라도) 참가 이력 → **전부 서류화**
- 팀 프로젝트는 **팀장으로 명시**, 다른 팀 PPT도 서로 **자료 공유**해서 재사용 (본인 이름으로 수정)
- 남은 기간 약 1달 → 12월 초 선생님 전체 컨펌 예정
- 지원은 **선생님이 시작하라고 할 때부터** (관심 있으면 메일)
- 해당 회사 지원 시 **PyTorch 먼저** 공부 권장
- 프로필 사진·정장 미리 준비 (갑자기 다음 날 면접 잡히는 경우 있음)

---

## 13. 다음 주 과제

- **논문 3개** 정리·발표 (이번 주보다 쉬운 편) — Reflection 등 **에이전트 관련 논문**
  - Reflection: 2023년경 모델 성능이 낮던 시절, 결과를 **여러 번 다시 돌려 스스로 고치게** 해서 품질을 올리는 기법 → 지금은 모델이 좋아져 한 번에 잘 나와서 실무 활용은 애매
  - 그래도 **ReAct, Tool Calling** 개념과 이어지므로 읽어둘 것
  - 아키텍처는 단순 → "이렇게 해보니 좋았다"를 표 몇 개 인용해서 설명하면 됨
- **잔차(Residual) 정리 페이퍼** 제출 (로보틱스 팀)
- 내일 재발표 팀은 발표 여부를 **메일로** 알릴 것
- 앞으로 매주 **논문 1개씩** 포트폴리오에 추가
