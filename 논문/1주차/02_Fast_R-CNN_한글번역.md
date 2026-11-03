# Fast R-CNN (고속 R-CNN)

- **원제**: Fast R-CNN
- **저자**: Ross Girshick (Microsoft Research)
- **학술대회**: IEEE International Conference on Computer Vision (ICCV) 2015
- **코드 저장소**: [https://github.com/rbgirshick/fast-rcnn](https://github.com/rbgirshick/fast-rcnn)
- **원문 링크**: [arXiv:1504.08083](https://arxiv.org/abs/1504.08083)

---

## 초록 (Abstract)

본 논문에서는 객체 검출(Object Detection)을 위한 **Fast R-CNN**(Fast Region-based Convolutional Network) 방식을 제안한다. Fast R-CNN은 심층 합성곱 신경망(Deep ConvNet)을 활용하여 객체 후보 영역(Object Proposals)을 효율적으로 분류하는 이전 연구들을 기반으로 구축되었다. 이전 연구들과 비교하여, Fast R-CNN은 검출 정확도를 높이는 동시에 학습 및 테스트 속도를 크게 개선하기 위해 몇 가지 혁신적인 기법을 도입하였다.

Fast R-CNN은 매우 깊은 신경망인 **VGG16**을 기준으로:
- 기존 R-CNN 대비 **학습 속도는 9배**, **테스트 속도는 213배** 빠르며, PASCAL VOC 2012에서 더 높은 mAP(Mean Average Precision)를 달성한다.
- SPPnet(Spatial Pyramid Pooling Network)과 비교해도 VGG16 학습 속도는 **3배**, 테스트 속도는 **10배** 빠르며, 검출 정확도 역시 더 우수하다.

Fast R-CNN은 Python과 C++(Caffe 프레임워크)로 구현되었으며, 오픈소스 MIT 라이선스로 공개되어 있다.

---

## 1. 서론 (Introduction)

최근 심층 합성곱 신경망(Deep ConvNets)은 이미지 분류(Image Classification)와 객체 검출(Object Detection)의 정확도를 비약적으로 향상시켰다. 이미지 분류와 비교할 때 객체 검출은 훨씬 더 복잡한 해결 방식이 요구되는 까다로운 작업이다. 이러한 복잡성으로 인해 기존의 접근법들(예: R-CNN, SPPnet 등)은 다단계 파이프라인(Multi-stage Pipeline)으로 모델을 학습시켰으며, 이는 느리고 비효율적이었다.

객체 검출이 복잡한 주된 이유는 객체의 **정확한 위치 추정(Accurate Localization)**이 필요하기 때문이며, 이로 인해 두 가지 주요 도전 과제가 발생한다:
1. 수많은 후보 객체 위치(이른바 **"영역 제안, Proposals"**)를 처리해야 한다.
2. 이러한 후보 위치들은 대략적인 위치 정보만을 제공하므로, 정밀한 위치를 얻기 위해 이를 **미세 정제(Refinement)**해야 한다.

이러한 문제들을 해결하려는 기존 방식들은 종종 속도, 정확도, 단순성 사이에서 타협해야 했다.

본 논문에서는 최신 ConvNet 기반 객체 검출기의 학습 과정을 대폭 간소화한다. 우리는 **객체 제안 영역의 분류(Classification)와 공간 위치 정제(Bounding-box Regression)를 단일 단계에서 결합 학습(Jointly Learn)하는 단일 단계(Single-stage) 학습 알고리즘**을 제안한다.

그 결과 VGG16과 같은 초심층 네트워크를 R-CNN 대비 **9배**, SPPnet 대비 **3배** 빠르게 학습시킬 수 있다. 런타임 시 검출 네트워크는 이미지당 0.3초(영역 제안 추출 시간 제외) 만에 이미지를 처리하면서, PASCAL VOC 2012에서 **66% mAP**의 최고 수준 정확도를 달성한다 (R-CNN은 62%).

---

### 1.1. R-CNN과 SPPnet의 한계

기존 **R-CNN**은 딥러닝을 객체 제안 분류에 적용하여 획기적인 검출 정확도를 달성했으나 다음과 같은 치명적인 단점들이 있었다:

1. **다단계 파이프라인(Multi-stage Pipeline) 학습**:
   - 1단계: 객체 제안 영역들에 대해 로그 손실(Log loss)로 ConvNet을 미세 조정(Fine-tuning).
   - 2단계: ConvNet에서 추출한 특징(Features)을 바탕으로 클래스별 선형 SVM을 학습하여 소프트맥스(Softmax) 분류기를 대체.
   - 3단계: 경계 상자 회귀기(Bounding-box Regressor)를 별도로 학습.
2. **학습에 막대한 공간과 시간이 소모됨**:
   - SVM과 경계 상자 회귀기를 학습하기 위해 각 이미지의 모든 객체 제안 영역에서 특징 벡터를 추출하여 하드 디스크에 저장해야 했다.
   - VGG16 네트워크의 경우 VOC 2007 trainval 세트(5,000장) 특징 추출에만 GPU로 2.5일이 걸렸으며, 수백 기가바이트(GB)의 저장 공간이 필요했다.
3. **객체 검출(추론) 속도가 극도로 느림**:
   - 테스트 시 각 테스트 이미지의 모든 후보 영역(이미지당 약 2,000개)마다 개별적으로 ConvNet의 순전파(Forward pass)를 수행해야 하므로 계산을 공유하지 못한다. VGG16 기준으로 GPU에서 이미지당 **47초**가 소요되었다.

**SPPnet(Spatial Pyramid Pooling Network)**은 계산을 공유함으로써 R-CNN의 속도를 높이고자 제안되었다. SPPnet은 전체 입력 이미지에 대해 합성곱 특징 맵(Convolutional Feature Map)을 단 한 번만 계산한 다음, 공유된 특징 맵에서 각 객체 제안 영역에 해당하는 특징 벡터를 추출한다. 제안 영역 내부의 특징 맵을 고정 크기 출력(예: $6 \times 6$)으로 맥스 풀링(Max-pooling)한 뒤, 이를 다중 크기로 풀링하여 연결(Concatenate)하는 방식이다. SPPnet은 테스트 속도를 10~100배 가속했고, 학습 시간도 3배 단축했다.

그러나 **SPPnet 역시 뚜렷한 한계**가 존재했다:
- R-CNN과 마찬가지로 특징 추출 $\rightarrow$ 네트워크 미세 조정 $\rightarrow$ SVM 학습 $\rightarrow$ 바운딩 박스 회귀 학습으로 이어지는 다단계 파이프라인이며, 특징을 디스크에 기록해야 했다.
- 가장 결정적으로, SPPnet의 미세 조정 알고리즘은 **공간 피라미드 풀링(Spatial Pyramid Pooling) 이전의 합성곱 계층(Convolutional layers)들을 업데이트할 수 없었다**. 합성곱 계층을 고정(Freeze)시켜야만 했기 때문에 초심층 네트워크(Deep networks)의 잠재적인 성능 향상을 제한했다.

---

### 1.2. 본 연구의 기여 (Contributions)

우리는 R-CNN과 SPPnet의 단점을 극복하고 속도와 정확도를 모두 개선한 새로운 학습 알고리즘인 **Fast R-CNN**을 제안한다. Fast R-CNN의 주요 장점은 다음과 같다:

1. R-CNN, SPPnet보다 **더 높은 검출 품질(mAP)** 달성
2. 다중 작업 손실(Multi-task Loss)을 사용한 **단일 단계(Single-stage) 학습**
3. **네트워크의 모든 계층(All network layers)을 역전파로 업데이트 가능**
4. 특징 저장을 위한 **디스크 캐싱(Disk caching)이 전혀 불필요**

---

## 2. Fast R-CNN 아키텍처 및 학습

Fast R-CNN의 아키텍처는 전체 이미지와 일련의 객체 제안 영역(Object Proposals)을 입력으로 받는다:
1. 전체 이미지를 여러 합성곱(Conv) 및 맥스 풀링(Max pooling) 계층으로 처리하여 **합성곱 특징 맵(Conv Feature Map)**을 생성한다.
2. 각 객체 제안 영역에 대해 **RoI 풀링 계층(Region of Interest Pooling Layer)**이 특징 맵에서 고정된 크기(예: $7 \times 7$)의 특징 벡터를 추출한다.
3. 각 특징 벡터는 일련의 완전 연결 계층(Fully Connected Layers, FCs)으로 공급되어 최종적으로 두 개의 형제 출력 계층(Sibling Output Layers)으로 분기된다:
   - $K+1$개 클래스(배경 클래스 포함)에 대한 사후 확률 추정치 $p = (p_0, \dots, p_K)$를 출력하는 **소프트맥스(Softmax) 계층**
   - $K$개 클래스 각각에 대해 정제된 4개의 경계 상자 좌표 오프셋 $t^k = (t_x^k, t_y^k, t_w^k, t_h^k)$을 출력하는 **경계 상자 회귀 계층**

```
[입력 이미지] ───────────────────► [기저 ConvNet] ──► [전체 Conv 특징 맵]
                                                          │
[영역 제안(RoIs)] ────────────────────────────────────────┤
                                                          ▼
                                                  [RoI Pooling 계층]
                                                          │
                                                          ▼
                                                   [FC 계층들]
                                                          │
                                         ┌────────────────┴────────────────┐
                                         ▼                                 ▼
                             [Softmax 분류기 (K+1)]           [Bounding Box 회귀기 (4×K)]
```

---

### 2.1. RoI 풀링 계층 (The RoI Pooling Layer)

RoI 풀링 계층은 유효한 관심 영역(RoI) 내부의 특징 맵 영역을 고정된 $H \times W$(예: $7 \times 7$)의 작은 특징 맵으로 변환하기 위해 맥스 풀링을 사용한다. 여기서 $H, W$는 계층 하이퍼파라미터이며 임의의 $(h \times w)$ RoI와 무관하게 고정된다.

- 각 RoI는 4개 튜플 $(r, c, h, w)$로 정의되는 사각형 윈도우다. 여기서 $(r, c)$는 좌상단 모서리 좌표, $(h, w)$는 높이와 너비다.
- RoI 맥스 풀링은 크기가 $h \times w$인 RoI 윈도우를 대략 크기가 $h/H \times w/W$인 $H \times W$개의 서브 윈도우 그리드로 나눈 뒤, 각 서브 윈도우 내의 값들을 최대 풀링하여 해당 출력 그리드 셀에 매핑한다.
- 이는 SPPnet에서 사용된 공간 피라미드 풀링의 단일 스케일 특수 사례에 해당한다.

---

### 2.2. 사전 학습된 네트워크 초기화 (Initializing from Pre-trained Networks)

ImageNet으로 사전 학습된 네트워크(예: AlexNet, VGG16)를 Fast R-CNN으로 변환할 때 3가지 변형을 가한다:
1. 마지막 공간 풀링 계층(예: VGG16의 `pool5`)을 **RoI 풀링 계층**으로 교체하며, 이때 출력 크기 $H \times W$는 네트워크의 첫 번째 완전 연결 계층과 호환되도록 설정한다(예: $7 \times 7$).
2. 네트워크의 마지막 완전 연결 계층 및 소프트맥스(기존 1,000개 클래스용)를 **두 개의 형제 계층(Sibling Layers)**으로 교체한다:
   - $K+1$개 출력을 갖는 완전 연결 계층(소프트맥스)
   - $4K$개 출력을 갖는 범주별 경계 상자 회귀 계층
3. 네트워크가 2개의 입력을 받도록 수정한다: **이미지 목록**과 **해당 이미지들의 RoI 목록**.

---

### 2.3. 검출을 위한 미세 조정 (Fine-tuning for Detection)

역전파(Back-propagation)를 통해 네트워크의 모든 가중치를 학습할 수 있다는 점이 Fast R-CNN의 핵심 역량이다.

#### 계층적 미니배치 샘플링 (Hierarchical Mini-batch Sampling)
SPPnet이 합성곱 계층을 업데이트하지 못한 근본적인 원인은 역전파가 RoI 풀링을 통과할 때 매우 비효율적이었기 때문이다. 각 RoI가 서로 다른 이미지에서 오면 수용 영역(Receptive field)이 전체 이미지를 아우르므로 계산량이 지나치게 컸다.

Fast R-CNN은 학습 시 **계층적 샘플링(Hierarchical Sampling)**을 제안한다:
- 각 미니배치마다 먼저 $N$개의 이미지를 샘플링한다 (기본값 $N = 2$).
- 그런 다음 각 이미지에서 $R/N$개의 RoI를 샘플링한다 (총 $R = 128$개 RoI, 즉 이미지당 64개).
- 동일한 이미지에서 나온 RoI들은 순전파와 역전파 시 합성곱 연산과 메모리를 공유하므로, $N=2, R=128$일 때의 학습 속도는 128장의 서로 다른 이미지에서 1개씩 샘플링하는 것보다 약 **64배** 빠르다.

샘플링 전략:
- 각 이미지에서 Ground Truth(GT) 경계 상자와의 IoU(Intersection over Union)가 **$0.5$ 이상**인 제안 영역에서 $25\%$($32$개)를 전경(Foreground, $u \ge 1$)으로 샘플링한다.
- IoU가 **$[0.1, 0.5)$** 구간에 있는 제안 영역들을 배경(Background, $u = 0$)으로 나머지 $75\%$($96$개)를 채운다.
- 데이터 증강으로는 수평 뒤집기(Horizontal flip, $p=0.5$)만 사용한다.

---

#### 다중 작업 손실 (Multi-task Loss)
Fast R-CNN은 분류와 경계 상자 회귀를 동시에 최적화하는 다중 작업 손실 $L$을 사용한다.

각 RoI에 대해:
- 정답 클래스 라벨: $u \in \{0, 1, \dots, K\}$ (배경은 $u=0$)
- 예측된 클래스 확률 분포: $p = (p_0, \dots, p_K)$
- 정답 경계 상자 오프셋: $v = (v_x, v_y, v_w, v_h)$
- 클래스 $u$에 대해 예측된 경계 상자 오프셋: $t^u = (t_x^u, t_y^u, t_w^u, t_h^u)$

총 손실 함수:
$$L(p, u, t^u, v) = L_{cls}(p, u) + \lambda [u \ge 1] L_{loc}(t^u, v)$$

여기서:
- **분류 손실**: $L_{cls}(p, u) = -\log p_u$ (음의 로그 우도)
- 아이버슨 괄호(Iverson bracket): $[u \ge 1]$은 $u \ge 1$(전경 객체)일 때 1이고, 배경($u=0$)일 때는 0이다. 즉, 배경에 대해서는 위치 오차를 계산하지 않는다.
- 하이퍼파라미터 $\lambda = 1$로 설정하여 두 작업 간의 균형을 맞춘다.

**경계 상자 위치 손실($L_{loc}$)**은 견고한(Robust) **Smooth $L_1$ 손실**을 사용한다:
$$L_{loc}(t^u, v) = \sum_{i \in \{x, y, w, h\}} \text{smooth}_{L_1}(t_i^u - v_i)$$

$$\text{smooth}_{L_1}(x) = \begin{cases} 0.5 x^2 & \text{if } |x| < 1 \\ |x| - 0.5 & \text{otherwise} \end{cases}$$

> **Smooth $L_1$을 사용하는 이유**:
> R-CNN이나 SPPnet에서 사용한 $L_2$ 손실은 오차가 클 때 기울기(Gradient)가 폭발할 위험이 있어 학습률을 매우 세심하게 튜닝해야 했다. Smooth $L_1$은 $|x| < 1$에서는 $L_2$처럼 매끄럽고, $|x| \ge 1$에서는 기울기가 $\pm 1$로 고정되어 이상치(Outlier)에 덜 민감하고 학습이 훨씬 안정적이다.

---

#### RoI 풀링 계층의 역전파 (Back-propagation Through RoI Pooling)
RoI 풀링 계층의 입력 $x_i \in \mathbb{R}$에 대한 편미분은 다음과 같이 계산된다:

$$\frac{\partial L}{\partial x_i} = \sum_r \sum_j [i = i^*(r, j)] \frac{\partial L}{\partial y_{rj}}$$

- $y_{rj}$는 $r$번째 RoI의 $j$번째 서브 윈도우 출력 값이다.
- $i^*(r, j) = \arg\max_{i' \in \mathcal{R}(r, j)} x_{i'}$는 해당 서브 윈도우 $\mathcal{R}(r, j)$에서 최대값을 기록했던 입력 위치(Argmax index)를 의미한다.
- 즉, 순전파 시 최댓값으로 선택되었던 입력 $x_i$에 대해서만 해당 출력 $y_{rj}$의 상위 기울기 $\frac{\partial L}{\partial y_{rj}}$가 누적 전달된다.

---

## 3. Fast R-CNN 검출 (추론 가속)

테스트 단계에서 Fast R-CNN은 입력 이미지에 대해 약 2,000개의 RoI를 평가한다. 대부분의 추론 시간은 완전 연결(FC) 계층 계산에서 소모된다.

### 잘린 SVD(Truncated SVD)를 통한 가속
완전 연결 계층의 가중치 행렬 $W$ ($u \times v$)는 잘린 SVD(Truncated Singular Value Decomposition)를 통해 다음과 같이 근사 분해할 수 있다:
$$W \approx U \Sigma_t V^T$$
- $U$: $u \times t$ 행렬 (좌측 특이 벡터)
- $\Sigma_t$: $t \times t$ 대각 행렬 ($t$개의 최상위 특이값)
- $V$: $v \times t$ 행렬 (우측 특이 벡터)

이를 적용하면 파라미터 수가 기존 $u \times v$에서 $t(u + v)$로 줄어든다. $t < \min(u, v)$일 때 계산량이 대폭 감소한다.
신경망 구현에서는 단일 FC 계층을 비선형 활성화 함수가 없는 2개의 연속된 FC 계층으로 분할 교체하여 구현하며, 검출 정확도의 손실 거의 없이 **추론 속도를 30% 이상 가속**한다.

---

## 4. 주요 실험 결과 (Main Results)

세 가지 모델 구조를 통해 검증하였다:
- **모델 S**: CaffeNet (AlexNet 변형, 5 conv + 2 fc)
- **모델 M**: VGG_CNN_M_1024
- **모델 L**: VGG16 (13 conv + 3 fc)

### 4.1. PASCAL VOC 정확도 (mAP %)
VOC 2007 테스트 세트 결과:

| 방식 | 사전 학습 모델 | mAP (%) |
| :--- | :--- | :---: |
| R-CNN | VGG16 | 66.0 |
| SPPnet | VGG16 | 63.1 |
| **Fast R-CNN** | **VGG16** | **66.9** |
| **Fast R-CNN (VOC07+12 학습)** | **VGG16** | **70.0** |

Fast R-CNN은 단일 단계 학습임에도 불구하고 R-CNN 및 SPPnet의 성능을 능가하였다.

### 4.2. 학습 및 테스트 시간 비교

| 방식 | 학습 시간 (시간) | 속도 비율 | 테스트 시간 (초/이미지, 제안 영역 제외) | 테스트 속도 비율 |
| :--- | :---: | :---: | :---: | :---: |
| R-CNN (VGG16) | 84 h | $1\times$ | 47.0 s | $1\times$ |
| SPPnet (VGG16) | 25 h | $3.4\times$ | 2.3 s | $20\times$ |
| **Fast R-CNN (VGG16)** | **9.5 h** | **$8.8\times$** | **0.32 s** | **$146\times$** |
| **Fast R-CNN + SVD** | **9.5 h** | **$8.8\times$** | **0.22 s** | **$213\times$** |

- 학습 시간: R-CNN의 84시간에서 **9.5시간**으로 단축.
- 추론 시간: R-CNN의 47초에서 **0.22초(SVD 적용 시)**로 **213배 가속**.
- 특징 데이터를 디스크에 저장하지 않으므로 수백 GB의 디스크 여유 공간 유지.

---

## 5. 설계 분석 및 연구 (Design Evaluation)

1. **다중 작업 학습(Multi-task Training)의 유효성**:
   - 분류와 경계 상자 회귀를 개별 단계로 학습시키는 것보다 동시에 다중 작업 손실로 학습시키는 것이 mAP를 0.8~1.1%p 일관되게 향상시켰다.
2. **모든 합성곱 계층을 미세 조정해야 하는가?**:
   - SPPnet처럼 합성곱 계층을 고정(Freeze)하고 FC 계층만 학습시키면 VGG16에서 mAP가 61.4%로 떨어진다.
   - `conv3_1` 이상의 상위 합성곱 계층들을 함께 미세 조정할 때 최고 성능(66.9%)을 달성할 수 있었다.
3. **Softmax vs. One-vs-Rest SVM**:
   - Fast R-CNN의 내장 소프트맥스 분류기는 별도의 사후 SVM 학습 대비 약간 더 높은 성능(mAP +0.1~0.8%p)을 보였다. 따라서 복잡한 SVM 다단계 학습은 더 이상 불필요하다.
4. **제안 영역(Proposal)이 많을수록 좋은가?**:
   - 제안 영역 수를 무작정 늘린다고(예: 1,000개 $\rightarrow$ 10,000개) 정확도가 계속 오르지 않으며, 오히려 약간 감소하거나 정체된다.

---

## 6. 결론 (Conclusion)

Fast R-CNN은 R-CNN과 SPPnet의 구조적 비효율성을 근본적으로 해결한 깔끔하고 빠른 단일 단계(Single-stage) 객체 검출 프레임워크를 정립하였다. RoI 풀링을 통한 특징 맵 공유, 다중 작업 손실(Smooth $L_1$ + Cross Entropy), 역전파를 통한 전 계층 종단간 학습(End-to-End Learning)을 통해 속도와 정확도 두 마리 토끼를 모두 잡았다.

그러나 Fast R-CNN에서도 **여전히 남아있는 병목**은 CPU 상에서 실행되는 영역 제안 알고리즘(Selective Search, 이미지당 약 2초 소요)이었으며, 이는 후속 연구인 **Faster R-CNN**의 RPN(Region Proposal Network)으로 이어지게 된다.
