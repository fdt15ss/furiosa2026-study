# Faster R-CNN: 영역 제안 네트워크를 통한 실시간 객체 검출
(Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks)

- **저자**: Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun (Microsoft Research)
- **학술대회**: Advances in Neural Information Processing Systems (NeurIPS) 2015 / IEEE TPAMI 2017
- **코드 저장소**: [https://github.com/rbgirshick/py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn)
- **원문 링크**: [arXiv:1506.01497](https://arxiv.org/abs/1506.01497)

---

## 초록 (Abstract)

최신 객체 검출 네트워크들은 후보 객체 위치를 추정하기 위해 영역 제안(Region Proposal) 알고리즘에 의존하고 있다. SPPnet과 Fast R-CNN 같은 진보된 모델들은 탐지 네트워크의 실행 시간을 대폭 줄였으나, CPU 기반으로 실행되는 **영역 제안 계산이 전체 파이프라인의 심각한 병목(Bottleneck)**으로 드러났다.

본 논문에서는 탐지 네트워크와 전체 이미지의 합성곱 특징(Convolutional Features)을 공유함으로써, **거의 비용 없이(Nearly Cost-free)** 영역 제안을 계산할 수 있는 **영역 제안 네트워크(Region Proposal Network, RPN)**를 소개한다. RPN은 각 위치에서 객체의 경계(Bounding Box)와 객체성 점수(Objectness Score)를 동시에 예측하는 완전 합성곱 신경망(Fully Convolutional Network, FCN)이다.

RPN은 고품질 영역 제안을 생성하도록 종단간(End-to-End) 학습되며, 생성된 제안들은 검출을 위해 Fast R-CNN에 전달된다. 우리는 RPN과 Fast R-CNN이 합성곱 특징을 공유하도록 교대 학습(Alternating Training) 기법을 사용하여 두 네트워크를 단일 네트워크로 통합하였다.

매우 깊은 **VGG16** 모델을 기준으로, 제안된 시스템은 단일 GPU에서 모든 단계(영역 제안 생성 포함)를 처리하면서 **초당 5프레임(5 fps, 이미지당 198ms)**의 실시간에 준하는 프레임 레이트를 달성하는 동시에, PASCAL VOC 2007 (73.2% mAP), 2012 (70.4% mAP), MS COCO 데이터셋에서 최고 수준의 객체 검출 정확도를 기록하였다. 소형 모델(ZF net)의 경우 **초당 17프레임(17 fps)**까지 도달한다.

---

## 1. 서론 (Introduction)

최근 객체 검출의 비약적인 발전은 영역 기반 합성곱 신경망(R-CNN 계열)과 영역 제안(Region Proposal) 방법론의 결합에 힘입은 바가 크다. Fast R-CNN 등은 심층 합성곱 계층의 계산을 공유함으로써 영역 분류 단계의 연산 비용을 매우 낮추었다.

그러나 Fast R-CNN의 추론 시간(이미지당 약 0.2초)을 고려할 때, 후보 영역을 추출하는 **선택적 탐색(Selective Search, SS)** 알고리즘은 CPU에서 **이미지당 약 2초**가 걸려 전체 시스템 시간의 90% 이상을 차지하게 되었다. EdgeBoxes 같은 빠른 제안 기법도 0.2초 정도가 소요되어 여전히 탐지 네트워크 자체보다 느리거나 맞먹는 수준이다. 또한 이러한 영역 제안 방식들은 고정된 수작업 엔지니어링 특징을 사용하므로 다운스트림 딥러닝 탐지기와 분리되어 있어 최적화에 한계가 있다.

본 논문에서는 영역 제안 생성을 심층 합성곱 신경망 내부로 통합하는 **Faster R-CNN**을 제안한다. 핵심 아이디어는 **Fast R-CNN의 합성곱 특징 맵을 영역 제안 생성을 위한 특징으로도 재사용(Re-use)**하는 것이다.

이를 실현하기 위해 합성곱 계층의 맨 위에 소수의 추가 합성곱 계층을 덧붙여 구성한 **RPN(Region Proposal Network)**을 도입한다. RPN은 정규 격자(Regular grid) 상의 각 위치에서 다양한 스케일(Scale)과 종횡비(Aspect ratio)를 갖는 경계 상자들을 동시에 회귀 예측한다.

이 과정에서 핵심적인 개념이 바로 **"앵커(Anchors)"**이다. 앵커는 다양한 크기와 비율의 기준 참조 상자(Reference Box) 역할을 하며, 이미지 피라미드나 필터 피라미드 없이도 단일 스케일 특징 맵 위에서 다양한 형태의 객체를 효과적으로 포착할 수 있게 해준다.

---

## 2. 관련 연구 (Related Work)

- **객체 제안 방법론(Object Proposal Methods)**: 슈퍼픽셀 그룹화 기반(Selective Search, CPMC) 방식과 슬라이딩 윈도우 기반(EdgeBoxes, BING) 방식이 널리 쓰였으나, 독립된 전처리 모듈로 동작하여 연산 부하가 컸다.
- **객체 검출 딥러닝 네트워크**: OverFeat은 슬라이딩 윈도우 방식으로 단일 물체 분류 및 위치 회귀를 수행했으나 다중 클래스 다중 객체 검출에서는 정확도가 낮았다. MultiBox는 제안 영역 생성을 시도했으나 합성곱 특징을 분류기와 공유하지 못했다.

---

## 3. Faster R-CNN 아키텍처

Faster R-CNN은 두 개의 모듈로 구성된 단일 통합 네트워크이다:
1. 영역을 제안하는 **심층 완전 합성곱 네트워크 (RPN)**
2. 제안된 영역을 활용하여 실제 객체를 분류하고 위치를 정제하는 **Fast R-CNN 탐지기**

RPN 모듈은 최근 신경망에서 널리 활용되는 **'주의 집중(Attention)' 메커니즘**과 같이 Fast R-CNN 모듈에게 "이미지의 어디를 보아야 하는지"를 지시하는 역할을 수행한다.

```
                  [입력 이미지]
                        │
                        ▼
             [공유 합성곱 계층 (Conv Layers)]
             (예: VGG-16의 conv1 ~ conv5_3)
                        │
         ┌──────────────┴──────────────┐
         ▼                             ▼
  [RPN (Region Proposal Net)]          │
         │ (영역 제안 RoI)             │ (특징 맵)
         └──────────────┬──────────────┘
                        ▼
                [RoI Pooling 계층]
                        │
                        ▼
                 [FC 분류 및 회귀]
                        │
         ┌──────────────┴──────────────┐
         ▼                             ▼
   [Softmax (클래스 분류)]      [Bounding Box 정제]
```

---

### 3.1. 영역 제안 네트워크 (Region Proposal Networks, RPN)

RPN은 임의 크기의 이미지를 입력으로 받아 사각형 형태의 객체 제안 영역들과 각각의 **객체성 점수(Objectness score, 배경이 아닌 객체일 확률)**를 출력한다.

#### 1) 동작 구조
- 마지막 공유 합성곱 계층에서 출력된 특징 맵 위를 $n \times n$ (본 논문에서는 **$3 \times 3$**) 크기의 작은 네트워크가 슬라이딩 윈도우 방식으로 훑는다.
- 각 슬라이딩 윈도우는 저차원 특징 벡터(ZF 모델은 256차원, VGG16은 512차원)로 매핑된 후 ReLU를 통과한다.
- 이 특징 벡터는 두 개의 병렬 형제 완전 연결(FC) 계층(실제로는 $1 \times 1$ Conv)으로 입력된다:
  - **$2k$개 출력의 분류 계층 (`cls`)**: 각 앵커가 객체인지 배경인지의 점수
  - **$4k$개 출력의 회귀 계층 (`reg`)**: 각 앵커의 4개 경계 상자 좌표 인코딩

#### 2) 앵커(Anchors)의 개념과 특성
각 슬라이딩 윈도우 위치마다 최대 $k$개의 후보 영역을 동시에 예측하는데, 이 $k$개의 기준 상자를 **앵커(Anchor)**라 부른다.
- 본 논문에서는 **3개 스케일**(영역 넓이: $128^2, 256^2, 512^2$)과 **3개 종횡비**(Aspect Ratio: $1:1, 1:2, 2:1$)를 조합하여 총 **$k = 9$개의 앵커**를 각 위치마다 사용한다.
- 크기가 $W \times H$인 특징 맵 전체에는 총 $W \times H \times k$개의 앵커가 존재한다 (예: $60 \times 40$ 특징 맵이면 약 21,600개 앵커).

> **앵커의 평행 이동 불변성 (Translation Invariance)**:
> 앵커와 앵커 기준 상대 좌표 예측 방식은 평행 이동 불변성을 갖는다. 이미지가 이동하면 제안 영역도 동일하게 이동하므로 모델 파라미터 수가 매우 적고(VGG16 기준 RPN 전용 파라미터 수 단 2.8M), 과적합 위험이 낮다.

---

### 3.2. 손실 함수 (Loss Function)

RPN 학습을 위해 각 앵커에 이진 클래스 라벨(객체인가 아닌가)을 부여한다:

- **양성 라벨(Positive, 전경 객체)** 조건:
  1. 어떤 Ground-Truth(GT) 상자와 가장 높은 IoU를 갖는 앵커 (중복 가능)
  2. 어떤 GT 상자와의 IoU가 **0.7 이상**인 앵커
- **음성 라벨(Negative, 배경)** 조건:
  - 모든 GT 상자와의 IoU가 **0.3 미만**인 앵커
- IoU가 0.3~0.7 사이인 앵커는 학습 목표에 기여하지 않고 무시한다.

RPN의 다중 작업 손실 함수:
$$L(\{p_i\}, \{t_i\}) = \frac{1}{N_{cls}} \sum_i L_{cls}(p_i, p_i^*) + \lambda \frac{1}{N_{reg}} \sum_i p_i^* L_{reg}(t_i, t_i^*)$$

- $i$: 미니배치 내 앵커의 인덱스
- $p_i$: 앵커 $i$가 객체일 예측 확률
- $p_i^*$: 정답 라벨 (양성 앵커면 1, 음성 앵커면 0)
- $t_i$: 예측된 경계 상자의 4개 파라미터화된 좌표 벡터
- $t_i^*$: 양성 앵커에 할당된 GT 상자의 좌표 벡터
- $L_{cls}$: 두 클래스(객체 vs 배경) 간의 로그 손실(Log loss)
- $L_{reg}$: Smooth $L_1$ 손실 함수: $L_{reg}(t_i, t_i^*) = \text{smooth}_{L1}(t_i - t_i^*)$
- $p_i^* L_{reg}$: 오직 양성 앵커($p_i^* = 1$)에 대해서만 회귀 손실을 활성화하고 배경은 제외
- 정규화 상수 $N_{cls} = 256$, $N_{reg} \approx 2,400$, 가중치 계수 $\lambda = 10$

#### 경계 상자 파라미터화 (Bounding Box Parameterization)
$$t_x = (x - x_a)/w_a, \quad t_y = (y - y_a)/h_a$$
$$t_w = \log(w/w_a), \quad t_h = \log(h/h_a)$$
$$t_x^* = (x^* - x_a)/w_a, \quad t_y^* = (y^* - y_a)/h_a$$
$$t_w^* = \log(w^*/w_a), \quad t_h^* = \log(h^*/h_a)$$
여기서 $x, y, w, h$는 예측 상자의 중심 좌표와 너비/높이이며, $x_a$는 앵커, $x^*$는 GT 상자를 나타낸다.

---

### 3.3. RPN과 Fast R-CNN의 특징 공유 학습 (4단계 교대 학습)

RPN과 Fast R-CNN은 서로 독립적으로 학습되면 각자의 합성곱 계층을 다르게 수정하게 된다. 본 논문에서는 두 모듈이 공통의 합성곱 계층을 공유할 수 있도록 **4단계 교대 학습(4-Step Alternating Training)**을 개발하였다:

1. **1단계: RPN 독립 학습**
   - ImageNet 사전 학습 가중치로 백본을 초기화하고, RPN을 종단간 미세 조정한다.
2. **2단계: Fast R-CNN 학습**
   - 1단계 RPN에서 생성된 제안 영역들을 사용하여 별도의 Fast R-CNN 네트워크를 학습한다 (이 단계의 백본 역시 ImageNet 가중치로 초기화). 이 시점까지는 두 네트워크가 합성곱 계층을 공유하지 않는다.
3. **3단계: RPN 미세 조정 (공유 계층 고정)**
   - 2단계 Fast R-CNN의 합성곱 계층 가중치로 RPN을 초기화하되, **공유 합성곱 계층은 학습을 동결(Freeze)**시키고 오직 **RPN 고유의 계층들만 미세 조정**한다.
4. **4단계: Fast R-CNN 미세 조정 (공유 계층 고정)**
   - 공유 합성곱 계층을 동결한 상태로, 3단계 RPN에서 새로 생성된 제안 영역을 입력받아 **Fast R-CNN의 완전 연결 계층들만 미세 조정**한다.

이 과정을 거치면 두 네트워크는 동일한 합성곱 특징 맵을 완벽하게 공유하는 단일 통합 네트워크를 형성하게 된다.

*(참고: 이후 공개된 구현체에서는 역전파 시 RPN 손실과 Fast R-CNN 손실을 하나로 결합하여 한 번에 역전파하는 **근사 결합 학습(Approximate Joint Training)**이 가능해져 학습 시간이 25~50% 추가 단축되었다.)*

---

## 4. 실험 결과 (Experiments)

### 4.1. PASCAL VOC 성능 비교 (VOC 2007 테스트 세트)

| 방법 | 영역 제안 기법 | 백본 모델 | 제안 수 | mAP (%) | 전체 추론 시간 (ms) |
| :--- | :--- | :--- | :---: | :---: | :---: |
| Fast R-CNN | Selective Search | ZF | 2,000 | 58.7 | 1,470 (SS: 1,280 + Net: 190) |
| Fast R-CNN | Selective Search | VGG16 | 2,000 | 66.9 | 1,830 (SS: 1,510 + Net: 320) |
| **Faster R-CNN** | **RPN (공유)** | **ZF** | **300** | **59.9** | **59 (RPN: 10 + Net: 49) $\rightarrow$ 17 fps** |
| **Faster R-CNN** | **RPN (공유)** | **VGG16** | **300** | **69.9** | **198 (RPN: 10 + Net: 188) $\rightarrow$ 5 fps** |
| **Faster R-CNN (07+12 학습)** | **RPN (공유)** | **VGG16** | **300** | **73.2** | **198 ms** |

**주요 시사점**:
1. **정확도 향상**: RPN 제안을 사용하는 Faster R-CNN(69.9%)이 Selective Search를 사용하는 Fast R-CNN(66.9%)보다 **+3.0%p 더 우수**하다. 딥러닝 기반 RPN이 수작업 방식보다 훨씬 질 높은 후보 영역을 제안하기 때문이다.
2. **속도 혁신**: 이미지당 2,000개의 제안 영역 대신 **상위 300개의 제안 영역**만으로도 더 높은 정확도를 얻었으며, RPN의 제안 생성 시간은 단 **10ms**에 불과하여 전체 검출 속도가 **실시간(5 ~ 17 fps)** 영역에 진입했다.

---

### 4.2. MS COCO 데이터셋 결과

MS COCO 검증 세트에서 Faster R-CNN은 Fast R-CNN 대비 큰 폭의 성능 향상을 기록하였다:
- Fast R-CNN (SS): mAP@0.5 = 39.3%
- Faster R-CNN (RPN): mAP@0.5 = **42.7%** (COCO 표준 mAP@[.5, .95] = 21.9%)

대규모의 다양한 객체 크기를 가진 MS COCO에서도 RPN의 다중 스케일 앵커 구조가 탁월한 일반화 성능을 발휘함을 입증하였다.

---

## 5. 결론 (Conclusion)

Faster R-CNN은 객체 검출 분야에서 오랫동안 계산상의 난제로 남아 있던 **영역 제안(Region Proposal) 문제를 딥러닝 합성곱 계층 내부로 완전히 흡수 통합**한 최초의 완전한 종단간(End-to-End) 실시간 객체 검출 프레임워크이다.

- **RPN**과 **앵커(Anchor)** 메커니즘을 도입하여 거의 추가 비용 없이(10ms) 고품질 후보 영역을 추출할 수 있게 되었다.
- 이를 통해 전통적인 Selective Search 의존성을 완전히 제거하고, 정확도와 속도 양면에서 획기적인 돌파구를 열었다.
- Faster R-CNN의 아키텍처는 이후 컴퓨터 비전의 수많은 2단계(Two-stage) 검출 모델 및 세그멘테이션 모델(**Mask R-CNN**)의 확고한 뼈대(Baseline)로 자리매김하였다.
