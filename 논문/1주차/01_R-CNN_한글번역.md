# R-CNN: 정확한 객체 검출과 시맨틱 분할을 위한 풍부한 특징 계층
(Rich feature hierarchies for accurate object detection and semantic segmentation)

- **저자**: Ross Girshick, Jeff Donahue, Trevor Darrell, Jitendra Malik (UC Berkeley)
- **학술대회**: IEEE Conference on Computer Vision and Pattern Recognition (CVPR) 2014 / Tech report v5
- **코드 저장소**: [https://github.com/rbgirshick/rcnn](https://github.com/rbgirshick/rcnn)
- **원문 링크**: [arXiv:1311.2524](https://arxiv.org/abs/1311.2524)

---

## 초록 (Abstract)

표준 PASCAL VOC 데이터셋에서 측정된 객체 검출(Object Detection) 성능은 최근 몇 년간 정체되어 있었다. 당시 가장 성능이 우수한 방법들은 저수준 시각 특징(Low-level Visual Features)과 고수준 문맥 정보를 결합한 복잡한 앙상블 시스템들이었다.

본 논문에서는 PASCAL VOC 2012 기준 이전 최고 성능 대비 mAP를 30% 이상 비약적으로 향상시켜 **53.3% mAP**를 달성한 단순하고 확장 가능한 객체 검출 알고리즘을 제안한다.

우리의 접근 방식은 두 가지 핵심적인 통찰(Key Insights)에 기반한다:
1. **객체의 위치를 파악하고 분할하기 위해 상향식 영역 제안(Bottom-up Region Proposals)에 고용량 합성곱 신경망(Deep CNN)을 적용**할 수 있다 ("Regions with CNN features", 즉 **R-CNN**).
2. **라벨이 지정된 학습 데이터가 부족할 때, 보조 작업(Auxiliary Task, ImageNet 분류)에 대한 지도 사전 학습(Supervised Pre-training)을 거친 후, 대상 도메인(PASCAL VOC 검출)에 맞추어 미세 조정(Domain-specific Fine-tuning)을 수행하면 엄청난 성능 향상**을 얻을 수 있다.

또한 우리는 R-CNN이 시맨틱 분할(Semantic Segmentation) 작업으로도 유연하게 확장되어 PASCAL VOC 2011 세그멘테이션 작업에서 47.9%의 최고 성능을 기록함을 입증하였다.

---

## 1. 서론 (Introduction)

지난 10년 동안 시각 인식 분야는 SIFT(Scale-Invariant Feature Transform)와 HOG(Histogram of Oriented Gradients) 같은 수작업(Hand-crafted) 국소 특징량의 사용에 크게 의존해 왔다. 그러나 PASCAL VOC 객체 검출 대회에서는 2010년부터 2012년까지 앙상블과 미세한 변형을 통해서만 소폭의 성능 향상이 이루어질 뿐 정체기에 접어들었다.

2012년 Krizhevsky 등(AlexNet)은 8계층 심층 합성곱 신경망(CNN)을 통해 ImageNet 대규모 시각 인식 챌린지(ILSVRC) 이미지 분류에서 압도적인 격차로 우승을 차지하였다. 이에 따라 컴퓨터 비전 분야의 핵심 화두는 다음과 같았다:
> *"ImageNet 분류에서 거둔 CNN의 획기적인 성공이 PASCAL VOC의 객체 검출 과제에도 그대로 적용될 수 있는가?"*

우리는 이 질문에 긍정적으로 답하며, 이미지 분류를 넘어 객체 검출을 해결하기 위해 두 가지 핵심 난제를 해결하였다:
1. **깊은 신경망으로 객체의 국소 위치를 어떻게 파악할 것인가 (Localizing Objects with a Deep Network)**:
   - 슬라이딩 윈도우(Sliding-window) 방식은 수용 영역(Receptive field)이 크고 스트라이드가 넓은 심층 CNN에서는 위치 해상도가 떨어져 직접 적용하기 어렵다.
   - 우리는 **"영역을 활용한 인식(Recognition using regions)"** 패러다임을 채택하였다. 먼저 입력 이미지로부터 약 2,000개의 무관한 카테고리의 후보 영역(Region Proposals)을 생성한 뒤, 각 제안 영역에서 CNN을 통해 고정 길이 특징 벡터(4,096차원)를 추출하고, 클래스별 선형 SVM으로 분류한다.
2. **희소한 주석 데이터로 대용량 모델을 어떻게 학습시킬 것인가 (Training with Scarce Data)**:
   - 당시 PASCAL VOC 검출 데이터셋은 수천 장 수준에 불과하여 수천만 개의 파라미터를 가진 대형 CNN을 바닥부터(From scratch) 학습시키면 극심한 과적합이 발생한다.
   - 우리는 **대규모 보조 데이터셋(ImageNet 100만 장)에서 지도 사전 학습(Supervised Pre-training)**을 먼저 수행한 후, **소규모 검출 데이터셋(VOC)의 왜곡된(Warped) 영역 제안들에 대해 확률적 경사 하강법(SGD)으로 도메인 특화 미세 조정(Fine-tuning)**을 수행하는 패러다임을 정립하였다.

---

## 2. R-CNN을 활용한 객체 검출

R-CNN 시스템은 크게 3개의 모듈로 구성된다:
1. 카테고리 독립적인 **영역 제안 생성 모듈(Region Proposals)**: 검출기가 고려할 후보 영역을 정의.
2. 각 제안 영역에서 고정 크기의 특징 벡터를 추출하는 **대형 심층 합성곱 신경망(Deep CNN)**.
3. 선형 **서포트 벡터 머신(SVM) 분류기 세트**.

```
[입력 이미지]
      │
      ▼
[영역 제안 (Selective Search)] ──► 약 2,000개의 후보 영역 추출
      │
      ▼
[영역 워핑 (Warping + Padding)] ──► 모든 영역을 동일 크기 (227×227)로 리사이징
      │
      ▼
[심층 합성곱 신경망 (AlexNet)] ──► 4,096차원 특징 벡터 추출 (fc7)
      │
      ├──────────────────────────────┐
      ▼                              ▼
[클래스별 선형 SVM 분류기]      [경계 상자 회귀기 (Bounding Box Regressor)]
(객체 클래스 점수 평가)         (DPM 방식 기반 위치 보정)
      │
      ▼
[비최대 억제 (NMS)] ──► 중복 제거 후 최종 검출 결과 도출
```

---

### 2.1. 모듈 설계 세부사항

#### 1) 영역 제안 (Region Proposals)
기존 연구들과의 공정한 비교를 위해 비지도 세그멘테이션 기반 알고리즘인 **Selective Search(선택적 탐색)**의 'Fast mode'를 사용한다. 이미지당 약 2,000개의 다양한 크기와 종횡비를 갖는 후보 영역이 추출된다.

#### 2) 특징 추출 (Feature Extraction)
각 제안 영역으로부터 Caffe 라이브러리로 구현된 5개의 합성곱 계층과 2개의 완전 연결 계층(AlexNet 구조)을 통과시켜 **4,096차원 특징 벡터**를 추출한다.
- CNN의 입력 크기는 $227 \times 227$ 픽셀로 고정되어 있다.
- 임의의 종횡비를 가진 사각형 제안 영역을 CNN 입력으로 맞추기 위해 가장 단순한 **이방성 스케일링(Anisotropic Warping, 비율 무시 강제 리사이징)**을 적용한다.
- 이때 경계 부근의 문맥 정보(Context)를 보존하기 위해 원본 영역 주위로 **$p = 16$ 픽셀의 패딩(Padding)**을 추가한 후 워핑을 수행한다.

---

### 2.2. 테스트 단계 추론 (Test-time Detection)

1. 테스트 이미지에 대해 Selective Search를 수행하여 약 2,000개의 제안 영역을 추출한다.
2. 각 영역을 $227 \times 227$ 크기로 워핑하고 CNN을 통과시켜 4,096차원의 특징 벡터를 추출한다.
3. 각 클래스별로 미리 학습된 선형 SVM을 적용하여 점수를 매긴다.
4. 모든 영역 점수가 계산되면, 각 클래스별로 독립적으로 **탐욕적 비최대 억제(Greedy Non-Maximum Suppression, NMS)**를 적용한다:
   - 점수가 가장 높은 영역을 선택하고, 이와 IoU(Intersection over Union)가 **0.3 이상**인 다른 중복 영역들을 억제(제거)한다.

---

### 2.3. 모델 학습 파이프라인 (Training)

R-CNN의 학습은 엄격히 분리된 **3단계 파이프라인**으로 진행된다:

#### 1단계: 지도 사전 학습 (Supervised Pre-training)
ILSVRC 2012 분류 데이터셋(100만 장의 이미지, 1,000개 클래스)을 사용하여 바운딩 박스 라벨 없이 이미지 단위 분류 목표로 CNN 가중치를 사전 학습한다.

#### 2단계: 도메인 특화 미세 조정 (Fine-tuning)
PASCAL VOC의 왜곡된 제안 영역들에 맞추기 위해 CNN 파라미터를 SGD로 미세 조정한다:
- 기존의 ImageNet 1,000-way 분류 계층을 무작위로 초기화된 **$(N+1)$-way 분류 계층**($N$개 객체 클래스 + 1개 배경 클래스)으로 교체한다.
- **양성/음성 라벨링 기준**:
  - Ground-Truth(GT) 상자와 **IoU가 0.5 이상인 모든 제안 영역을 해당 클래스의 양성(Positive)**으로 간주한다.
  - 나머지(IoU < 0.5)는 모두 배경(Negative)으로 처리한다.
- **미니배치 구성**: 미니배치 크기 128 중 32개의 전경 창(모든 클래스에 걸침)과 96개의 배경 창을 샘플링하여 1:3의 전경-배경 비율을 유지한다. 학습률은 사전 학습의 1/10 수준인 0.001로 시작한다.

#### 3단계: 클래스별 객체 범주 선형 SVM 학습
미세 조정된 특징 벡터를 입력으로 받아 각 클래스별 선형 SVM을 독립적으로 학습시킨다:
- **SVM 학습 시의 양성/음성 기준 (미세 조정과 다름!)**:
  - **오직 실제 GT 바운딩 박스만을 양성(Positive)**으로 정의한다.
  - **IoU가 0.3 미만인 제안 영역만을 음성(Negative)**으로 정의한다.
  - IoU가 0.3~0.5 사이의 회색 지대 영역은 엄격성을 위해 학습에서 완전히 배제한다.
- 표준적인 **하드 네거티브 마이닝(Hard Negative Mining)**을 적용하여 대규모 음성 샘플을 효율적으로 학습한다.

> **Q: 왜 Fine-tuning의 소프트맥스를 그대로 쓰지 않고 별도의 SVM을 학습시켰는가?**  
> 미세 조정 시에는 데이터 부족을 극복하기 위해 IoU 0.5 기준을 사용하여 다소 느슨하게 양성 샘플 수를 늘려야 했다. 반면 SVM 학습 시에는 엄밀한 GT만을 양성으로 삼고 0.3 미만을 음성으로 삼는 엄격한 정의를 적용했을 때 mAP가 50.9%에서 54.2%로 3.3%p 더 높게 나왔기 때문이다. (이후 Fast R-CNN에서 단일 소프트맥스로 통합됨)

---

## 3. 절제 연구 및 주요 발견 (Ablation Studies)

### 3.1. 계층별 특징 분석 및 Fine-tuning의 영향 (PASCAL VOC 2007)

| 특징 계층 | Fine-tuning 여부 | mAP (%) | 비고 |
| :--- | :---: | :---: | :--- |
| `pool5` | No | 44.2 | 6%의 파라미터만으로 상당한 성능 달성 |
| `fc6` | No | 46.2 |  |
| `fc7` | No | 44.9 | 파라미터가 많은 상위 계층일수록 사전학습만으로는 일반화 한계 |
| `pool5` | **Yes (FT)** | 47.3 | 미세 조정 효과 (+3.1%p) |
| `fc6` | **Yes (FT)** | 53.1 |  |
| `fc7` | **Yes (FT)** | **54.2** | **도메인 특화 미세 조정으로 +9.3%p 성능 대폭 폭발** |
| `fc7 + BB` | **Yes (FT)** | **58.5** | **바운딩 박스 회귀 추가 시 최종 최고 성능 달성** |

- **핵심 시사점**: ImageNet 사전 학습 후 대상 도메인 데이터로 상위 완전 연결 계층들을 미세 조정하는 것이 검출 성능 향상의 절대적인 열쇠임이 밝혀졌다.

---

### 3.2. 바운딩 박스 회귀 (Bounding-Box Regression, 부록 C)

위치 오차를 줄이기 위해 DPM(Deformable Part Models)의 아이디어에서 착안한 선형 회귀 모델을 도입하였다.
선택적 탐색의 제안 영역 $P = (P_x, P_y, P_w, P_h)$를 정답 상자 $G = (G_x, G_y, G_w, G_h)$로 매핑하는 4개의 변환 함수를 학습한다:

$$\hat{G}_x = P_w d_x(P) + P_x$$
$$\hat{G}_y = P_h d_y(P) + P_y$$
$$\hat{G}_w = P_w \exp(d_w(P))$$
$$\hat{G}_h = P_h \exp(d_h(P))$$

각 $d_*(P)$는 CNN의 `pool5` 특징 벡터 $\phi_5(P)$의 선형 함수 $w_*^T \phi_5(P)$로 모델링되며, 정규화된 릿지 회귀(Ridge Regression, $\lambda = 1000$)를 통해 폐쇄형(Closed-form)으로 최적화된다:

$$w_* = \arg\min_{\hat{w}_*} \sum_i \left(t_*^i - \hat{w}_*^T \phi_5(P^i)\right)^2 + \lambda \|\hat{w}_*\|^2$$

여기서 목표 변환 타깃 $t_*$는 다음과 같이 정의된다:
$$t_x = (G_x - P_x)/P_w, \quad t_y = (G_y - P_y)/P_h$$
$$t_w = \log(G_w / P_w), \quad t_h = \log(G_h / P_h)$$

- **효과**: 바운딩 박스 회귀를 추가하는 것만으로 mAP가 **3~4%p 즉각 상승**하였다. 이 좌표 정규화 공식은 훗날 Fast R-CNN, Faster R-CNN, Mask R-CNN, YOLO, SSD 등 수많은 딥러닝 객체 검출 모델의 표준 바운딩 박스 회귀 공식이 되었다.

---

## 4. PASCAL VOC 성능 비교 종합

VOC 2010 테스트 세트 결과:

| 방식 | 사전 학습 모델 | mAP (%) |
| :--- | :--- | :---: |
| DPM v5 (전통적 최고 모델) | - | 33.4 |
| SegDPM | - | 40.4 |
| UVA Detection | - | 35.1 |
| **R-CNN (AlexNet)** | **ImageNet** | **50.2** |
| **R-CNN (VGG16 / O-Net)** | **ImageNet** | **62.9** |

- 이전 최고 기록(SegDPM, 40.4%)을 10%p 이상 뛰어넘었으며, VGG16 백본 사용 시 **62.9%**까지 상승하였다.

---

## 5. R-CNN의 의의와 한계점

### 1) 역사적 의의
- 딥러닝(CNN)을 객체 검출 분야에 성공적으로 안착시킨 최초의 기념비적 연구.
- **"대규모 데이터 사전학습 $\rightarrow$ 소규모 목표 도메인 미세조정(Transfer Learning)"**이라는 현대 딥러닝 연구의 보편적 패러다임을 확립.
- 영역 기반 객체 검출 및 바운딩 박스 회귀의 표준 수학적 프레임워크를 정립.

### 2) 명확한 한계점 (후속 연구의 계기)
1. **극도로 느린 추론 속도**: 이미지 1장당 약 2,000개의 후보 영역 모두에 대해 CNN 순전파를 개별적으로 수행해야 하므로 GPU에서도 장당 수십 초 소요.
2. **다단계 학습 파이프라인**: CNN 미세조정, SVM 분류기 학습, 바운딩 박스 회귀기 학습이 모두 따로 분리되어 번거롭고 최적화가 비효율적임.
3. **막대한 디스크 용량 소모**: SVM과 회귀기를 학습하기 위해 수천 장 이미지의 모든 영역 특징 벡터(수백 GB)를 디스크에 캐싱해야 함.

$\Longrightarrow$ 이 한계점들을 극복하기 위해 **Fast R-CNN**(단일 단계 다중 손실 및 RoI 풀링)과 **Faster R-CNN**(RPN을 통한 종단간 네트워크)이 탄생하게 된다.
