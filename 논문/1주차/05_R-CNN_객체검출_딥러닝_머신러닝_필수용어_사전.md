# [1주차 논문 스터디] 객체 검출 · 딥러닝 · 머신러닝 · 파이썬 필수 용어 대백과사전

> **대상 범위**: R-CNN, Fast R-CNN, Faster R-CNN, Mask R-CNN 및 최신 컴퓨터 비전 / 딥러닝 논문 전반  
> **핵심 분야**: 딥러닝(Deep Learning), 머신러닝(Machine Learning), 파이썬/파이토치(PyTorch) 엔지니어링  
> 💡 *참고: 브라우저에서 실시간 검색, 카테고리 필터링, IoU 및 PR 곡선 슬라이더를 직접 조작해볼 수 있는 대화형 웹 문서는 [`05_R-CNN_객체검출_딥러닝_머신러닝_필수용어_사전.html`](file:///c:/study/논문/1주차/05_R-CNN_객체검출_딥러닝_머신러닝_필수용어_사전.html)을 열어보세요!*

---

## 목차 (Table of Contents)
1. [평가 지표 (Metrics: mAP, IoU, Precision, Recall, COCO 표기법)](#1-평가-지표-metrics)
2. [머신러닝 & 딥러닝 핵심 개념 (사전학습, 미세조정, SVM, 하드네거티브, SVD)](#2-머신러닝--딥러닝-핵심-개념)
3. [백본 & 신경망 구조 (CNN, Receptive Field, ResNet, FPN, FCN)](#3-백본--신경망-구조)
4. [영역 제안 & 검출 핵심 부품 (Selective Search, Anchor, RPN, NMS, 회귀)](#4-영역-제안--검출-핵심-부품)
5. [피처 정합 (RoI Pooling vs RoIAlign 비교 분석)](#5-피처-정합-feature-alignment)
6. [손실 함수 & 활성화 함수 (Multi-task Loss, Smooth L1, Softmax vs Sigmoid)](#6-손실-함수--활성화-함수)
7. [파이썬 & 파이토치(PyTorch) 코드 매핑 (IoU 벡터화, torchvision.ops, collate_fn)](#7-파이썬--파이토치pytorch-구현-매핑)
8. [1분 총정리 치트시트](#8-1분-총정리-치트시트)

---

## 1. 평가 지표 (Metrics)

### 1-1. mAP (mean Average Precision, 전체 클래스 평균 정밀도)
- **개념**: 객체 검출 모델의 성능을 나타내는 **단 하나의 종합 성적표(평균 점수)**입니다.
- **계산 방식**: 물체 클래스가 $K$개(사람, 자동차, 의자 등)라면, 각 클래스별 **AP(Average Precision, PR 곡선 아래 면적)**를 구한 뒤 산술평균합니다:
  $$\text{mAP} = \frac{1}{K} \sum_{k=1}^{K} \text{AP}_k$$
- **💡 1초 비유**: 전교생 기말고사 평균 점수. 국어(사람) AP 80점, 수학(자동차) AP 90점, 영어(신호등) AP 70점 $\rightarrow$ 평균 80점!
- **❓ 왜 단순 정확도(Accuracy) 대신 mAP를 쓸까?**:
  사진 1장의 90% 이상은 아무것도 없는 **'배경(Background)'**입니다. 모델이 물체를 1개도 못 찾고 *"아무것도 없음"*이라고만 대답해도 배경 맞춘 비율이 90%가 넘어버리는 **극단적 클래스 불균형(Class Imbalance)**이 발생하기 때문에, 찾은 물체와 위치의 질을 엄밀히 측정하는 mAP가 필수적입니다.

---

### 1-2. IoU (Intersection over Union, 교집합 대 합집합 비율)
- **개념**: 모델이 그린 예측 박스(Prediction)와 실제 정답 박스(Ground Truth)의 **공간적 일치율(0~1)**입니다:
  $$\text{IoU} = \frac{\text{교집합 면적 (Area of Overlap)}}{\text{합집합 면적 (Area of Union)}} = \frac{A \cap B}{A \cup B}$$
- **판정 기준**:
  - $\text{IoU} \ge 0.5$: PASCAL VOC 대회 기준 (물체를 잘 찾았다고 인정해 주는 관대한 합격선)
  - $\text{IoU} \ge 0.75$: COCO 엄격한 기준 (박스 테두리가 딱 들어맞아야 합격)
  - $\text{IoU} < 0.5$: 오답(False Positive) 판정

---

### 1-3. TP, FP, FN (객체 검출 관점)
객체 검출에서는 배경(True Negative)이 무한대이므로 계산에서 제외합니다.
- **TP (True Positive)**: $\text{IoU} \ge 0.5$ 이상이고 클래스 라벨도 맞춘 경우 $\rightarrow$ **진범 검거 성공**
- **FP (False Positive)**: $\text{IoU} < 0.5$ 미달이거나, 엉뚱한 배경을 물체라고 잡았거나, 중복으로 잡은 경우 $\rightarrow$ **무고한 시민 오인 체포 (헛다리)**
- **FN (False Negative)**: 진짜 물체가 존재하는데 모델이 아예 못 찾고 지나친 경우 $\rightarrow$ **진범을 놓침 (미검출)**

---

### 1-4. Precision(정밀도) vs Recall(재현율)의 줄다리기
- **정밀도 (Precision) = $\frac{TP}{TP + FP}$**: 모델이 검출했다고 외친 것 중 **진짜 정답의 비율**
  - *비유*: 100% 확실한 용의자만 잡는 **신중한 경찰** (무고한 사람 안 잡음)
- **재현율 (Recall) = $\frac{TP}{TP + FN}$**: 실제 존재하는 모든 정답 중 모델이 **찾아낸 비율**
  - *비유*: 조금이라도 의심되면 다 체포하는 **의욕 넘치는 경찰** (범인을 안 놓침)
- **트레이드오프와 PR Curve**:
  신뢰도(Confidence Score) 기준을 100%에서 0%로 낮출수록 모델이 박스를 남발하여 **Recall은 올라가지만 오답(FP)이 늘어나 Precision은 떨어집니다**. 이 변화를 그린 곡선이 **PR 곡선(우하향 곡선)**이며, **그 아래 면적(AUC)이 바로 AP**입니다.

---

### 1-5. 논문 결과표 표기법 완벽 해독 사전

| 논문 표기 | 공식 명칭 | IoU 기준선 | 특징 및 해석 |
| :--- | :--- | :--- | :--- |
| **$AP$ (COCO 표준)** | $mAP@[0.50:0.95]$ | 0.50 ~ 0.95 (0.05 단위 10개) | **현대 딥러닝 논문의 표준**. 위치가 정밀하게 맞아야 점수가 높게 나옴 |
| **$AP_{50}$** | $mAP@0.50$ | 0.50 | 전통적인 VOC 대회 기준. 대충 절반만 겹쳐도 정답으로 인정 |
| **$AP_{75}$** | $mAP@0.75$ | 0.75 | 고정밀 박스 검출 성능 |
| **$AP_S$** | AP Small | 면적 $< 32^2$ px | 원거리/소형 물체 검출 점수 (대부분 모델이 가장 낮음) |
| **$AP_M$ / $AP_L$** | AP Medium / Large | $32^2 \sim 96^2$ px / $> 96^2$ px | 중형 및 대형 물체 검출 점수 |

---

## 2. 머신러닝 & 딥러닝 핵심 개념

### 2-1. 지도 사전학습(Supervised Pre-training) & 도메인 특화 미세조정(Fine-tuning)
- **배경**: 당시 PASCAL VOC 검출 데이터는 수천 장 수준에 불과하여 수천만 개의 파라미터를 가진 CNN을 바닥부터(From scratch) 학습하면 극심한 과적합(Overfitting) 발생.
- **해결책 (R-CNN의 혁신)**:
  1. 100만 장의 ImageNet 분류 데이터셋으로 시각 일반 지식(선, 질감 등)을 **지도 사전학습(Pre-training)**.
  2. 사전 학습된 가중치를 가져와 소규모 VOC 데이터셋의 왜곡된 제안 영역들에 대해 SGD로 **도메인 특화 미세조정(Fine-tuning)**을 수행 $\rightarrow$ 전이학습(Transfer Learning)의 정석을 수립함.

### 2-2. 선형 서포트 벡터 머신 (Linear SVM)
- **개념**: 두 클래스 사이의 **마진(Margin)을 최대화하는 최적의 결정 초평면(Hyperplane)**을 찾는 고전 머신러닝 알고리즘.
- **R-CNN에서 따로 쓴 이유**: CNN 미세조정 시에는 양성 샘플을 늘리려고 IoU 0.5 기준(느슨함)을 썼지만, 최종 SVM을 학습시킬 때는 **"오직 실제 GT만 양성, IoU 0.3 미만만 음성"**이라는 매우 엄격한 기준으로 별도 학습시켜 mAP를 3.3%p 끌어올림. (Fast R-CNN에서 단일 Softmax로 통합)

### 2-3. 하드 네거티브 마이닝 (Hard Negative Mining)
- 이미지 1장에는 무수히 많은 배경이 있지만, 하늘이나 단색 벽 같은 뻔한 배경(Easy Negatives)은 학습에 도움이 안 됨.
- 배경인데도 모델이 높은 점수를 주며 **"자신 있게 헷갈려 한 어려운 오답 샘플(Hard Negatives)"**만 선별 채굴(Mining)하여 집중적으로 재학습시키는 기법.

### 2-4. 절단 특이값 분해 (Truncated SVD)
- **개념**: Fast R-CNN에서 전체 추론 시간의 절반 가까이를 차지하는 완전 연결 계층(FC layer)의 가중치 행렬 $W$($u \times v$)를 작은 두 행렬의 곱으로 근사 압축:
  $$W \approx U \Sigma_t V^T$$
- **효과**: 파라미터 수를 $u \times v$에서 $t(u + v)$로 대폭 축소하여 정확도 손실 없이 연산 속도를 30% 이상 가속함.

---

## 3. 백본 & 신경망 구조

### 3-1. 백본(Backbone) & 특징 맵(Feature Map)
- **백본**: 입력 이미지로부터 고차원 시각 특징을 뽑아내는 신경망 본체 (AlexNet, VGG16, ResNet-50 등).
- **수용 영역 (Receptive Field)**: 특징 맵의 특정 픽셀 1개가 원본 이미지에서 바라보는 실제 시야각 영역. 층이 깊어질수록 수용 영역이 커져 전체적인 맥락(Context)을 이해함.

### 3-2. ResNet & 스킵 연결 (Skip Connection)
- 망이 100층 이상 깊어질 때 발생하는 **기울기 소실(Vanishing Gradient)** 문제를 입력값을 건너뛰어 더해주는 잔차 연결($F(x) + x$)로 해결한 초심층 신경망. Mask R-CNN의 강력한 백본으로 활용됨.

### 3-3. FPN (Feature Pyramid Network, 특징 피라미드 네트워크)
- **문제**: 얕은 층은 해상도가 높아 위치를 잘 알지만 의미 정보가 부족하고, 깊은 층은 의미 정보는 풍부하지만 해상도가 낮아 작은 물체를 놓침.
- **해결책**: 깊은 층의 풍부한 의미를 위에서 아래로(Top-down) 전달하고, 얕은 층의 고해상도 특징을 측면 연결(Lateral Connection)로 합쳐 **$P_2 \sim P_6$의 다중 스케일 특징 피라미드를 구축**. 작은 물체($AP_S$) 탐지 성능을 비약적으로 끌어올림.

### 3-4. FCN (Fully Convolutional Network, 완전 합성곱 신경망)
- 위치 정보가 파괴되는 1차원 FC 계층을 걷어내고 $1 \times 1, 3 \times 3$ 합성곱 계층으로만 구성한 신경망. 2차원 공간 정보를 끝까지 보존하므로 **픽셀 단위 마스크(Mask R-CNN 마스크 헤드)** 예측에 필수적임.

---

## 4. 영역 제안 & 검출 핵심 부품

### 4-1. Selective Search (선택적 탐색)
- 색상, 무늬, 크기가 비슷한 픽셀들을 바닥에서부터 뭉쳐가며(Hierarchical Grouping) 후보 박스 2,000개를 제안하는 전통 영상처리 알고리즘.
- **한계**: CPU에서 동작하여 **장당 2초**나 소요되어 Fast R-CNN의 최악의 병목이 됨 $\rightarrow$ Faster R-CNN의 RPN으로 교체.

### 4-2. Bounding Box Regression (바운딩 박스 회귀)
- 제안 박스 $P$를 실제 정답 박스 $G$에 맞게 미세 조정하는 기술.
- **정규화 오프셋 공식**:
  $$t_x = \frac{G_x - P_x}{P_w}, \quad t_y = \frac{G_y - P_y}{P_h}, \quad t_w = \log\left(\frac{G_w}{P_w}\right), \quad t_h = \log\left(\frac{G_h}{P_h}\right)$$
- 픽셀 절대 좌표가 아닌 박스 크기 대비 상대적 이동/확대 비율(오프셋)로 학습시켜 **스케일 불변성(Scale-invariance)**을 확보.

### 4-3. Anchor Box (앵커 박스) & RPN (Region Proposal Network)
- **앵커 박스**: 다양한 물체 형태에 대비하기 위해 피처맵 각 위치마다 미리 깔아둔 템플릿 틀 (보통 3개 크기 $\times$ 3개 비율 = **9개 앵커**).
- **RPN**: Selective Search를 완전히 대체한 소형 FCN. 피처맵 위에서 앵커들을 기준으로 객체 유무 점수(Objectness)와 박스 오프셋을 **단 10ms 만에 초고속 계산**.

### 4-4. NMS (Non-Maximum Suppression, 비최대 억제)
- 동일한 물체에 중복으로 쳐진 수십 개의 박스 중 **가장 신뢰도가 높은 1등 박스만 남기고, 1등과 IoU가 높은(예: 0.7 이상) 나머지 박스들을 삭제**하는 필수 후처리 알고리즘.

---

## 5. 피처 정합 (Feature Alignment)

### RoI Pooling vs RoIAlign 비교

| 비교 항목 | RoI Pooling (Fast / Faster R-CNN) | RoIAlign (Mask R-CNN) |
| :--- | :--- | :--- |
| **좌표 처리** | 정수로 반올림/내림 (**양자화, Quantization**) | 소수점 실수 좌표 그대로 유지 (**No Quantization**) |
| **샘플링 기법** | 정수 격자 내 Max Pooling | 빈(bin) 내부 4개 위치에서 **쌍선형 보간법 (Bilinear Interpolation)** |
| **위치 오차** | 피처맵 0.5px 오차 $\rightarrow$ **원본에서 8~16px 심각한 위치 어긋남** | **위치 오차 정확히 0 (완벽 정합)** |
| **효과** | 대략적인 박스 분류에는 충분했음 | **픽셀 단위 마스크 분할(Mask AP) 3~7포인트 수직 상승** |

---

## 6. 손실 함수 & 활성화 함수

### 6-1. Multi-task Loss (다중 작업 손실)
$$L = L_{\text{cls}} + \lambda [u \ge 1] L_{\text{box}} + L_{\text{mask}}$$
- $L_{\text{cls}}$: 정답 클래스 $u$에 대한 분류 손실 (Cross Entropy)
- $L_{\text{box}}$: 바운딩 박스 위치 오차 손실 (Smooth L1)
- $[u \ge 1]$: **아이버슨 스위치**. 배경($u=0$)일 때는 박스 위치를 맞출 필요가 없으므로 박스 손실을 $0$으로 끔!
- $L_{\text{mask}}$: Mask R-CNN에서 추가된 $m \times m$ 픽셀 단위 마스크 손실

### 6-2. Smooth L1 Loss (왜 L2 대신 쓸까?)
$$\text{Smooth}_{L1}(x) = \begin{cases} 0.5 x^2 & \text{if } |x| < 1 \quad (\text{0 부근에서는 부드러운 L2}) \\ |x| - 0.5 & \text{otherwise} \quad (\text{멀어지면 이상치에 강한 L1}) \end{cases}$$
- $L_2(x^2)$는 엉뚱한 이상치가 들어오면 기울기가 폭발하고, $L_1(|x|)$는 0에서 미분이 불연속적.
- Smooth L1은 0 근처에서는 매끄럽게 수렴하고 멀어지면 1차 선형으로 증가하여 **이상치에 매우 강건(Robust)**함.

### 6-3. Softmax vs Sigmoid (클래스 경쟁 분리)
- **Softmax**: 전체 클래스 확률의 합이 100%이므로 클래스 간 제로섬 경쟁을 유발.
- **Sigmoid (BCE)**: 각 클래스별 마스크를 독립적으로 예측하여 **클래스 간 경쟁을 배제(Decoupling)**. Mask R-CNN 고성능의 핵심 비결.

---

## 7. 파이썬 & 파이토치(PyTorch) 구현 매핑

### 7-1. 행렬 브로드캐스팅을 활용한 벡터화된 IoU 계산 (Python/PyTorch)
```python
import torch

def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    boxes1: (N, 4) [x1, y1, x2, y2]
    boxes2: (M, 4) [x1, y1, x2, y2]
    반환: (N, M) 쌍별 IoU 행렬
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    # Broadcasting을 통한 교집합 좌상단/우하단 좌표 계산
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # (N, M, 2)
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # (N, M, 2)

    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]  # (N, M)

    union = area1[:, None] + area2 - inter
    return inter / union  # (N, M)
```

### 7-2. PyTorch 공식 C++/CUDA 내장 연산자 활용 (`torchvision.ops`)
```python
from torchvision.ops import nms, roi_align

# 1. NMS 실행 (중복 박스 제거 후 살아남을 인덱스 반환)
keep_indices = nms(boxes=boxes, scores=scores, iou_threshold=0.5)
final_boxes = boxes[keep_indices]

# 2. RoIAlign 실행 (소수점 쌍선형 보간 풀링)
aligned_features = roi_align(
    input=feature_map, 
    boxes=roi_boxes,         # (K, 5) -> [batch_idx, x1, y1, x2, y2]
    output_size=(7, 7), 
    spatial_scale=1.0 / 16.0, # 백본 Stride 반영
    sampling_ratio=2         # bin당 4개 포인트 샘플링
)
```

### 7-3. 객체 검출 DataLoader의 핵심: 왜 `collate_fn`이 필수일까?
```python
# 이미지마다 검출될 바운딩 박스 개수(고양이 2마리 vs 사람 10명)가 다르므로
# 기본 torch.stack 대신 튜플로 묶어주는 커스텀 collate_fn 필수
def collate_fn(batch):
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return torch.stack(images, dim=0), tuple(targets)

data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=4, shuffle=True, collate_fn=collate_fn
)
```

---

## 8. 1분 총정리 치트시트

1. **IoU**: 교집합 / 합집합 (0.5 이상이면 합격)
2. **Precision**: 검출했다고 한 것 중 진짜 맞춘 비율 (신중한 경찰)
3. **Recall**: 전체 정답 중 찾아낸 비율 (의욕 넘치는 경찰)
4. **AP / mAP**: PR 곡선 아래 면적 / 전 클래스 AP의 산술평균
5. **Selective Search**: CPU 기반 픽셀 뭉치기 (장당 2초 병목)
6. **RPN & Anchor**: 딥러닝 내부 10ms 초고속 영역 제안기 & 9개 템플릿 틀
7. **NMS**: 겹친 여러 박스 중 최고 점수 1개만 남기고 삭제
8. **RoIAlign**: 소수점 안 버리고 쌍선형 보간으로 픽셀 칼정렬 (Mask R-CNN 핵심)
9. **Smooth L1**: 0 근처에서는 부드러운 L2, 멀어지면 이상치에 강한 L1
10. **collate_fn**: 사진마다 박스 개수가 제각각인 검출 데이터를 배치로 묶는 필수 파이썬 함수
