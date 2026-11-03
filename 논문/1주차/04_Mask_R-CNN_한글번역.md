# Mask R-CNN (마스크 R-CNN)

- **원제**: Mask R-CNN
- **저자**: Kaiming He, Georgia Gkioxari, Piotr Dollár, Ross Girshick (Facebook AI Research, FAIR)
- **학술대회**: IEEE International Conference on Computer Vision (ICCV) 2017 (최우수 논문상, **Marr Prize 수상**)
- **코드 저장소**: [https://github.com/facebookresearch/Detectron](https://github.com/facebookresearch/Detectron)
- **원문 링크**: [arXiv:1703.06870](https://arxiv.org/abs/1703.06870)

---

## 초록 (Abstract)

본 논문에서는 이미지 내 객체 인스턴스 분할(Instance Segmentation)을 위한 개념적으로 단순하고 유연하며 범용적인 프레임워크인 **Mask R-CNN**을 제안한다. 이 방식은 이미지 내 각 객체에 대해 고품질의 분할 마스크(Segmentation Mask)를 생성하면서 동시에 기존 객체 검출(Bounding Box Object Detection)을 수행한다.

Mask R-CNN은 기존의 **Faster R-CNN**에 각 관심 영역(Region of Interest, RoI)에 대한 분할 마스크를 예측하는 작은 브랜치(Branch)를 기존의 경계 상자 인식 브랜치와 **병렬(In parallel)**로 추가함으로써 이를 달성한다. Mask R-CNN은 학습이 매우 간단하며, Faster R-CNN에 비해 약간의 연산 오버헤드만 추가되므로 약 **5 fps**의 빠른 속도로 실행된다.

더 나아가 Mask R-CNN은 다른 작업으로의 일반화가 매우 용이하다. 예를 들어 동일한 프레임워크 내에서 사람의 관절 위치를 추정하는 **인체 포즈 추정(Human Pose Estimation)** 작업에도 쉽게 확장 적용할 수 있다.

COCO 챌린지의 인스턴스 분할(Instance Segmentation), 바운딩 박스 객체 검출(Bounding Box Detection), 인체 키포인트 검출(Person Keypoint Detection)의 모든 과제에서 기존의 모든 단일 모델 최신 성능(SOTA)을 압도하였다.

---

## 1. 서론 (Introduction)

컴퓨터 비전 커뮤니티는 객체 검출(Bounding Box Detection)과 시맨틱 분할(Semantic Segmentation, 픽셀 단위 범주 분류)에서 눈부신 발전을 이루었다. **인스턴스 분할(Instance Segmentation)**은 이 두 작업의 융합으로, 이미지 내의 모든 개별 객체를 올바르게 탐지하는 동시에 각 객체의 정밀한 픽셀 마스크를 정확하게 분할해야 하므로 훨씬 도전적인 과제이다.

우리의 목표는 인스턴스 분할을 위한 놀랍도록 단순하고, 유연하며, 빠른 최고 수준의 시스템을 구축하는 것이다.

우리가 제안하는 **Mask R-CNN**은 Faster R-CNN을 확장한 구조이다. Faster R-CNN은 각 후보 객체에 대해 클래스 라벨(Class label)과 경계 상자 오프셋(Bounding-box offset)이라는 2개의 출력을 갖는다. 여기에 우리는 **객체의 픽셀 단위 공간 마스크(Object Mask)를 출력하는 세 번째 브랜치를 병렬로 추가**하였다.

```
                         ┌──► [클래스 분류 (Classification)]
                         │
[RoI 특징 맵] ──► [RoIAlign] ───┼──► [경계 상자 회귀 (Box Regression)]
                         │
                         └──► [마스크 FCN 브랜치 (Mask Segmentation)] (신규 추가)
```

원리상 직관적인 확장이지만, 고품질 마스크를 얻기 위해서는 두 가지 핵심적인 설계가 결정적이었다:

1. **마스크와 클래스 예측의 분리 (Decoupling Mask and Class Prediction)**:
   - 전통적인 FCN 기반 시맨틱 세그멘테이션은 픽셀마다 클래스 간 다항 소프트맥스(Softmax) 경쟁을 유발한다.
   - Mask R-CNN은 각 클래스별로 독립적인 바이너리 마스크(Binary Mask)를 시그모이드(Sigmoid)로 예측하고, 전용 분류 브랜치가 예측한 클래스에 해당하는 마스크만 선택하여 클래스 간 경쟁을 배제하였다.
2. **픽셀 단위 공간 정합을 위한 RoIAlign (Pixel-to-Pixel Alignment)**:
   - Faster R-CNN의 핵심 연산이었던 **RoIPool**은 부동소수점 좌표를 정수 격자로 반올림하는 거친 공간 **양자화(Quantization)**를 수행한다.
   - 이는 작은 위치 편향에 둔감한 분류 작업에는 문제가 없으나, 픽셀 단위의 정밀도가 요구되는 마스크 분할에는 치명적인 위치 오차(Misalignment)를 유발한다.
   - 이를 해결하기 위해 양자화를 완전히 제거하고 **쌍선형 보간법(Bilinear Interpolation)**을 통해 연속 좌표의 특징값을 정확히 추출하는 **RoIAlign** 계층을 고안하였다. RoIAlign의 도입만으로 마스크 AP가 3~7포인트 이상 급상승하였다.

---

## 2. 관련 연구 (Related Work)

- **R-CNN 계열**: R-CNN $\rightarrow$ Fast R-CNN $\rightarrow$ Faster R-CNN으로 진화하며 영역 제안과 특징 추출을 통합하여 유연하고 강건한 2단계 검출 프레임워크를 확립했다.
- **인스턴스 분할의 선행 기법들**:
  - 하향식(Top-down) 방식: MNC, FCIS 등은 마스크 제안을 먼저 생성한 후 분류하거나 복잡한 결합 계층을 사용했으나, 세그멘테이션과 분류가 서로 결합되어 성능에 제약이 있었다.
  - 상향식(Bottom-up) 방식: 픽셀 임베딩 후 클러스터링하는 방식 등이 있었으나 복잡하고 느렸다.

---

## 3. Mask R-CNN 아키텍처

### 3.1. 다중 작업 손실 함수 (Multi-task Loss)

학습 중 각 샘플링된 RoI에 대해 다중 작업 손실을 다음과 같이 정의한다:
$$L = L_{cls} + L_{box} + L_{mask}$$

- $L_{cls}$ (분류 손실)와 $L_{box}$ (경계 상자 회귀 손실): Fast/Faster R-CNN에서 정의된 것과 동일하다.
- **마스크 브랜치($L_{mask}$)**:
  - 각 RoI에 대해 $K \times m \times m$ 차원의 출력을 생성한다. 이는 $K$개의 클래스 각각에 대해 $m \times m$ 해상도의 바이너리 마스크를 의미한다.
  - 여기에 픽셀 단위 시그모이드(Per-pixel Sigmoid)를 적용하고, **평균 이진 교차 엔트로피(Average Binary Cross-Entropy) 손실**로 $L_{mask}$를 정의한다.
  - 해당 RoI의 정답 클래스가 $k$라면, **오직 $k$번째 마스크에 대해서만 손실이 계산**되며, 나머지 클래스의 마스크는 손실에 반영되지 않는다.

> **마스크와 클래스의 분리 (Decoupling)**:
> 클래스 라벨 선택은 분류 브랜치에 전적으로 맡기고, 마스크 브랜치는 오직 "해당 물체의 형태(Shape) 마스크"만을 전담하여 학습하므로, 클래스 간 경쟁 없이 마스크 품질이 비약적으로 향상된다.

---

### 3.2. RoIAlign 계층

RoIPool의 문제점을 해결하기 위해 고안된 양자화 없는(Quantization-free) 정밀 계층이다.

#### RoIPool의 공간 양자화 문제:
- 입력 이미지의 RoI 좌표 $x$를 스트라이드 16인 특징 맵으로 사영할 때 $[x/16]$으로 반올림(양자화)한다.
- 이를 다시 $7 \times 7$ bin으로 나눌 때 각 bin의 경계에서도 또 한 번의 반올림(양자화)이 발생한다.
- 이 과정에서 최대 1개 픽셀 이상의 오차가 발생하며, 원본 이미지 해상도(스트라이드 16배)로 환산하면 **수십 픽셀의 오정렬(Misalignment)**이 생겨 정밀한 마스크 경계가 무너진다.

#### RoIAlign의 동작 메커니즘:
1. **양자화 제거**: RoI 경계 좌표나 bin 분할 시 반올림을 일체 수행하지 않고 부동소수점(Continuous floating point) 좌표를 그대로 유지한다 ($x/16$).
2. **규칙 샘플링 지점 배치**: 각 RoI bin 내부에 4개의 규칙적인 샘플링 지점(예: 각 사분면의 중심점)을 배치한다.
3. **쌍선형 보간법 (Bilinear Interpolation)**: 특징 맵 상의 가장 가까운 4개 정수 격자 좌표의 값들로부터 각 샘플링 지점의 특징값을 연속적으로 보간 계산한다.
4. **풀링 집계**: 각 bin 내부의 4개 샘플링 지점 값들을 맥스 풀링 또는 평균 풀링(Average pooling)하여 해당 bin의 최종 특징값으로 취한다.

이 간단한 변화로 특징 맵과 원본 이미지 간의 완벽한 픽셀 단위 일치(Pixel-to-pixel alignment)가 유지된다.

---

### 3.3. 네트워크 아키텍처 (백본 및 헤드)

우리는 Mask R-CNN을 다양한 백본 네트워크로 인스턴스화하였다:

1. **백본 아키텍처 (Backbone)**:
   - **ResNet-C4**: 표준 ResNet-50/101의 4번째 잔차 블록(conv4)까지를 백본으로 사용하고, RoI 추출 후 5번째 블록(conv5)을 헤드로 사용.
   - **FPN (Feature Pyramid Network)**: 하향식(Top-down) 경로와 측면 연결(Lateral connection)을 통해 다중 스케일의 고수준 의미 특징 피라미드를 구축. ResNet-FPN 백본은 속도와 정확도 모두에서 가장 우수한 성능을 발휘한다.
2. **헤드 아키텍처 (Head)**:
   - **박스 헤드**: 기존 Faster R-CNN과 동일한 완전 연결(FC) 계층 또는 합성곱 계층 구조.
   - **마스크 헤드**: 공간적 배치를 보존하기 위해 완전 연결 계층 대신 작은 **완전 합성곱 네트워크(FCN)**로 구성된다. $14 \times 14$ RoIAlign 출력 위에 $3 \times 3$ Conv 계층들을 쌓고 마지막에 $2 \times 2$ Transposed Conv(Deconv)를 적용하여 $28 \times 28$ 해상도의 마스크를 출력한다.

---

## 4. 인스턴스 세그멘테이션 실험 결과

### 4.1. MS COCO 벤치마크 결과 (test-dev)

| 방식 | 백본 | $AP^{mask}$ | $AP^{mask}_{50}$ | $AP^{mask}_{75}$ | $AP^{box}$ (객체검출) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| MNC (CVPR 2016 우승) | ResNet-101 | 24.6 | 44.3 | 24.8 | - |
| FCIS (COCO 2016 우승) | ResNet-101 | 29.2 | 49.5 | - | - |
| **Mask R-CNN** | **ResNet-50-FPN** | **34.2** | **55.9** | **36.2** | **37.3** |
| **Mask R-CNN** | **ResNet-101-FPN** | **35.7** | **58.0** | **37.8** | **39.8** |
| **Mask R-CNN** | **ResNeXt-101-FPN** | **37.1** | **60.0** | **39.4** | **41.3** |

Mask R-CNN은 이전 연도 COCO 대회 우승 모델들을 **단일 모델만으로 8%p 이상 큰 격차**로 압도하며 최고 성능을 달성하였다.

---

### 4.2. 핵심 절제 연구 (Ablation Study)

#### 1) RoIAlign의 효과 (vs. RoIPool, RoIWarp)

| 정합 방식 | $AP^{mask}$ | $AP^{mask}_{50}$ | $AP^{mask}_{75}$ | $AP^{box}$ | $AP^{box}_{75}$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| RoIPool | 23.6 | 46.5 | 21.6 | 28.2 | 29.3 |
| RoIWarp | 23.6 | 47.9 | 22.0 | 28.5 | 29.6 |
| **RoIAlign** | **30.2** | **51.8** | **31.7** | **30.9** | **33.2** |

- RoIAlign은 RoIPool 대비 마스크 AP를 **+6.6%p**, 엄격한 기준인 $AP_{75}$에서는 무려 **+10.1%p** 향상시켰다.
- 바운딩 박스 검출($AP^{box}$)에서도 정합성 개선 덕분에 +2.7%p 상승하였다.

#### 2) 독립 시그모이드 분리 vs 다항 소프트맥스

| 마스크 손실 유형 | $AP^{mask}$ | $AP^{mask}_{50}$ | $AP^{mask}_{75}$ |
| :--- | :---: | :---: | :---: |
| Softmax (다항 경쟁) | 24.8 | 44.1 | 25.1 |
| **Sigmoid (바이너리 분리)** | **30.3** | **51.2** | **31.9** |

- 클래스 경쟁을 없애고 바이너리 시그모이드로 마스크를 분리하자 **AP가 +5.5%p 급증**하였다. 이는 마스크 생성 시 클래스 간 경쟁을 배제하는 것이 얼마나 중요한지를 보여준다.

---

## 5. 인체 포즈 추정 확장 (Human Pose Estimation)

Mask R-CNN의 뛰어난 유연성을 입증하기 위해, 우리는 모델을 인체 키포인트 검출(Keypoint Detection) 작업으로 확장하였다:
- 사람 신체의 17개 관절 키포인트(머리, 어깨, 팔꿈치, 손목, 무릎 등) 각각을 $m \times m$ 해상도의 One-hot 바이너리 마스크로 취급한다.
- 마스크 브랜치를 $17$개 채널의 마스크를 출력하도록 수정하고, 각 채널에 대해 공간 소프트맥스(Spatial Softmax)를 적용하여 정답 키포인트 위치의 크로스 엔트로피를 최소화한다.
- 결과: COCO 2016 키포인트 챌린지 우승작을 능가하는 **63.1 AP**를 기록하며, 객체 검출, 인스턴스 세그멘테이션, 포즈 추정까지 **초당 5프레임의 단일 모델로 동시에 통합 처리**할 수 있음을 입증하였다.

---

## 6. 결론 (Conclusion)

Mask R-CNN은 Faster R-CNN의 바운딩 박스 객체 검출 파이프라인에 FCN 마스크 브랜치와 **RoIAlign**을 결합함으로써, 인스턴스 세그멘테이션 분야의 패러다임을 바꾼 혁신적인 연구이다.
- 구조가 직관적이고 구현이 깔끔하며, 높은 정밀도와 실시간에 준하는 빠른 속도를 동시에 달성하였다.
- 픽셀 수준의 정밀한 정합(RoIAlign)과 클래스-마스크 분리의 중요성을 학술적으로 규명하였다.
- 현재까지도 수많은 최신 컴퓨터 비전 연구와 산업계 응용 분야에서 가장 강력하고 표준적인 인스턴스 분할 및 다중 태스크 학습의 베이스라인으로 널리 사용되고 있다.
