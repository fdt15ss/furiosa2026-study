# Mask R-CNN 10분 발표 완성 대본 (Presentation Script)

> **대상 슬라이드**: `Mask_RCNN_Balanced_Revision_21slides.pptx` (총 21장)  
> **발표 목표 시간**: **9분 30초 ~ 10분** (질의응답 제외 순수 발표 기준)  
> **발표 톤앤매너**: 자신감 있고 명확한 학술/엔지니어링 발표 톤 (자연스러운 구어체)

---

## ⏱️ 슬라이드별 시간 배분표 (Time Allocation)

| 파트 | 슬라이드 번호 및 주제 | 권장 시간 | 누적 시간 |
| :--- | :--- | :---: | :---: |
| **Part 1. 인트로 & 과업 정의** | Slide 1~2: 도입 및 Detection $\rightarrow$ Instance Segmentation | 55초 | 00:55 |
| **Part 2. R-CNN 계보의 진화** | Slide 3~6: R-CNN $\rightarrow$ Fast $\rightarrow$ Faster 병목 극복사 | 1분 35초 | 02:30 |
| **Part 3. Mask R-CNN 핵심 구조** | Slide 7~10: Two RoI Path, RoIAlign, Mask Head | 2분 00초 | 04:30 |
| **Part 4. 핵심 구현 코드 7선** | Slide 11~18: Head, Predictor, Branch, Target, Loss, Infer, PostProc | 3분 50초 | 08:20 |
| **Part 5. 실험 결과 & 결론** | Slide 19~21: Highway 예시 결과, 4편 논문 총정리, Q&A | 1분 20초 | **09:40** |

---

## 🎤 슬라이드별 발표 대본 (Slide-by-Slide Script)

---

### [Slide 1] 표지 · 도입
- **누적 시간**: `00:00 - 00:25` (25초)
- **슬라이드 핵심**: R-CNN 계보의 발전 흐름에서 Mask R-CNN의 구조와 직접 구현 코드까지 원스톱으로 정리.
- **발표 대본**:
  > "안녕하십니까, **'Mask R-CNN: Architecture에서 Implementation까지'** 발표를 맡은 김상현입니다.  
  > 오늘 발표는 R-CNN, Fast, Faster R-CNN으로 이어지는 객체 탐지 계보의 핵심 병목 해결 과정을 짚어보고, Mask R-CNN이 이를 바탕으로 어떻게 픽셀 단위 분할을 달성했는지 **아키텍처 다이어그램과 7가지 파이토치 핵심 코드**를 중심으로 10분간 명쾌하게 설명해 드리겠습니다."

---

### [Slide 2] 01 · PROBLEM: 객체 탐지에서 인스턴스 분할까지
- **누적 시간**: `00:25 - 00:55` (30초)
- **슬라이드 핵심**: Classification vs Detection vs Instance Segmentation의 차이.
- **행동 팁**: `[하단 3개 비교 박스를 좌에서 우로 시선 이동하며]`
- **발표 대본**:
  > "먼저 우리가 풀고자 하는 과업의 정의입니다.  
  > **분류(Classification)**가 '무엇인가', **객체 탐지(Detection)**가 '어디에 있는가'를 네모난 박스로 찾는다면,  
  > 오늘 다룰 **인스턴스 분할(Instance Segmentation)**은 '어떤 픽셀인가'까지 찾아내는 작업입니다.  
  > 특히 같은 클래스의 객체라도 서로 다른 개체로 완벽히 분리해 내야 합니다.  
  > 즉, Mask R-CNN의 최종 목표는 하나의 객체에 대해 **'무엇인지(Class), 어디인지(Box), 어떤 픽셀인지(Mask)'**를 동시에 맞추는 것입니다."

---

### [Slide 3] 02 · EVOLUTION: R-CNN (2014)
- **누적 시간**: `00:55 - 01:20` (25초)
- **슬라이드 핵심**: 2,000개 후보마다 독립 CNN 실행 $\rightarrow$ 막대한 중복 연산.
- **발표 대본**:
  > "이 목표를 달성하기 위해 계보의 출발점인 2014년 R-CNN부터 빠르게 살펴보겠습니다.  
  > R-CNN은 Selective Search로 약 2,000개의 후보 영역(RoI)을 뽑은 뒤, **후보 영역마다 독립적으로 CNN을 2,000번 반복 실행**했습니다.  
  > 딥러닝 특징 추출을 도입했다는 의의는 크지만, 연산 중복이 극심했고 CNN, SVM 분류기, 박스 회귀가 따로 노는 분리된 학습 구조라는 한계가 있었습니다."

---

### [Slide 4] 02 · EVOLUTION: Fast R-CNN (2015)
- **누적 시간**: `01:20 - 01:45` (25초)
- **슬라이드 핵심**: 전체 이미지 1회 CNN $\rightarrow$ Shared Feature Map과 RoI Pooling.
- **발표 대본**:
  > "이 중복을 해결한 것이 2015년 Fast R-CNN입니다.  
  > 이미지를 통째로 CNN에 딱 한 번만 통과시켜 **공유 특징 맵(Shared Feature Map)**을 만들고,  
  > 후보 영역 특징은 **RoI Pooling**을 통해 이 맵에서 잘라내기만 했습니다.  
  > 연산량이 획기적으로 줄고 박스와 클래스를 동시에 학습하는 Multi-task Loss를 도입했지만, 여전히 후보 박스를 뽑는 Selective Search는 네트워크 외부의 병목으로 남아 있었습니다."

---

### [Slide 5] 02 · EVOLUTION: Faster R-CNN (2015)
- **누적 시간**: `01:45 - 02:10` (25초)
- **슬라이드 핵심**: RPN(Region Proposal Network) 통합 $\rightarrow$ 진정한 End-to-End Two-Stage Detector.
- **발표 대본**:
  > "이 마지막 외부 병목을 신경망 안으로 흡수한 것이 바로 Faster R-CNN입니다.  
  > 특징 맵 위에 **Anchor Box** 개념을 적용한 **RPN(Region Proposal Network)**을 얹어 후보 영역 생성까지 GPU에서 학습 가능하게 만들었습니다.  
  > 이로써 제안과 검출이 하나로 연결된 완전한 **Two-Stage Detector**가 완성되었습니다."

---

### [Slide 6] 02 · EVOLUTION: 세 논문의 병목과 Mask R-CNN의 출발점
- **누적 시간**: `02:10 - 02:30` (20초)
- **슬라이드 핵심**: RoI 반복 $\rightarrow$ Feature 공유 $\rightarrow$ RPN 통합 $\rightarrow$ RoIAlign + Mask Branch.
- **행동 팁**: `[슬라이드 4단계 흐름 화살표를 손으로 짚으며]`
- **발표 대본**:
  > "정리하자면, R-CNN의 연산 중복을 Fast R-CNN이 피처 공유로 풀었고, 외부 제안 병목을 Faster R-CNN이 RPN으로 풀었습니다.  
  > 그렇다면 Mask R-CNN의 출발점은 무엇일까요?  
  > 바로 완성된 Faster R-CNN 검출기 위에 **'픽셀 단위 마스크'를 어떻게 최소한의 오버헤드로 올릴 것인가**였습니다. 그 해답이 **RoIAlign**과 **Mask Branch**입니다."

---

### [Slide 7] 03 · MASK R-CNN: 전체 아키텍처 (공통 Proposal, 두 RoI 경로)
- **누적 시간**: `02:30 - 03:00` (30초)
- **슬라이드 핵심**: 공통 Proposal 공유, Box 경로(7×7)와 Mask 경로(14×14)의 병렬 분기.
- **발표 대본**:
  > "Slide 7의 전체 구조를 보시겠습니다.  
  > FPN 백본과 RPN이 만든 공통 Proposal을 기반으로 두 경로가 완전히 병렬로 나뉩니다.  
  > 위쪽 탐지 경로는 기존대로 **7×7 RoIAlign** 후 FC 계층을 거쳐 클래스와 박스 오프셋을 예측합니다.  
  > 반면 아래쪽 신규 Mask 경로는 정밀한 공간 정보를 살리기 위해 **2배 더 큰 14×14 RoIAlign**을 수행한 뒤, Fully Convolution 계층을 거쳐 $K$개의 클래스별 28×28 마스크를 출력합니다."

---

### [Slide 8] 03 · MASK R-CNN: 재사용 요소 vs 신규 추가 요소
- **누적 시간**: `03:00 - 03:25` (25초)
- **슬라이드 핵심**: Backbone/FPN/RPN은 재사용, Mask RoIAlign과 Mask Head는 신규 구현.
- **발표 대본**:
  > "즉, Mask R-CNN은 바닥부터 새로 만드는 것이 아닙니다.  
  > Backbone, FPN, RPN, Box Head는 기존 검출 모듈을 100% 재사용하고,  
  > 우리는 오직 **14×14 Mask RoIAlign과 전용 Mask Head, 그리고 손실 함수**만을 새롭게 설계하여 결합하는 매우 효율적인 확장 구조를 취합니다."

---

### [Slide 9] 03 · MASK R-CNN: RoIAlign (픽셀 위치 보존)
- **누적 시간**: `03:25 - 03:55` (30초)
- **슬라이드 핵심**: RoIPool의 좌표 반올림(양자화) 오차 극복 $\rightarrow$ 실수 좌표 유지 및 쌍선형 보간(Bilinear Interpolation).
- **행동 팁**: `[양자화 오차 다이어그램을 가리키며 강조]`
- **발표 대본**:
  > "Mask R-CNN의 핵심 기술 혁신인 **RoIAlign**입니다.  
  > 기존 RoIPool은 실수 좌표를 정수로 반올림하는 **양자화(Quantization)** 과정에서 최대 1픽셀 가까운 공간 오차가 누적되었습니다. 거친 박스 탐지에는 괜찮았지만, 픽셀 1개가 생명인 마스크에는 치명적이었습니다.  
  > RoIAlign은 좌표 반올림을 완전히 없애고 실수 좌표를 그대로 유지한 채, 각 bin 내부의 4개 샘플 포인트를 **쌍선형 보간(Bilinear Interpolation)**으로 계산하여 피처 위치가 1도 어긋나지 않도록 정렬합니다."

---

### [Slide 10] 03 · MASK R-CNN: Mask Head 구조
- **누적 시간**: `03:55 - 04:30` (35초)
- **슬라이드 핵심**: 14×14×256 $\rightarrow$ 4단 Conv $\rightarrow$ 2×2 Deconv $\rightarrow$ 1×1 Conv $\rightarrow$ $K \times 28 \times 28$.
- **발표 대본**:
  > "Mask Head의 내부 텐서 흐름입니다.  
  > 14×14 크기의 RoI 피처를 입력받아 4번의 3×3 Conv로 특징을 정제하고,  
  > **2×2 Deconvolution(전치합성곱)**을 통해 해상도를 2배인 28×28로 업샘플링합니다.  
  > 마지막으로 1×1 Conv를 통해 전경 클래스 개수인 **$K$개 채널의 28×28 독립 마스크 로짓**을 출력합니다.  
  > 완전 연결 계층을 배제한 Fully Convolution 구조이기 때문에 마스크의 2차원 공간 배치가 끝까지 보존됩니다."

---

### [Slide 11] 04 · IMPLEMENTATION: 코드 ① MaskHead
- **누적 시간**: `04:30 - 05:00` (30초)
- **슬라이드 핵심**: 4단 Conv tower + ConvTranspose2d (stride=2).
- **발표 대본**:
  > "이제 직접 구현한 코드를 차례로 보시겠습니다. 먼저 코드 ① **MaskHead**입니다.  
  > `conv_tower`에서 3×3 Conv와 ReLU를 4회 반복하며 14×14 해상도와 256 채널을 유지합니다.  
  > 그리고 `self.deconv`에 주목해 주십시오.  
  > 파이토치의 `ConvTranspose2d`를 사용해 `kernel_size=2, stride=2`로 공간 해상도를 28×28로 정확히 2배 확장합니다. 보간 방식보다 윤곽선 복원 가중치를 역전파로 직접 학습할 수 있는 장점이 있습니다."

---

### [Slide 12] 04 · IMPLEMENTATION: 코드 ② MaskPredictor
- **누적 시간**: `05:00 - 05:30` (30초)
- **슬라이드 핵심**: 1×1 Conv로 $K$개 클래스별 마스크 로짓 `[N, K, 28, 28]` 생성.
- **발표 대본**:
  > "코드 ② **MaskPredictor**입니다.  
  > `nn.Conv2d`를 `kernel_size=1`로 사용하여 공간 해상도 28×28은 그대로 두고 채널만 256에서 `num_foreground_classes`, 즉 $K$개로 선형 사영합니다.  
  > COCO 데이터셋 기준 최종 출력은 `[N, 80, 28, 28]`이 되며, 각 클래스마다 전용 28×28 도안 채널을 독립적으로 갖게 됩니다."

---

### [Slide 13] 04 · IMPLEMENTATION: 코드 ③ MaskBranch
- **누적 시간**: `05:30 - 06:05` (35초)
- **슬라이드 핵심**: `MultiScaleRoIAlign` + `MaskHead` + `MaskPredictor` 통합.
- **발표 대본**:
  > "코드 ③ **MaskBranch**는 이 모듈들을 하나로 엮는 통합 forward 계층입니다.  
  > `torchvision.ops`의 `MultiScaleRoIAlign`을 선언하여 FPN 피처 맵 0번부터 3번까지 제안 박스의 크기에 맞게 자동 라우팅합니다.  
  > `output_size=14`로 고화질 피처를 뽑고, 이를 방금 정의한 `head`와 `predictor`에 연속으로 흘려보내 최종 `[sum(Ni), K, 28, 28]` 마스크 로짓을 깔끔하게 반환합니다."

---

### [Slide 14] 04 · IMPLEMENTATION: 코드 ④ Mask Target Generation
- **누적 시간**: `06:05 - 06:40` (35초)
- **슬라이드 핵심**: Positive RoI 선별 및 `roi_align(spatial_scale=1.0)`을 통한 28×28 GT 크롭.
- **행동 팁**: `[roi_align 파라미터 부분 강조]`
- **발표 대본**:
  > "코드 ④는 학습 단계에서 가장 기발한 **정답(Target) 마스크 생성 로직**입니다.  
  > Ground Truth와 매칭된 Positive RoI에 대해서만 마스크를 학습시키는데요,  
  > 놀랍게도 원본 크기의 실측 마스크 `gt_masks` 자체에 `roi_align` 연산자를 적용합니다.  
  > 원본 이미지 좌표계이므로 `spatial_scale=1.0`을 주고 `aligned=True`로 실행하면, CPU 루프 없이 GPU 상에서 수천 개의 양성 RoI에 딱 맞춘 28×28 정답 텐서 `[P_total, 28, 28]`이 밀리초 단위로 초고속 생성됩니다."

---

### [Slide 15] 04 · IMPLEMENTATION: 코드 ⑤ Class-Specific Mask Loss
- **누적 시간**: `06:40 - 07:15` (35초)
- **슬라이드 핵심**: GT 라벨 채널만 인덱싱 슬라이싱 + `F.binary_cross_entropy_with_logits`.
- **발표 대본**:
  > "코드 ⑤는 Mask R-CNN의 핵심 철학인 **Class-Mask Decoupling 손실**입니다.  
  > 80개 마스크 채널 중에서 **오직 해당 객체의 Ground Truth 라벨에 해당하는 단 1개 채널만** `mask_logits[row_idx, class_idx]`로 쏙 뽑아냅니다.  
  > 나머지 79개 채널에는 그래디언트가 흐르지 않으므로 클래스 간 간섭이 원천 차단됩니다.  
  > 그리고 파이토치의 `binary_cross_entropy_with_logits`를 적용해 수치적 언더플로우나 NaN 폭발 없이 안정적인 픽셀 손실을 계산합니다."

---

### [Slide 16] 04 · IMPLEMENTATION: 전체 학습 흐름과 5중 종합 Loss
- **누적 시간**: `07:15 - 07:45` (30초)
- **슬라이드 핵심**: $L_{total} = L_{RPN,obj} + L_{RPN,box} + L_{cls} + L_{box} + L_{mask}$.
- **발표 대본**:
  > "학습 파이프라인 전체를 조망해 보면,  
  > 논문에는 RoI 손실 3가지만 적혀 있지만, 실제 End-to-End 전체 탐지기 학습에서는 **RPN 손실 2개와 검출 손실 2개, 그리고 마스크 손실 1개가 더해진 총 5중 멀티태스크 손실($L_{total}$)**이 공통 백본에 역전파됩니다.  
  > 특히 배경 영역(Negative RoI)은 마스크 손실 계산에서 완전히 제외함으로써 마스크 헤드가 불필요하게 위축되는 현상을 방지합니다."

---

### [Slide 17] 04 · IMPLEMENTATION: 코드 ⑥ 추론(Inference) 마스크 선택
- **누적 시간**: `07:45 - 08:15` (30초)
- **슬라이드 핵심**: Box 분류 예측 라벨 채널 슬라이싱 $\rightarrow$ `sigmoid() > 0.5`.
- **발표 대본**:
  > "추론 단계인 코드 ⑥입니다.  
  > NMS까지 통과해 최종 살아남은 $D$개의 박스에 대해, 이번에는 **Box Head가 1위로 분류한 예측 라벨 채널**을 꺼냅니다.  
  > 그리고 `sigmoid()`를 씌워 0에서 1 사이 확률로 바꾸고, 0.5 임계값을 적용하여 `[D, 28, 28]` 크기의 깔끔한 불리언 이진 마스크를 생성합니다."

---

### [Slide 18] 04 · IMPLEMENTATION: 코드 ⑦ 캔버스 복원 (Post-Processing)
- **누적 시간**: `08:15 - 08:45` (30초)
- **슬라이드 핵심**: Bilinear Interpolate 확대 $\rightarrow$ 원본 이미지 Canvas Paste.
- **발표 대본**:
  > "마지막 코드 ⑦는 28×28 로컬 마스크를 **원본 고화질 이미지 좌표계로 복원**하는 기하학적 후처리입니다.  
  > 검출된 박스의 실제 가로세로 크기에 맞추어 `F.interpolate` 쌍선형 보간으로 마스크 확률을 부드럽게 확대하고,  
  > 원본 크기 영행렬 캔버스(`canvas`)의 해당 박스 슬라이싱 영역에 붙여넣습니다.  
  > 이렇게 하면 원본 이미지와 1:1로 일치하는 최종 풀 해상도 인스턴스 마스크가 완성됩니다."

---

### [Slide 19~20] 05 · EXAMPLE: 실험 입력 및 추론 결과
- **누적 시간**: `08:45 - 09:15` (30초)
- **슬라이드 핵심**: Highway Scene 차량들의 Bounding Box와 Instance Mask 동시 검출 시각화.
- **행동 팁**: `[Slide 20 결과 이미지의 차량 윤곽선을 손으로 짚으며]`
- **발표 대본**:
  > "Slide 19와 20의 실제 고속도로 주행 영상 추론 결과입니다.  
  > 겹쳐 있거나 다양한 크기를 가진 여러 대의 차량 인스턴스에 대해,  
  > 초록색, 파란색, 빨간색 등 각각 독립된 색상의 Bounding Box와 정밀한 픽셀 마스크가 오차 없이 동시에 깔끔하게 분리 검출된 것을 확인할 수 있습니다."

---

### [Slide 21] 06 · CONCLUSION: 핵심 요약 및 마무리
- **누적 시간**: `09:15 - 09:45` (30초)
- **슬라이드 핵심**: 4대 논문 계보 총정리, RoIAlign + Mask Branch 혁신, Q&A 유도.
- **발표 대본**:
  > "발표를 요약하겠습니다.  
  > R-CNN에서 시작된 객체 탐지 연구는 특징 맵 공유(Fast)와 RPN 통합(Faster)을 거쳐 완벽한 Two-Stage 탐지 체계를 갖추었습니다.  
  > 그리고 **Mask R-CNN은 픽셀 위치를 정밀 보존하는 RoIAlign과 병렬 Mask Branch를 추가**하여 탐지 속도를 거의 떨어뜨리지 않으면서도 최고 수준의 인스턴스 분할을 달성했습니다.  
  > 준비한 발표는 여기까지입니다. 경청해 주셔서 대단히 감사합니다. 질의응답 받겠습니다."

---

## 💡 발표자 전용 비상 Q&A 대비 카드 (Top 3 단골 질문)

1. **Q. 왜 Mask Branch는 Box Head처럼 FC Layer를 쓰지 않고 FCN(Conv)만 썼나요?**
   - **A**: "FC Layer는 공간 정보를 1차원 벡터로 찌그러뜨려 픽셀 간의 2차원 공간적 배치(Spatial Layout)를 파괴합니다. 마스크는 형태와 위치가 가장 중요하므로 2차원 공간성을 끝까지 보존하는 FCN 구조를 채택했습니다."

2. **Q. 마스크 손실에서 왜 Softmax 대신 Sigmoid와 Decoupling을 썼나요?**
   - **A**: "Softmax는 픽셀 하나를 두고 클래스 간에 제로섬 경합(Mutual Exclusion)을 일으킵니다. Mask R-CNN은 클래스 분류는 이미 Box Head가 잘하므로, Mask Head는 오직 형태(Shape)만 그리도록 Sigmoid로 분리하여 mAP를 5.5%p나 끌어올렸습니다."

3. **Q. RoIPool과 RoIAlign의 근본적 차이는 무엇인가요?**
   - **A**: "RoIPool은 정수 반올림(양자화)으로 인해 특징 맵과 실제 객체 위치 사이에 최대 1픽셀 수준의 어긋남이 발생합니다. RoIAlign은 반올림 없이 연속 실수 좌표를 유지하고 쌍선형 보간으로 4개 샘플 값을 계산하여 서브픽셀 정밀도를 달성합니다."
