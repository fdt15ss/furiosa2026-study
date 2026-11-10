# TensorFlow Mask R-CNN 인스턴스 세그멘테이션 & 인물 식별 가이드

본 프로젝트는 **TensorFlow / Keras** 기반으로 사전학습된 ResNet-50 백본과 통합 Mask R-CNN 헤드(Integrated Head)를 활용하여 특정 인물들을 정밀하게 검출, 분류, 세그멘테이션(실루엣 마스크 추출)하는 파이프라인입니다.

---

## 📋 목차
1. [실행 환경 및 설치](#-실행-환경-및-설치)
2. [프로젝트 구조](#-프로젝트-구조)
3. [클래스 매핑 및 색상 정의](#-클래스-매핑-및-색상-정의)
4. [스크립트별 사용법](#-스크립트별-사용법)
   - [1. 모델 학습 (`train_maskrcnn.py` / `train_maskrcnn_tf.py`)](#1-모델-학습-train_maskrcnnpy--train_maskrcnn_tfpy)
   - [2. 모델 추론 및 시각화 (`infer_maskrcnn.py`)](#2-모델-추론-및-시각화-infer_maskrcnnpy)
5. [자주 묻는 질문 및 트러블슈팅](#-자주-묻는-질문-및-트러블슈팅)

---

## 💻 실행 환경 및 설치

### 1. Conda 가상환경 활성화
본 프로젝트는 TensorFlow 2.9+ GPU 환경에 최적화되어 있습니다. 다음 명령어로 가상환경을 활성화하세요.

```bash
conda activate tf29x-gpu
```

### 2. 주요 필수 라이브러리
가상환경에 아래 패키지가 설치되어 있어야 합니다.
- Python 3.10+
- TensorFlow 2.9+ (CUDA/cuDNN 호환 GPU 권장)
- Pillow (PIL)
- NumPy

---

## 📁 프로젝트 구조

```plaintext
Photos/
├── train_maskrcnn.py        # Mask R-CNN 통합 헤드 모델 학습 메인 스크립트
├── train_maskrcnn_tf.py     # train_maskrcnn.py 호출용 실행 래퍼 스크립트
├── infer_maskrcnn.py        # 학습된 모델 기반 추론 및 마스크 시각화 스크립트
├── data/                    # 학습용 데이터셋 디렉터리
│   ├── Kim_Gwonwoo/         # 인물별 이미지(.jpg) 및 LabelMe 어노테이션(.json)
│   ├── Kim_Sanghyun/
│   ├── Lee_Dongwook/
│   ├── Im_Kyeongbin/
│   ├── Group/               # 단체 사진 및 다중 인물 어노테이션
│   └── __background__/      # 배경 이미지 (선택 사항)
├── saved_models/            # 학습된 최적 가중치 저장소
│   └── mask_rcnn_heads_best.h5 # 최고 성능 가중치 파일
├── _추론용 - Group/         # 테스트/추론 대상 이미지 폴더
├── inference_results/       # 추론 결과 시각화 이미지 저장 폴더
└── README.md                # 사용법 가이드 문서
```

---

## 🎨 클래스 매핑 및 색상 정의

모델은 총 5개 클래스(배경 1개 + 인물 4명)를 분류 및 세그멘테이션합니다.

| 클래스 ID | 영문 클래스명 (`label`) | 한글 라벨 | 시각화 표시 색상 |
| :---: | :--- | :---: | :--- |
| **0** | `__background__` | 배경 | 회색 (Gray) |
| **1** | `Kim_Gwonwoo` | 김권우 | 파란색 (Blue) |
| **2** | `Kim_Sanghyun` | 김상현 | 에메랄드 초록 (Emerald Green) |
| **3** | `Lee_Dongwook` | 이동욱 | 주황색 (Orange) |
| **4** | `Im_Kyeongbin` | 임경빈 | 보라색 (Purple) |

---

## 🚀 스크립트별 사용법

### 1. 모델 학습 (`train_maskrcnn.py` / `train_maskrcnn_tf.py`)

ResNet-50 백본을 동결(Freeze)한 뒤, 특징 축소(`feature_reduce: 2048->256`), `RoIAlign`, 분류/회귀 FC 레이어, 마스크 Conv/Deconv 브랜치를 단일 통합 모델로 학습합니다.

#### 빠른 실행 (기본 설정)
```bash
python train_maskrcnn.py
```
> `train_maskrcnn_tf.py`를 실행해도 동일하게 동작합니다 (`python train_maskrcnn_tf.py`).

#### 주요 동작 특징
- **통합 헤드 가중치 저장**: 17,069,854개 파라미터가 단일 `.h5` 파일로 온전하게 저장/복원됩니다.
- **온라인 데이터 증강**: 밝기/대비 변화, 좌우 반전(50%), 미세 바운딩 박스 지터링(±2%)이 자동 적용됩니다.
- **Early Stopping & Best Restore**: `patience` 기간 동안 Total Loss 개선이 없으면 조기 종료하며, 가장 Loss가 낮았던 에폭의 최적 가중치를 자동으로 복원하여 `saved_models/mask_rcnn_heads_best.h5`에 저장합니다.

#### CLI 옵션 안내

| 옵션 | 타입 | 기본값 | 설명 |
| :--- | :---: | :---: | :--- |
| `--data_dir` | `str` | `data` | 학습 데이터셋 디렉터리 경로 |
| `--epochs` | `int` | `100` | 총 학습 에폭 수 |
| `--batch_size` | `int` | `4` | 미니배치 크기 |
| `--lr` | `float` | `0.0003` | Adam 옵티마이저 학습률 (Learning Rate) |
| `--patience` | `int` | `20` | Early Stopping 대기 에폭 수 |
| `--no_aug` | `flag` | `False` | 데이터 증강(Data Augmentation)을 비활성화할 때 지정 |
| `--no_restore_best` | `flag` | `False` | 학습 종료 시 최고 가중치 복원을 끄고 마지막 에폭 가중치 유지 |

#### 실행 예시

- **에폭 150, 배치 크기 8, 학습률 0.0002로 학습:**
  ```bash
  python train_maskrcnn.py --epochs 150 --batch_size 8 --lr 0.0002
  ```

- **데이터 증강 없이 빠르게 테스트 학습:**
  ```bash
  python train_maskrcnn.py --epochs 30 --patience 10 --no_aug
  ```

- **다른 데이터 폴더 지정:**
  ```bash
  python train_maskrcnn.py --data_dir my_custom_data
  ```

---

### 2. 모델 추론 및 시각화 (`infer_maskrcnn.py`)

학습된 가중치(`saved_models/mask_rcnn_heads_best.h5`)를 읽어와 단일 이미지 또는 폴더 전체 이미지에 대해 인스턴스 세그멘테이션을 수행하고, 결과 시각화 이미지를 저장합니다.

#### 빠른 실행 (기본 설정)
`_추론용 - Group` 폴더 내의 단체 사진을 기본으로 읽어와 추론을 수행합니다.
```bash
python infer_maskrcnn.py
```

#### 두 가지 추론 모드 지원
1. **정밀 RoI 어노테이션 모드 (자동 감지)**:
   - 이미지와 동일한 경로에 `.json` 파일이 존재하면, JSON에 등록된 인물 영역(RoI)을 기준으로 각 인물의 클래스 식별 및 28×28 정밀 실루엣 마스크를 생성합니다.
2. **자동 탐색 검출 모드 (일반 이미지 / JSON 없을 때 / `--no_gt` 사용 시)**:
   - JSON 어노테이션이 없는 일반 사진의 경우 Multi-scale Grid Candidate Proposal을 생성한 뒤 NMS(Non-Maximum Suppression, IoU 0.35) 알고리즘을 거쳐 인물을 자동 검출합니다.

#### 시각화 결과 특징
- 원본 이미지 해상도 100% 유지
- 인물 실루엣 내부 반투명 컬러 오버레이 (`alpha=0.50`)
- 인물 외곽 윤곽선(Contour Edge) 100% 불투명 강조
- 인물 머리 위에 한글 라벨 텍스트 박스 표시 (`이름 (확률%)`)
- 기본적으로 사각형 바운딩 박스를 숨겨 자연스러운 세그멘테이션 뷰를 제공하며, 필요 시 `--draw_box` 옵션으로 사각형 박스도 동시 렌더링 가능

#### CLI 옵션 안내

| 옵션 | 타입 | 기본값 | 설명 |
| :--- | :---: | :---: | :--- |
| `--weights` | `str` | `saved_models/mask_rcnn_heads_best.h5` | 로드할 학습 가중치 파일 경로 |
| `--input` | `str` | `_추론용 - Group` | 추론할 단일 이미지 파일 또는 폴더 경로 |
| `--output_dir` | `str` | `inference_results` | 추론 결과 이미지 저장 디렉터리 |
| `--conf_thresh` | `float` | `0.35` | 인물 검출 확신도(Confidence) 임계값 |
| `--mask_thresh` | `float` | `0.5` | 세그멘테이션 마스크 이진화 임계값 |
| `--no_gt` | `flag` | `False` | JSON 어노테이션이 있어도 무시하고 자동 탐색(Auto Proposal) 모드로 강제 실행 |
| `--draw_box` | `flag` | `False` | 실루엣 마스크 외에 네모난 바운딩 박스 테두리선도 함께 표시 |

#### 실행 예시

- **특정 단일 이미지 1장만 추론:**
  ```bash
  python infer_maskrcnn.py --input "data/Group/20261109_162559.jpg"
  ```

- **JSON 파일이 있어도 무시하고 순수 자동 검출(Auto Proposal + NMS)로 추론:**
  ```bash
  python infer_maskrcnn.py --input "_추론용 - Group" --no_gt
  ```

- **마스크 외곽선 + 사각형 바운딩 박스 테두리선 함께 렌더링:**
  ```bash
  python infer_maskrcnn.py --draw_box
  ```

- **신뢰도 기준을 50% 이상으로 엄격하게 필터링:**
  ```bash
  python infer_maskrcnn.py --conf_thresh 0.50
  ```

- **출력 저장 디렉터리 변경:**
  ```bash
  python infer_maskrcnn.py --output_dir "my_results"
  ```

---

## 💡 자주 묻는 질문 및 트러블슈팅

### Q1. `ModuleNotFoundError: No module named 'tensorflow'` 에러가 발생합니다.
> 현재 터미널이 base 환경이거나 TensorFlow가 설치되지 않은 환경입니다. 아래 명령어로 전용 가상환경을 활성화하세요.
> ```bash
> conda activate tf29x-gpu
> ```

### Q2. 한글 라벨 폰트가 깨지거나 네모로 나오지 않나요?
> `infer_maskrcnn.py`는 Windows 환경의 `malgun.ttf` (맑은 고딕), `gulim.ttc` (굴림)를 자동으로 검색하여 로드하므로 별도 폰트 설정 없이 한글 인물명이 정상 출력됩니다.

### Q3. 추론 결과물은 어디서 확인하나요?
> 기본 설정 시 `inference_results/result_<원본파일명>.png` 경로에 저장됩니다.
