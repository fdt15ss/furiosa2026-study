# YOLO v1~v26 arXiv 논문 모음

## 버전별 목록

| 버전 | 상태 | 파일 | arXiv | 제목 / 저자 (첫 게시일) |
|---|---|---|---|---|
| v1 | ✅ 공식 | `YOLOv1_1506.02640v5.pdf` | [1506.02640](https://arxiv.org/abs/1506.02640) v5 | You Only Look Once: Unified, Real-Time Object Detection — Redmon, Divvala, Girshick, Farhadi (2015-06-08) |
| v2 | ✅ 공식 | `YOLOv2_YOLO9000_1612.08242v1.pdf` | [1612.08242](https://arxiv.org/abs/1612.08242) v1 | YOLO9000: Better, Faster, Stronger — Redmon, Farhadi (2016-12-25). YOLOv2와 YOLO9000을 함께 다룬다 |
| v3 | ✅ 공식 | `YOLOv3_1804.02767v1.pdf` | [1804.02767](https://arxiv.org/abs/1804.02767) v1 | YOLOv3: An Incremental Improvement — Redmon, Farhadi (2018-04-08) |
| v4 | ✅ 공식 | `YOLOv4_2004.10934v1.pdf` | [2004.10934](https://arxiv.org/abs/2004.10934) v1 | YOLOv4: Optimal Speed and Accuracy of Object Detection — Bochkovskiy, Wang, Liao (2020-04-23) |
| v5 | ❌ 논문 없음 | (아래 참고 자료) | — | 아래 "없는 버전" 참고 |
| v6 | ✅ 공식 (2편) | `YOLOv6_2209.02976v1.pdf`, `YOLOv6-v3.0_2301.05586v1.pdf` | [2209.02976](https://arxiv.org/abs/2209.02976), [2301.05586](https://arxiv.org/abs/2301.05586) | YOLOv6: A Single-Stage Object Detection Framework for Industrial Applications — Li 외, Meituan (2022-09-07) / YOLOv6 v3.0: A Full-Scale Reloading (2023-01-13) |
| v7 | ✅ 공식 | `YOLOv7_2207.02696v1.pdf` | [2207.02696](https://arxiv.org/abs/2207.02696) v1 | YOLOv7: Trainable bag-of-freebies… — Wang, Bochkovskiy, Liao (2022-07-06). v6보다 arXiv 게시는 빠르다 |
| v8 | ❌ 논문 없음 | (아래 참고 자료) | — | 아래 "없는 버전" 참고 |
| v9 | ✅ 공식 | `YOLOv9_2402.13616v2.pdf` | [2402.13616](https://arxiv.org/abs/2402.13616) v2 | YOLOv9: Learning What You Want to Learn Using Programmable Gradient Information — Wang, Yeh, Liao (2024-02-21) |
| v10 | ✅ 공식 | `YOLOv10_2405.14458v2.pdf` | [2405.14458](https://arxiv.org/abs/2405.14458) v2 | YOLOv10: Real-Time End-to-End Object Detection — Wang 외, 칭화대 (2024-05-23) |
| 11 | ❌ 논문 없음 | (아래 참고 자료) | — | 아래 "없는 버전" 참고 |
| v12 | ✅ 공식 | `YOLOv12_2502.12524v1.pdf` | [2502.12524](https://arxiv.org/abs/2502.12524) v1 | YOLOv12: Attention-Centric Real-Time Object Detectors — Tian, Ye, Doermann (2025-02-18) |
| v13 | ✅ 공식 | `YOLOv13_2506.17733v2.pdf` | [2506.17733](https://arxiv.org/abs/2506.17733) v2 | YOLOv13: Real-Time Object Detection with Hypergraph-Enhanced Adaptive Visual Perception — Lei 외, iMoonLab (2025-06-21) |
| v14 | ⚠️ 철회됨 | `withdrawn/YOLOv14_WITHDRAWN_2608.04720v3.pdf` | [2608.04720](https://arxiv.org/abs/2608.04720) v3 | YOLOv14: Adaptive Real-Time Object Detection for Diverse Imaging Conditions — Jian Lu 외 (2026-08-05). 아래 참고 |
| v15~v25 | ❌ 해당 버전 없음 | — | — | 아래 참고 |
| 26 | ✅ 공식 | `YOLO26_Ultralytics-official_2606.03748v1.pdf` | [2606.03748](https://arxiv.org/abs/2606.03748) v1 | Ultralytics YOLO26: Unified Real-Time End-to-End Vision Models — Jocher, Qiu, Liu 외, Ultralytics (2026-06-02) |

## 없는 버전·주의할 버전의 사유

### v5, v8, 11 — Ultralytics가 논문 없이 코드·문서로만 공개
- 세 버전 모두 Ultralytics가 GitHub 저장소와 온라인 문서로 공개했고 arXiv 공식 논문은 없다.
- 근거 1. Ultralytics의 YOLO26 공식 논문(2606.03748)이 참고문헌에서 이 버전들을 이렇게 적었다.
  - YOLOv5: Zenodo의 코드 공개본 "Ultralytics/yolov5: initial release (2020)"
  - YOLOv8: 문서 링크와 함께 "Online documentation (no formal paper)"
  - YOLO11: 문서 링크와 함께 "Online documentation (no formal paper)"
- 근거 2. [Ultralytics YOLO11 문서](https://docs.ultralytics.com/models/yolo11/)는 모델이 빠르게 바뀌어 정식 논문을 내지 않았다고 설명하고, 소프트웨어 인용(@software)을 안내한다.
- 대신 공부할 자료
  - 공식 문서: [YOLOv5](https://docs.ultralytics.com/models/yolov5/), [YOLOv8](https://docs.ultralytics.com/models/yolov8/), [YOLO11](https://docs.ultralytics.com/models/yolo11/), 그리고 [GitHub ultralytics](https://github.com/ultralytics/ultralytics)·[yolov5](https://github.com/ultralytics/yolov5) 코드
  - 비공식 개관 논문(같이 넣음): `no-official-paper-reference/UNOFFICIAL_Ultralytics-YOLO-overview_v5-v8-11-26_2510.09653v4.pdf`
    - [arXiv 2510.09653](https://arxiv.org/abs/2510.09653) v4, Sapkota·Karkee, "Ultralytics YOLO Evolution: An Overview of YOLO27, YOLO26, YOLO11, YOLOv8, and YOLOv5…"
    - Ultralytics가 쓴 논문이 아니라 제3자의 정리 자료다. 공식 출처로 인용하지 않는다.
- 이름 표기: Ultralytics는 YOLO11부터 "v"를 빼고 `YOLO11`, `YOLO26`으로 쓴다.

### v14 — 논문은 있으나 저자가 철회
- arXiv 2608.04720의 최신판(v4)은 철회 표시이고, 저자 코멘트는 "Sorry, we need to evaluate and revise the paper more scientifically"다.
- 내용이 있는 마지막 판본 v3를 `withdrawn/` 폴더에 참고용으로 넣었다. 저자가 철회한 논문이니 결과를 근거로 인용하지 않는다.
- 저자는 Jian Lu 외이며 원래 YOLO 저자나 Ultralytics와 관계가 확인되지 않은 별도 연구다. v1 판의 제목도 지금과 달랐다("Unified Cross-Domain Real-Time Object Detection with Adaptive Multi-View Representation").

### v15~v25 — 그런 버전이 없음
- arXiv API로 제목 `YOLOv15`~`YOLOv25`와 `YOLO15`~`YOLO25`를 검색했지만 해당 논문은 없었다. 초록 검색에 걸린 것은 이름이 비슷한 다른 논문뿐이었다.
- Ultralytics는 YOLO11 다음 모델을 YOLO26으로 공개했다. 2025년 9월에 발표했고 2026년 1월 출시 보도자료를 냈다.
- 번호를 건너뛴 공식 이유는 찾지 못했다. 출시 연도 2026을 따른 이름이라는 해석이 흔하지만 공식 문서·논문에서는 확인하지 못했다.
- 그 사이 번호인 v12·v13은 Ultralytics가 아닌 다른 연구팀의 논문이다. 버전 번호를 한 조직이 관리하지 않아 순서가 실제 계보와 다를 수 있다.

### 26 — 다른 YOLO26 논문들(포함하지 않음)
- Ultralytics 공식 논문(2606.03748)만 넣었다.
- 아래는 제3자 분석이라 넣지 않았다. 필요하면 링크로 확인한다.
  - [2509.25164](https://arxiv.org/abs/2509.25164) Sapkota 외, 성능 비교
  - [2601.12882](https://arxiv.org/abs/2601.12882) Chakrabarty, 분석 리뷰
  - [2602.14582](https://arxiv.org/abs/2602.14582) Hidayatullah 외, 구조 개관

## 저자·기관 계보 (이어 읽을 때 참고)
- **v1~v3**: Redmon(·Farhadi). Redmon은 v3 이후 연구를 그만뒀다.
- **v4·v7·v9**: Bochkovskiy, Chien-Yao Wang, Hong-Yuan Mark Liao 계열. v4는 Darknet을 이었다.
- **v5·v8·11·26**: Ultralytics.
- **v6**: Meituan.
- **v10**: 칭화대.
- **v12**: Tian 외.
- **v13**: iMoonLab.
