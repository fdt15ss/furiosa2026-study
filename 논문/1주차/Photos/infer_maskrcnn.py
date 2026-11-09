#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TensorFlow Mask R-CNN 추론 및 세그멘테이션 시각화 스크립트 (v3 - 통합 헤드 대응)
======================================================================
1. 학습된 최적 가중치(saved_models/mask_rcnn_heads_best.h5) 100% 온전하게 로드
   - feature_reduce(2048->256)부터 RoIAlign, FC1/FC2, Mask 분기까지 모두 포함
2. ResNet-50 정밀 전처리(preprocess_input) 적용
3. 이미지 입력 처리 (단일 이미지 파일 또는 폴더 내 전체 이미지)
   - JSON 어노테이션이 있는 경우: 해당 인물 RoI에 대한 정밀 세그멘테이션 마스크 추론
   - JSON이 없는 일반 이미지인 경우: Multi-scale Proposal 및 NMS 기반 자동 검출 추론
4. 원본 해상도 위에 인물별 고유 색상 마스크 반투명 오버레이 + BBox + 한글 라벨 렌더링
5. 결과 이미지를 inference_results/ 디렉터리에 저장
======================================================================
"""

import os
import sys
import glob
import json
import argparse
from collections import Counter

# Windows 콘솔 한글 인코딩 처리
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input

# ----------------------------------------------------------------------
# 1. 클래스 매핑 및 인물별 시각화 색상 정의
# ----------------------------------------------------------------------
NAME_TO_KOREAN = {
    'Kim_Gwonwoo': '김권우',
    'Kim_Sanghyun': '김상현',
    'Lee_Dongwook': '이동욱',
    'Im_Kyeongbin': '임경빈'
}

VALID_PERSON_CLASSES = ['Kim_Gwonwoo', 'Kim_Sanghyun', 'Lee_Dongwook', 'Im_Kyeongbin']


def build_class_mapping():
    sorted_persons = sorted(VALID_PERSON_CLASSES, key=lambda x: NAME_TO_KOREAN.get(x, x))
    class_map = {0: {'name': '__background__', 'korean_name': '배경'}}
    name_to_id = {'__background__': 0, 'background': 0, '배경': 0}
    for idx, name in enumerate(sorted_persons, start=1):
        kor = NAME_TO_KOREAN.get(name, name)
        class_map[idx] = {'name': name, 'korean_name': kor}
        name_to_id[name] = idx
        name_to_id[name.lower()] = idx
        name_to_id[kor] = idx
    return class_map, name_to_id, len(class_map)


CLASS_MAP, NAME_TO_ID, NUM_CLASSES = build_class_mapping()

# 인물별 고유 색상 (RGB)
CLASS_COLORS = {
    0: (128, 128, 128),  # 배경: 회색
    1: (41, 128, 185),   # 김권우: 파란색 (Blue)
    2: (39, 174, 96),    # 김상현: 초록색 (Emerald Green)
    3: (230, 126, 34),   # 이동욱: 주황색 (Orange)
    4: (155, 89, 182)    # 임경빈: 보라색 (Purple)
}


# ----------------------------------------------------------------------
# 2. 통합 Mask R-CNN 헤드 모델 아키텍처
# ----------------------------------------------------------------------
def build_integrated_mask_rcnn_head(num_classes=5, dropout_rate=0.25):
    """
    train_maskrcnn.py 와 100% 동일한 통합 헤드 모델 아키텍처
    (feature_reduce 포함 17,069,854개 파라미터 완전 복원)
    """
    backbone_feat_in = layers.Input(shape=(16, 16, 2048), name='backbone_feat_input')
    boxes_in = layers.Input(shape=(4,), name='boxes_input')
    box_indices_in = layers.Input(shape=(), dtype=tf.int32, name='box_indices_input')

    # 1) 특징 차원 축소 (2048 -> 256)
    reduced_feat = layers.Conv2D(256, (1, 1), padding='same', activation='relu', name='feature_reduce')(backbone_feat_in)

    # 2) RoIAlign (crop_and_resize)
    roi_feat = tf.image.crop_and_resize(reduced_feat, boxes_in, box_indices_in, crop_size=(7, 7))
    mask_feat = tf.image.crop_and_resize(reduced_feat, boxes_in, box_indices_in, crop_size=(14, 14))

    # 3) Classification & BBox Regressor Branch
    x = layers.Flatten(name='roi_flatten')(roi_feat)
    x = layers.Dense(1024, activation='relu', name='mrcnn_class_fc1')(x)
    x = layers.Dropout(dropout_rate, name='mrcnn_class_drop1')(x)
    x = layers.Dense(1024, activation='relu', name='mrcnn_class_fc2')(x)
    x = layers.Dropout(dropout_rate, name='mrcnn_class_drop2')(x)

    class_logits = layers.Dense(num_classes, activation='linear', name='mrcnn_class_logits')(x)
    class_probs = layers.Softmax(name='mrcnn_class')(class_logits)
    bbox_deltas = layers.Dense(num_classes * 4, activation='linear', name='mrcnn_bbox_fc')(x)

    # 4) Mask Branch
    m = layers.Conv2D(256, (3, 3), padding='same', activation='relu', name='mrcnn_mask_conv1')(mask_feat)
    m = layers.Conv2D(256, (3, 3), padding='same', activation='relu', name='mrcnn_mask_conv2')(m)
    m = layers.Conv2D(256, (3, 3), padding='same', activation='relu', name='mrcnn_mask_conv3')(m)
    m = layers.Conv2D(256, (3, 3), padding='same', activation='relu', name='mrcnn_mask_conv4')(m)
    m = layers.Conv2DTranspose(256, (2, 2), strides=2, padding='same', activation='relu', name='mrcnn_mask_deconv')(m)
    mask_logits = layers.Conv2D(num_classes, (1, 1), strides=1, activation='sigmoid', name='mrcnn_mask')(m)

    model = models.Model(
        inputs=[backbone_feat_in, boxes_in, box_indices_in],
        outputs=[class_probs, bbox_deltas, mask_logits, class_logits],
        name='mask_rcnn_integrated_head'
    )
    return model


class MaskRCNNInferenceEngine:
    def __init__(self, weights_path='saved_models/mask_rcnn_heads_best.h5'):
        self.weights_path = weights_path
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"가중치 파일을 찾을 수 없습니다: {os.path.abspath(weights_path)}")

        print("\n" + "=" * 70)
        print(" [Mask R-CNN 추론 엔진 초기화]")
        print(f"  * 가중치 파일: {os.path.abspath(weights_path)}")

        # ResNet-50 백본 로드 (ImageNet)
        self.backbone = ResNet50(
            include_top=False,
            weights='imagenet',
            input_shape=(512, 512, 3)
        )
        self.backbone.trainable = False

        # 통합 헤드 모델 로드 (feature_reduce 포함 모든 가중치 100% 복원)
        self.head_model = build_integrated_mask_rcnn_head(num_classes=NUM_CLASSES, dropout_rate=0.25)
        self.head_model.load_weights(weights_path)
        print("  * 통합 헤드 가중치 (feature_reduce + FC + Conv 전체) 로드 완료!")
        print("=" * 70 + "\n")

    def extract_features(self, img_pil):
        """이미지를 (512, 512)로 리사이즈 후 ResNet50 전처리 거쳐 백본 특징 추출"""
        img_resized = img_pil.resize((512, 512), Image.Resampling.BILINEAR)
        img_raw_arr = np.array(img_resized, dtype=np.float32)  # 0~255 RGB
        img_prep_arr = preprocess_input(img_raw_arr)
        img_tensor = tf.expand_dims(tf.convert_to_tensor(img_prep_arr, dtype=tf.float32), axis=0)

        features = self.backbone(img_tensor, training=False)
        return features  # [1, 16, 16, 2048]

    def predict_rois(self, backbone_features, boxes):
        """
        주어진 RoI 박스들에 대해 클래스 확률, BBox 보정값, 28x28 마스크 예측
        boxes: shape (N, 4) [ymin, xmin, ymax, xmax] (0.0 ~ 1.0)
        """
        boxes_tensor = tf.convert_to_tensor(boxes, dtype=tf.float32)
        n_boxes = tf.shape(boxes_tensor)[0]
        # 백본 특징은 1개 이미지이므로 batch index는 모두 0
        box_indices = tf.zeros((n_boxes,), dtype=tf.int32)

        # 백본 피처를 N개 박스에 맞게 타일링
        tiled_features = tf.repeat(backbone_features, repeats=n_boxes, axis=0)

        pred_probs, pred_deltas, pred_masks, pred_logits = self.head_model(
            [tiled_features, boxes_tensor, box_indices],
            training=False
        )
        return pred_probs.numpy(), pred_deltas.numpy(), pred_masks.numpy()


# ----------------------------------------------------------------------
# 3. RoI Proposal 생성기 (자동 탐색용)
# ----------------------------------------------------------------------
def generate_candidate_rois(orig_w, orig_h):
    rois = []
    # 1) 전체 화면
    rois.append([0.0, 0.0, 1.0, 1.0])
    
    # 2) 가로 4등분 (단체 사진 전형적 인물 배치)
    for i in range(4):
        xmin = max(0.0, i * 0.25 - 0.05)
        xmax = min(1.0, (i + 1) * 0.25 + 0.05)
        rois.append([0.1, xmin, 0.98, xmax])
        rois.append([0.2, xmin, 0.90, xmax])

    # 3) 가로 3등분
    for i in range(3):
        xmin = max(0.0, i * 0.333 - 0.05)
        xmax = min(1.0, (i + 1) * 0.333 + 0.05)
        rois.append([0.1, xmin, 0.98, xmax])

    # 4) 가로 2등분
    for i in range(2):
        xmin = max(0.0, i * 0.5 - 0.05)
        xmax = min(1.0, (i + 1) * 0.5 + 0.05)
        rois.append([0.1, xmin, 0.98, xmax])

    # 5) 중앙 전신
    rois.append([0.1, 0.2, 0.95, 0.8])
    rois.append([0.15, 0.25, 0.90, 0.75])

    return np.array(rois, dtype=np.float32)


def compute_iou(boxA, boxB):
    yA = max(boxA[0], boxB[0])
    xA = max(boxA[1], boxB[1])
    yB = min(boxA[2], boxB[2])
    xB = min(boxA[3], boxB[3])
    inter = max(0.0, yB - yA) * max(0.0, xB - xA)
    areaA = max(0.0, boxA[2] - boxA[0]) * max(0.0, boxA[3] - boxA[1])
    areaB = max(0.0, boxB[2] - boxB[0]) * max(0.0, boxB[3] - boxB[1])
    union = areaA + areaB - inter
    return inter / union if union > 0 else 0.0


def apply_nms(detections, iou_thresh=0.35):
    if len(detections) == 0:
        return []
    detections = sorted(detections, key=lambda x: x['conf'], reverse=True)
    kept = []
    while len(detections) > 0:
        best = detections.pop(0)
        kept.append(best)
        detections = [
            d for d in detections
            if compute_iou(best['box'], d['box']) < iou_thresh or best['class_id'] != d['class_id']
        ]
    return kept


# ----------------------------------------------------------------------
# 4. 세그멘테이션 시각화 렌더러 (인물 실루엣 마스크 + 외곽 윤곽선)
# ----------------------------------------------------------------------
def render_segmentation_results(img_pil, detections, mask_thresh=0.5, alpha=0.50, draw_box=False):
    """
    원본 이미지 위에 인물별 세그멘테이션 실루엣 마스크, 외곽 윤곽선(Contour), 라벨 렌더링
    - draw_box=False (기본값): 네모난 바운딩 박스를 제거하고 진짜 세그멘테이션 윤곽선만 표시
    """
    orig_w, orig_h = img_pil.size
    
    # RGBA 마스크 오버레이 레이어 생성
    mask_layer = Image.new('RGBA', (orig_w, orig_h), (0, 0, 0, 0))
    draw_layer = ImageDraw.Draw(mask_layer)

    font = None
    for fpath in ['C:/Windows/Fonts/malgun.ttf', 'C:/Windows/Fonts/gulim.ttc', 'malgun.ttf']:
        if os.path.exists(fpath):
            try:
                font = ImageFont.truetype(fpath, max(24, int(orig_h * 0.02)))
                break
            except Exception:
                pass
    if font is None:
        font = ImageFont.load_default()

    for det in detections:
        cid = det['class_id']
        conf = det['conf']
        box = det['box']
        mask_28 = det['mask_28']
        color = CLASS_COLORS.get(cid, (255, 0, 0))
        kor_name = CLASS_MAP.get(cid, {}).get('korean_name', f'Class {cid}')

        ymin, xmin, ymax, xmax = box
        px_xmin = max(0, min(orig_w - 1, int(xmin * orig_w)))
        px_ymin = max(0, min(orig_h - 1, int(ymin * orig_h)))
        px_xmax = max(px_xmin + 1, min(orig_w, int(xmax * orig_w)))
        px_ymax = max(px_ymin + 1, min(orig_h, int(ymax * orig_h)))

        roi_w = px_xmax - px_xmin
        roi_h = px_ymax - px_ymin

        # 1) 28x28 마스크를 원본 RoI 크기로 Bilinear 리사이즈
        mask_pil = Image.fromarray((mask_28 * 255).astype(np.uint8))
        mask_roi = mask_pil.resize((roi_w, roi_h), Image.Resampling.BILINEAR)
        mask_arr = np.array(mask_roi, dtype=np.float32) / 255.0
        binary_mask = (mask_arr >= mask_thresh)

        # 2) 인물 실루엣 내부 반투명 컬러 오버레이
        color_rgba = np.zeros((roi_h, roi_w, 4), dtype=np.uint8)
        color_rgba[binary_mask, 0] = color[0]
        color_rgba[binary_mask, 1] = color[1]
        color_rgba[binary_mask, 2] = color[2]
        color_rgba[binary_mask, 3] = int(255 * alpha)

        # 3) 인물 실루엣 외곽 윤곽선(Contour Edge) 추출 및 선명하게 그리기
        edge_img = Image.fromarray((binary_mask * 255).astype(np.uint8)).filter(ImageFilter.FIND_EDGES)
        edge_arr = np.array(edge_img) > 50
        color_rgba[edge_arr, 0] = color[0]
        color_rgba[edge_arr, 1] = color[1]
        color_rgba[edge_arr, 2] = color[2]
        color_rgba[edge_arr, 3] = 255  # 외곽선은 100% 불투명으로 선명하게 강조

        roi_rgba = Image.fromarray(color_rgba, mode='RGBA')
        mask_layer.paste(roi_rgba, (px_xmin, px_ymin), roi_rgba)

        # 4) 사용자가 명시적으로 원할 때만 사각형 바운딩 박스 그리기
        if draw_box:
            line_w = max(2, int(orig_h * 0.002))
            draw_layer.rectangle([px_xmin, px_ymin, px_xmax, px_ymax], outline=color + (180,), width=line_w)

        # 5) 라벨 텍스트 (인물 머리 위 상단에 표시)
        text = f"{kor_name} ({conf * 100:.1f}%)"
        if hasattr(draw_layer, 'textbbox'):
            tb = draw_layer.textbbox((px_xmin, px_ymin), text, font=font)
            text_w = tb[2] - tb[0]
            text_h = tb[3] - tb[1]
        else:
            text_w, text_h = 180, 30

        label_ymin = max(0, px_ymin - text_h - 8)
        draw_layer.rectangle(
            [px_xmin, label_ymin, px_xmin + text_w + 12, label_ymin + text_h + 8],
            fill=color + (230,)
        )
        draw_layer.text((px_xmin + 6, label_ymin + 4), text, fill=(255, 255, 255, 255), font=font)

    img_rgba = img_pil.convert('RGBA')
    combined = Image.alpha_composite(img_rgba, mask_layer)
    return combined.convert('RGB')


# ----------------------------------------------------------------------
# 5. 이미지 단일/배치 추론 실행 함수
# ----------------------------------------------------------------------
def run_inference_on_image(engine, img_path, output_dir='inference_results', conf_thresh=0.35, mask_thresh=0.5, use_gt_json=True, draw_box=False):
    os.makedirs(output_dir, exist_ok=True)
    fname = os.path.basename(img_path)
    base_name = os.path.splitext(fname)[0]
    out_path = os.path.join(output_dir, f"result_{base_name}.png")

    print("\n" + "-" * 70)
    print(f" ▶ 추론 대상 이미지: {fname}")
    print(f"   - 경로: {os.path.abspath(img_path)}")

    with Image.open(img_path) as raw_img:
        img_rgb = raw_img.convert('RGB')
        orig_w, orig_h = img_rgb.size

    # ResNet-50 백본 특징 추출
    features = engine.extract_features(img_rgb)

    # RoI 설정
    json_path = os.path.splitext(img_path)[0] + '.json'
    rois = []
    gt_labels = []

    if use_gt_json and os.path.exists(json_path):
        print(f"   * 어노테이션 JSON 발견 -> 정밀 RoI 세그멘테이션 모드로 실행")
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            w = float(data.get('imageWidth', orig_w))
            h = float(data.get('imageHeight', orig_h))
            for s in data.get('shapes', []):
                pts = np.array(s['points'], dtype=np.float32)
                xmin = float(np.min(pts[:, 0])) / w
                ymin = float(np.min(pts[:, 1])) / h
                xmax = float(np.max(pts[:, 0])) / w
                ymax = float(np.max(pts[:, 1])) / h
                rois.append([max(0.0, ymin), max(0.0, xmin), min(1.0, ymax), min(1.0, xmax)])
                gt_labels.append(s.get('label', 'Unknown'))
        except Exception as e:
            print(f"   * JSON 읽기 오류 ({e}) -> 자동 RoI 모드로 전환")
            rois = []

    is_gt_mode = len(rois) > 0
    if not is_gt_mode:
        print("   * JSON 없음 -> Candidate Grid Proposals + NMS 모드로 자동 탐색")
        rois = generate_candidate_rois(orig_w, orig_h)

    rois = np.array(rois, dtype=np.float32)

    # 모델 순전파 예측
    probs, deltas, masks = engine.predict_rois(features, rois)

    raw_detections = []
    for i in range(len(rois)):
        roi_box = rois[i]
        prob_vec = probs[i]
        cid = int(np.argmax(prob_vec))
        conf = float(prob_vec[cid])

        # 배경(0) 제외 및 임계값 필터링
        if cid == 0 or conf < conf_thresh:
            continue

        class_mask = masks[i, :, :, cid]

        raw_detections.append({
            'class_id': cid,
            'conf': conf,
            'box': roi_box,
            'mask_28': class_mask,
            'gt_label': gt_labels[i] if is_gt_mode and i < len(gt_labels) else None
        })

    if is_gt_mode:
        final_detections = raw_detections
    else:
        final_detections = apply_nms(raw_detections, iou_thresh=0.35)

    print(f"   * 검출된 인스턴스 수: {len(final_detections)}개")
    for idx, det in enumerate(final_detections, start=1):
        cid = det['class_id']
        conf = det['conf']
        kor = CLASS_MAP.get(cid, {}).get('korean_name', '알수없음')
        eng = CLASS_MAP.get(cid, {}).get('name', 'Unknown')
        ymin, xmin, ymax, xmax = det['box']
        gt_info = f" (정답 GT: {det['gt_label']})" if det['gt_label'] else ""
        print(f"     [{idx}] {kor:<5} ({eng:<14}) | 확신도: {conf*100:5.1f}% | BBox: [{ymin:.2f}, {xmin:.2f}, {ymax:.2f}, {xmax:.2f}]{gt_info}")

    # 진짜 세그멘테이션 마스크 + 외곽 실루엣 윤곽선 렌더링 (draw_box=False 기본)
    rendered_img = render_segmentation_results(img_rgb, final_detections, mask_thresh=mask_thresh, draw_box=draw_box)
    rendered_img.save(out_path, format='PNG')
    print(f"   => 세그멘테이션 결과 저장 완료: {os.path.abspath(out_path)}")
    return out_path


# ----------------------------------------------------------------------
# 6. CLI 메인 진입점
# ----------------------------------------------------------------------
def parse_arguments():
    parser = argparse.ArgumentParser(
        description="학습된 TensorFlow Mask R-CNN 모델 기반 인스턴스 세그멘테이션 추론 스크립트"
    )
    parser.add_argument(
        '--weights',
        type=str,
        default='saved_models/mask_rcnn_heads_best.h5',
        help="학습된 가중치 파일 경로 (기본값: saved_models/mask_rcnn_heads_best.h5)"
    )
    parser.add_argument(
        '--input',
        type=str,
        default='_추론용 - Group',
        help="추론할 이미지 파일 또는 폴더 경로 (기본값: _추론용 - Group)"
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='inference_results',
        help="추론 결과 이미지 저장 폴더 (기본값: inference_results)"
    )
    parser.add_argument(
        '--conf_thresh',
        type=float,
        default=0.35,
        help="인물 검출 확신도(Confidence) 임계값 (기본값: 0.35)"
    )
    parser.add_argument(
        '--mask_thresh',
        type=float,
        default=0.5,
        help="세그멘테이션 마스크 이진화 임계값 (기본값: 0.5)"
    )
    parser.add_argument(
        '--no_gt',
        action='store_true',
        help="JSON 어노테이션이 있어도 무시하고 자동 탐색(Auto Proposal) 모드로 실행"
    )
    parser.add_argument(
        '--draw_box',
        action='store_true',
        help="세그멘테이션 마스크 외에 네모난 바운딩 박스 사각형 테두리도 함께 표시"
    )
    return parser.parse_args()


def main():
    args = parse_arguments()

    target_files = []
    if os.path.isfile(args.input):
        target_files.append(args.input)
    elif os.path.isdir(args.input):
        exts = ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.PNG')
        for ext in exts:
            target_files.extend(glob.glob(os.path.join(args.input, ext)))
        target_files = sorted(list(set(target_files)))
    else:
        alt_paths = ['_추론용 - Group', 'data/Group', 'data']
        for p in alt_paths:
            if os.path.exists(p):
                for ext in ('*.jpg', '*.png'):
                    target_files.extend(glob.glob(os.path.join(p, ext)))
                if target_files:
                    break

    if not target_files:
        print(f"오류: 추론할 이미지 파일을 찾을 수 없습니다! (--input 경로를 확인하세요: {args.input})")
        sys.exit(1)

    print("\n" + "=" * 70)
    print(" [Mask R-CNN 추론 작업 시작]")
    print(f"  * 대상 이미지 수:    {len(target_files)}장")
    print(f"  * 신뢰도 임계값:      {args.conf_thresh}")
    print(f"  * 마스크 임계값:      {args.mask_thresh}")
    print(f"  * 결과 저장 디렉터리: {os.path.abspath(args.output_dir)}")
    print("=" * 70)

    engine = MaskRCNNInferenceEngine(weights_path=args.weights)

    results = []
    for img_p in target_files:
        try:
            out_p = run_inference_on_image(
                engine=engine,
                img_path=img_p,
                output_dir=args.output_dir,
                conf_thresh=args.conf_thresh,
                mask_thresh=args.mask_thresh,
                use_gt_json=not args.no_gt,
                draw_box=args.draw_box
            )
            results.append(out_p)
        except Exception as e:
            print(f"오류: {img_p} 추론 중 예외 발생: {e}")

    print("\n" + "=" * 70)
    print(f" [추론 완료] 총 {len(results)}장의 결과 이미지가 생성되었습니다.")
    print(f"  * 저장 위치: {os.path.abspath(args.output_dir)}")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    main()
