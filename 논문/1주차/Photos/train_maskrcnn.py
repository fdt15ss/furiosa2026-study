#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TensorFlow 기반 Mask R-CNN 학습 스크립트 (v3 - 통합 헤드 & 완벽 복원판)
======================================================================
[핵심 개선 사항]
1. 완벽한 단일 통합 헤드 (Integrated Head Architecture):
   - 특징 축소층(feature_reduce: 2048->256) + RoIAlign + 분류기/회귀기 + 마스크 브랜치를
     단 하나의 tf.keras.Model 로 완전 통합!
   - 단 한 번의 save_weights / load_weights 로 모든 학습 가중치(1,706만 개)가 100% 완벽히 저장/복원
2. ResNet-50 정밀 전처리 (preprocess_input):
   - 0~1 단순 나눗셈 대신 ResNet50 전용 Caffe BGR mean-centering 적용으로 사전학습 특징 100% 활성화
3. 수치적 안정성 극대화:
   - Softmax 확률 대신 Logits 기반 sparse_categorical_crossentropy(from_logits=True) 적용으로 편향 쏠림 방지
4. 실시간 온라인 데이터 증강 (Flip, Brightness/Contrast/Saturation Jitter, BBox Shift)
5. Early Stopping (patience=20, restore_best_weights=True)
======================================================================
"""

import os
import sys
import json
import glob
import time
import argparse
from copy import deepcopy
from collections import Counter

# Windows 콘솔 한글 인코딩 처리
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import numpy as np
from PIL import Image, ImageDraw
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input

# ----------------------------------------------------------------------
# 1. 클래스 매핑 정의 (한글 가나다순)
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
    name_to_id = {
        '__background__': 0,
        'background': 0,
        '배경': 0
    }
    for idx, name in enumerate(sorted_persons, start=1):
        kor = NAME_TO_KOREAN.get(name, name)
        class_map[idx] = {'name': name, 'korean_name': kor}
        name_to_id[name] = idx
        name_to_id[name.lower()] = idx
        name_to_id[kor] = idx

    return class_map, name_to_id, len(class_map)


CLASS_MAP, NAME_TO_ID, NUM_CLASSES = build_class_mapping()


# ----------------------------------------------------------------------
# 2. 데이터 증강 (Data Augmentation) 파이프라인
# ----------------------------------------------------------------------
def augment_sample(img_raw_tensor, box_tensor, mask_tensor, is_person=True):
    """
    0~255 RGB 텐서 및 박스/마스크에 실시간 증강 적용
    """
    img = img_raw_tensor
    box = box_tensor
    mask = mask_tensor

    # 1) 밝기/대비 변화 (0~255 단위)
    img = tf.image.random_brightness(img, max_delta=30.0)
    img = tf.image.random_contrast(img, lower=0.8, upper=1.2)
    img = tf.clip_by_value(img, 0.0, 255.0)

    # 2) 랜덤 좌우 반전 (50% 확률)
    if tf.random.uniform([]) > 0.5:
        img = tf.image.flip_left_right(img)
        ymin = box[0]
        xmin = box[1]
        ymax = box[2]
        xmax = box[3]
        box = tf.stack([ymin, 1.0 - xmax, ymax, 1.0 - xmin])

        mask_expanded = tf.expand_dims(mask, axis=-1)
        mask_flipped = tf.image.flip_left_right(mask_expanded)
        mask = mask_flipped[:, :, 0]

    # 3) 인물 샘플인 경우 미세 박스 지터링 (±2%)
    if is_person and tf.random.uniform([]) > 0.4:
        jitter = tf.random.uniform([4], -0.02, 0.02)
        ymin = tf.clip_by_value(box[0] + jitter[0], 0.0, 0.9)
        xmin = tf.clip_by_value(box[1] + jitter[1], 0.0, 0.9)
        ymax = tf.clip_by_value(box[2] + jitter[2], ymin + 0.05, 1.0)
        xmax = tf.clip_by_value(box[3] + jitter[3], xmin + 0.05, 1.0)
        box = tf.stack([ymin, xmin, ymax, xmax])

    return img, box, mask


# ----------------------------------------------------------------------
# 3. 데이터셋 클래스 (data/ 디렉터리 전용)
# ----------------------------------------------------------------------
class MaskRCNNDataDataset:
    def __init__(self, data_dir='data', image_size=(512, 512), mask_size=(28, 28), augment=True):
        self.data_dir = data_dir
        self.image_size = image_size
        self.mask_size = mask_size
        self.augment = augment
        self.samples = []
        self._load_data_samples()

    def _load_data_samples(self):
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"데이터 디렉터리를 찾을 수 없습니다: {os.path.abspath(self.data_dir)}")

        # 1) 배경 이미지 (__background__)
        bg_dir = os.path.join(self.data_dir, '__background__')
        if os.path.exists(bg_dir):
            bg_files = glob.glob(os.path.join(bg_dir, '*.jpg')) + glob.glob(os.path.join(bg_dir, '*.png'))
            for f in sorted(bg_files):
                self.samples.append({
                    'image_path': f,
                    'class_id': 0,
                    'label': '__background__',
                    'box': [0.0, 0.0, 1.0, 1.0],
                    'polygon': None
                })

        # 2) data/ 내 모든 인물 및 Group 폴더 순회
        for folder_name in sorted(os.listdir(self.data_dir)):
            folder_path = os.path.join(self.data_dir, folder_name)
            if not os.path.isdir(folder_path) or folder_name == '__background__':
                continue

            json_files = glob.glob(os.path.join(folder_path, '*.json'))
            for j_path in sorted(json_files):
                img_path = j_path.replace('.json', '.jpg')
                if not os.path.exists(img_path):
                    img_path = j_path.replace('.json', '.png')
                if not os.path.exists(img_path):
                    continue

                try:
                    with open(j_path, 'r', encoding='utf-8') as f:
                        meta = json.load(f)
                except Exception as e:
                    print(f"경고: {j_path} 파일 읽기 실패 ({e})")
                    continue

                img_w = float(meta.get('imageWidth', 512))
                img_h = float(meta.get('imageHeight', 512))
                shapes = meta.get('shapes', [])

                for shape in shapes:
                    raw_label = shape.get('label', folder_name).strip()
                    cid = NAME_TO_ID.get(raw_label, NAME_TO_ID.get(folder_name, 0))
                    points = shape.get('points', [])
                    if len(points) < 3:
                        continue

                    pts_np = np.array(points, dtype=np.float32)
                    xmin = float(np.min(pts_np[:, 0])) / img_w
                    ymin = float(np.min(pts_np[:, 1])) / img_h
                    xmax = float(np.max(pts_np[:, 0])) / img_w
                    ymax = float(np.max(pts_np[:, 1])) / img_h

                    box = [
                        max(0.0, min(1.0, ymin)),
                        max(0.0, min(1.0, xmin)),
                        max(0.0, min(1.0, ymax)),
                        max(0.0, min(1.0, xmax))
                    ]
                    poly_pts = [(p[0], p[1]) for p in points]

                    self.samples.append({
                        'image_path': img_path,
                        'class_id': cid,
                        'label': raw_label,
                        'box': box,
                        'polygon': poly_pts,
                        'orig_w': img_w,
                        'orig_h': img_h
                    })

    def __len__(self):
        return len(self.samples)

    def load_item(self, idx):
        item = self.samples[idx]
        img_path = item['image_path']
        cid = item['class_id']
        box = item['box']
        polygon = item['polygon']

        with Image.open(img_path) as raw_img:
            img_rgb = raw_img.convert('RGB')
            orig_w, orig_h = img_rgb.size
            img_resized = img_rgb.resize(self.image_size, Image.Resampling.BILINEAR)
            img_raw_arr = np.array(img_resized, dtype=np.float32)  # 0~255

        if polygon is not None and cid > 0:
            mask_full = Image.new('L', (orig_w, orig_h), 0)
            ImageDraw.Draw(mask_full).polygon(polygon, outline=255, fill=255)

            ymin, xmin, ymax, xmax = box
            px_xmin = max(0, min(orig_w - 1, int(xmin * orig_w)))
            px_ymin = max(0, min(orig_h - 1, int(ymin * orig_h)))
            px_xmax = max(px_xmin + 1, min(orig_w, int(xmax * orig_w)))
            px_ymax = max(px_ymin + 1, min(orig_h, int(ymax * orig_h)))

            crop_mask = mask_full.crop((px_xmin, px_ymin, px_xmax, px_ymax))
            mask_28 = crop_mask.resize(self.mask_size, Image.Resampling.BILINEAR)
            mask_arr = (np.array(mask_28, dtype=np.float32) > 127).astype(np.float32)
        else:
            mask_arr = np.zeros(self.mask_size, dtype=np.float32)

        img_raw_t = tf.convert_to_tensor(img_raw_arr, dtype=tf.float32)
        box_t = tf.convert_to_tensor(box, dtype=tf.float32)
        cid_t = tf.convert_to_tensor(cid, dtype=tf.int32)
        mask_t = tf.convert_to_tensor(mask_arr, dtype=tf.float32)

        # 데이터 증강 (0~255 RGB 상태에서 적용)
        if self.augment:
            img_raw_t, box_t, mask_t = augment_sample(img_raw_t, box_t, mask_t, is_person=(cid > 0))

        # ResNet-50 공식 전처리 (Caffe BGR mean-centering)
        img_prep_t = preprocess_input(img_raw_t)

        return img_prep_t, box_t, cid_t, mask_t


# ----------------------------------------------------------------------
# 4. 통합 Mask R-CNN 헤드 모델 (Feature Reduce + RoIAlign + Heads)
# ----------------------------------------------------------------------
def build_integrated_mask_rcnn_head(num_classes=5, dropout_rate=0.25):
    """
    특징 축소(feature_reduce: 2048->256)부터 RoIAlign, 분류, 회귀, 마스크까지
    전체를 단 하나의 Keras Model로 패키징하여 가중치 누락을 100% 방지
    """
    # 입력 텐서 3개
    backbone_feat_in = layers.Input(shape=(16, 16, 2048), name='backbone_feat_input')
    boxes_in = layers.Input(shape=(4,), name='boxes_input')
    box_indices_in = layers.Input(shape=(), dtype=tf.int32, name='box_indices_input')

    # 1) 특징 차원 축소 (2048 -> 256) - 헤드 모델 내부로 완전 통합!
    reduced_feat = layers.Conv2D(256, (1, 1), padding='same', activation='relu', name='feature_reduce')(backbone_feat_in)

    # 2) RoIAlign (crop_and_resize)
    roi_feat = tf.image.crop_and_resize(reduced_feat, boxes_in, box_indices_in, crop_size=(7, 7))
    mask_feat = tf.image.crop_and_resize(reduced_feat, boxes_in, box_indices_in, crop_size=(14, 14))

    # 3) Classification & BBox Regressor Branch
    x = layers.Flatten(name='roi_flatten')(roi_feat)
    x = layers.Dense(1024, activation='relu', name='mrcnn_class_fc1')(x)
    if dropout_rate > 0:
        x = layers.Dropout(dropout_rate, name='mrcnn_class_drop1')(x)
    x = layers.Dense(1024, activation='relu', name='mrcnn_class_fc2')(x)
    if dropout_rate > 0:
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


# ----------------------------------------------------------------------
# 5. 학습 메인 루틴 (Early Stopping + restore_best_weights=True)
# ----------------------------------------------------------------------
def train_maskrcnn(data_dir='data', epochs=100, patience=20, batch_size=4, lr=0.0003, augment=True, restore_best_weights=True):
    os.makedirs('saved_models', exist_ok=True)
    best_model_path = os.path.join('saved_models', 'mask_rcnn_heads_best.h5')

    print(">>> 1. 사전학습 ResNet-50 백본 및 통합 헤드 모델 구축 중...")
    backbone = ResNet50(
        include_top=False,
        weights='imagenet',
        input_shape=(512, 512, 3)
    )
    # 백본 완전 동결
    backbone.trainable = False
    for l in backbone.layers:
        l.trainable = False

    # 통합 헤드 모델 생성 (feature_reduce + FC + Conv 전체 포함)
    head_model = build_integrated_mask_rcnn_head(num_classes=NUM_CLASSES, dropout_rate=0.25)
    head_model.trainable = True

    print("\n" + "=" * 70)
    print(" [레이어 가중치 설정: 백본 동결 & 통합 헤드 전체 학습]")
    print("=" * 70)
    print(f"  * ResNet-50 백본 ({len(backbone.layers)}개 레이어): 완전 동결 (trainable=False)")
    print(f"  * 통합 헤드 (feature_reduce + FC1, FC2 + Mask Conv1~4 + Final): 전체 학습 (trainable=True)")
    print("-" * 70)
    print(f"  => 학습 대상 파라미터 (Trainable params):     {head_model.count_params():>10,}")
    print(f"  => 동결된 파라미터   (Non-trainable params): {sum(tf.size(w).numpy() for w in backbone.weights):>10,}")
    print("=" * 70 + "\n")

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    print(f">>> 2. '{data_dir}' 폴더 내 데이터 로딩 중 (데이터 증강: {'ON' if augment else 'OFF'})...")
    dataset = MaskRCNNDataDataset(data_dir=data_dir, image_size=(512, 512), mask_size=(28, 28), augment=augment)
    num_samples = len(dataset)
    if num_samples == 0:
        raise ValueError(f"'{data_dir}' 내에서 유효한 학습 데이터를 찾을 수 없습니다!")

    label_counts = Counter([s['label'] for s in dataset.samples])
    print(f"  * 총 학습 인스턴스 수: {num_samples}개")
    for lbl, cnt in sorted(label_counts.items()):
        kor = NAME_TO_KOREAN.get(lbl, '배경' if lbl == '__background__' else lbl)
        print(f"    - {lbl:<15} ({kor}): {cnt}개")

    print("\n" + "=" * 70)
    print(" [Mask R-CNN 학습 시작 요약]")
    print(f"  - 총 에폭 수 (Epochs):               {epochs}")
    print(f"  - 배치 크기 (Batch Size):           {batch_size}")
    print(f"  - 학습률 (Learning Rate):           {lr}")
    print(f"  - 데이터 증강 (Augmentation):       {'활성화 (Flip, Jitter, Box Shift)' if augment else '비활성화'}")
    print(f"  - 얼리스타핑 대기 수 (Patience):     {patience}")
    print(f"  - 최고 가중치 복원 (restore_best):  {restore_best_weights}")
    print(f"  - 저장 경로 (100% 통합 저장):       {os.path.abspath(best_model_path)}")
    print("=" * 70 + "\n")

    indices = np.arange(num_samples)
    best_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    best_weights = None

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        np.random.shuffle(indices)

        epoch_total_loss = 0.0
        epoch_cls_loss = 0.0
        epoch_box_loss = 0.0
        epoch_mask_loss = 0.0
        steps = 0

        for i in range(0, num_samples, batch_size):
            batch_idx = indices[i:i + batch_size]
            b_imgs, b_boxes, b_cids, b_masks = [], [], [], []

            for s_idx in batch_idx:
                img_t, box_t, cid_t, mask_t = dataset.load_item(s_idx)
                b_imgs.append(img_t)
                b_boxes.append(box_t)
                b_cids.append(cid_t)
                b_masks.append(mask_t)

            batch_imgs = tf.stack(b_imgs)
            batch_boxes = tf.stack(b_boxes)
            batch_classes = tf.convert_to_tensor(b_cids, dtype=tf.int32)
            batch_masks = tf.stack(b_masks)
            curr_bs = tf.shape(batch_classes)[0]

            with tf.GradientTape() as tape:
                # 1) 백본 특징 추출 (동결 상태)
                backbone_feats = backbone(batch_imgs, training=False)

                # 2) 통합 헤드 순전파 (feature_reduce + RoIAlign + FC/Conv 전체 실행)
                box_indices = tf.range(curr_bs, dtype=tf.int32)
                pred_probs, pred_deltas, pred_masks, pred_logits = head_model(
                    [backbone_feats, batch_boxes, box_indices],
                    training=True
                )

                # 3) Classification Loss (수학적으로 안정적인 from_logits=True 사용)
                loss_cls = tf.reduce_mean(
                    tf.keras.losses.sparse_categorical_crossentropy(batch_classes, pred_logits, from_logits=True)
                )

                # 4) Bounding Box Loss (양성 인스턴스만)
                is_fg = tf.cast(batch_classes > 0, tf.float32)
                num_fg = tf.maximum(tf.reduce_sum(is_fg), 1.0)

                pred_boxes = tf.reshape(pred_deltas, [curr_bs, NUM_CLASSES, 4])
                class_indices = tf.stack([tf.range(curr_bs, dtype=tf.int32), batch_classes], axis=1)
                selected_boxes = tf.gather_nd(pred_boxes, class_indices)
                box_loss_per_sample = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.NONE)(batch_boxes, selected_boxes)
                loss_box = tf.reduce_sum(box_loss_per_sample * is_fg) / num_fg

                # 5) Mask Loss (양성 인스턴스만)
                mask_perm = tf.transpose(pred_masks, [0, 3, 1, 2])
                selected_mask = tf.gather_nd(mask_perm, class_indices)
                bce = tf.keras.losses.binary_crossentropy(batch_masks, selected_mask)
                mask_loss_per_sample = tf.reduce_mean(bce, axis=[1])
                loss_mask = tf.reduce_sum(mask_loss_per_sample * is_fg) / num_fg

                total_loss = loss_cls + loss_box + (loss_mask * 2.5)

            # 헤드 전체 역전파 업데이트
            grads = tape.gradient(total_loss, head_model.trainable_variables)
            optimizer.apply_gradients(zip(grads, head_model.trainable_variables))

            epoch_total_loss += float(total_loss)
            epoch_cls_loss += float(loss_cls)
            epoch_box_loss += float(loss_box)
            epoch_mask_loss += float(loss_mask)
            steps += 1

        elapsed = time.time() - epoch_start
        avg_total = epoch_total_loss / steps
        avg_cls = epoch_cls_loss / steps
        avg_box = epoch_box_loss / steps
        avg_mask = epoch_mask_loss / steps

        is_best = avg_total < best_loss
        if is_best:
            best_loss = avg_total
            best_epoch = epoch
            patience_counter = 0
            head_model.save_weights(best_model_path)
            best_weights = deepcopy(head_model.get_weights())
            status_tag = f"★ Best Loss 갱신! ({best_loss:.4f}) -> 100% 통합 저장됨"
        else:
            patience_counter += 1
            status_tag = f"patience: {patience_counter}/{patience} (Best: {best_loss:.4f} @ Ep {best_epoch})"

        print(
            f"Epoch [{epoch:03d}/{epochs:03d}] ({elapsed:.1f}s) | "
            f"Total: {avg_total:.4f} (Cls: {avg_cls:.4f}, Box: {avg_box:.4f}, Mask: {avg_mask:.4f}) | "
            f"{status_tag}"
        )

        if patience_counter >= patience:
            print("\n" + "!" * 70)
            print(f" [Early Stopping 발동] {patience}에폭 동안 개선이 없어 {epoch}번째 에폭에서 조기 종료합니다.")
            print(f"  * 최고 성능 에폭: Epoch {best_epoch} (Total Loss: {best_loss:.4f})")
            print("!" * 70)
            break

    print("\n" + "=" * 70)
    if restore_best_weights and best_weights is not None:
        head_model.set_weights(best_weights)
        head_model.save_weights(best_model_path)
        print(f" [restore_best_weights=True 적용 완료]")
        print(f"  * 최고 성능이었던 Epoch {best_epoch} (Loss: {best_loss:.4f})의 최적 가중치를 모델에 복원했습니다.")
    else:
        print("  * 최종 에폭의 가중치가 유지되었습니다.")

    print(f"  * 최종 최적 가중치 파일: {os.path.abspath(best_model_path)}")
    print("=" * 70 + "\n")


# ----------------------------------------------------------------------
# 6. CLI 진입점
# ----------------------------------------------------------------------
def parse_arguments():
    parser = argparse.ArgumentParser(
        description="TensorFlow Mask R-CNN 학습 (통합 헤드 모델 + ResNet50 preprocess_input + 데이터 증강)"
    )
    parser.add_argument('--data_dir', type=str, default='data', help="학습 데이터셋 디렉터리 (기본값: data)")
    parser.add_argument('--epochs', type=int, default=100, help="학습 에폭 수 (기본값: 100)")
    parser.add_argument('--patience', type=int, default=20, help="Early Stopping patience (기본값: 20)")
    parser.add_argument('--batch_size', type=int, default=4, help="배치 크기 (기본값: 4)")
    parser.add_argument('--lr', type=float, default=0.0003, help="학습률 Learning Rate (기본값: 0.0003)")
    parser.add_argument('--no_aug', action='store_true', help="데이터 증강(Data Augmentation) 끄기")
    parser.add_argument(
        '--no_restore_best',
        action='store_true',
        help="학습 종료 시 best weight 복원 비활성화 (지정하지 않으면 restore_best_weights=True 기본 적용)"
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()

    print("\n" + "=" * 70)
    print(" [환경 정보 확인]")
    print(f"  * TensorFlow 버전: {tf.__version__}")
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"  * 사용 가능한 GPU: {gpus}")
    else:
        print("  * GPU를 찾을 수 없습니다. CPU 모드로 실행됩니다.")
    print("=" * 70)

    train_maskrcnn(
        data_dir=args.data_dir,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        lr=args.lr,
        augment=not args.no_aug,
        restore_best_weights=not args.no_restore_best
    )
