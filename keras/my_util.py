import csv
import datetime
import os
import re
import sys

import numpy as np
from sklearn.metrics import mean_squared_error


def record_model_csv(model, data_shape, batch_size, history, training_time,
                     test_loss, random_num="-1", sub_score="",
                     csv_file_path="model_history_log.csv",
                     train_ration=0.7):

    # 1. 현재날짜시간
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 2. 실행 파일명
    file_name = os.path.basename(sys.argv[0])

    # 3. 모델 이름
    model_name = model.name

    # 4. 모델 구조
    structure_parts = []
    prev_layer_type = None

    for layer in model.layers:

        # 레이어 타입
        layer_type = layer.__class__.__name__

        # 출력 노드 수/차원 구하기
        output_shape = getattr(layer, 'output_shape', None)

        if isinstance(output_shape, list):
            output_shape = output_shape[0] if output_shape else None

        elif output_shape is not None:
            try:
                output_shape = tuple(output_shape)
            except TypeError:
                output_shape = None

        if isinstance(output_shape, (tuple, list)):
            dims = [dim for dim in output_shape if dim is not None]
            dim_str = str(dims[-1]) if dims else "?"

        elif hasattr(layer, 'units') and layer.units is not None:
            dim_str = str(layer.units)

        elif hasattr(layer, 'filters') and layer.filters is not None:
            dim_str = str(layer.filters)

        else:
            dim_str = ""

        # Activation
        extra_info = ""

        if hasattr(layer, 'activation') and layer.activation is not None:
            act = layer.activation

            if hasattr(act, '__class__') and act.__class__.__name__ != 'function':
                act_name = act.__class__.__name__.lower()

            elif hasattr(act, '__name__'):
                act_name = act.__name__.lower()

            else:
                act_name = str(act).lower()

            if act_name != 'linear':
                extra_info = f"({act_name})"

        # Dropout
        if layer_type == 'Dropout' and hasattr(layer, 'rate'):
            dim_str = f"({layer.rate})"

        # Conv2D
        if 'Conv' in layer_type and hasattr(layer, 'kernel_size'):
            k_size = "x".join(map(str, layer.kernel_size))
            extra_info = f"[{k_size}]{extra_info}"

        # 최종 노드 및 속성 조합
        node_desc = f"{dim_str}{extra_info}".strip()

        # 이전 레이어와 비교
        if layer_type == prev_layer_type:
            structure_parts.append(node_desc)

        else:
            if node_desc:
                structure_parts.append(f"{layer_type} {node_desc}")
            else:
                structure_parts.append(layer_type)

            prev_layer_type = layer_type

    # 모델 구조
    model_structure = " -> ".join(structure_parts)

    # 전체 파라미터
    total_params = model.count_params()

    # 5. loss 함수 및 optimizer
    loss_func = (
        model.loss
        if isinstance(model.loss, str)
        else getattr(model.loss, '__name__', type(model.loss).__name__)
    )

    optimizer_name = (
        model.optimizer.name
        if hasattr(model.optimizer, 'name')
        else type(model.optimizer).__name__
    )

    # ==========================================================
    # history가 없거나 loss가 없는 경우에도 작동하도록 처리
    # ==========================================================

    if history is not None and hasattr(history, 'history'):

        history_dict = history.history

        loss_history = history_dict.get('loss', [])
        val_loss_history = history_dict.get('val_loss', [])

        # epochs
        epochs = len(loss_history)

        # val_loss가 있으면 val_loss 사용
        if val_loss_history:
            first_loss = val_loss_history[0]
            last_loss = val_loss_history[-1]

        # val_loss가 없으면 loss 사용
        elif loss_history:
            first_loss = loss_history[0]
            last_loss = loss_history[-1]

        # history는 있지만 loss가 없는 경우
        else:
            first_loss = ""
            last_loss = ""

    # history 자체가 없는 경우
    else:
        epochs = ""
        first_loss = ""
        last_loss = ""

    # ==========================================================

    # CSV에 기록할 데이터
    log_data = [
        current_time,
        file_name,
        data_shape,
        random_num,
        model_name,
        model_structure,
        loss_func,
        optimizer_name,
        epochs,
        batch_size,
        round(training_time, 4),
        first_loss,
        last_loss,
        test_loss,
        sub_score,
        train_ration,
        total_params
    ]

    # CSV 경로
    csv_file_path = "C:/study/_data/record/" + csv_file_path

    # 파일 존재 여부
    file_exists = os.path.isfile(csv_file_path)

    with open(
        csv_file_path,
        mode='a',
        newline='',
        encoding='utf-8-sig'
    ) as f:

        writer = csv.writer(f)

        # 파일이 없으면 Header 작성
        if not file_exists:
            writer.writerow([
                "time",
                "pyfile",
                "data_shape",
                "random_num",
                "model",
                "model_structure",
                "loss",
                "optimizer",
                "epochs",
                "batch_size",
                "train_second",
                "first_loss",
                "last_loss",
                "test_loss",
                "sub_score",
                "train_ration",
                "total_params"
            ])

        # 데이터 기록
        writer.writerow(log_data)


def RMSE(y_test, y_predict):
    # RMSE 함수
    return np.sqrt(mean_squared_error(y_test, y_predict))


def leaveTop(path, prefix, subfix, count, mode="min"):

    if count < 0:
        raise ValueError("count must be greater than or equal to 0")

    if mode not in ("min", "max"):
        raise ValueError("mode must be either 'min' or 'max'")

    score_pattern = re.compile(
        rf"^{re.escape(prefix)}.*-(?P<score>-?\d+(?:\.\d+)?){re.escape(subfix)}$"
    )

    candidates = []

    for filename in os.listdir(path):

        match = score_pattern.match(filename)

        if match:
            candidates.append(
                (
                    float(match.group("score")),
                    filename
                )
            )

    candidates.sort(
        key=lambda item: (item[0], item[1]),
        reverse=mode == "max"
    )

    deleted_files = []

    for _, filename in candidates[count:]:

        filepath = os.path.join(path, filename)

        os.remove(filepath)

        deleted_files.append(filepath)

    return deleted_files