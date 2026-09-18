import csv
import datetime
import os
import re
import sys

import numpy as np
from sklearn.metrics import mean_squared_error

def record_model_csv(model, data_shape, batch_size, history, training_time, test_loss, random_num="-1", sub_score="", csv_file_path="model_history_log.csv", train_ration = 0.7):
    # 1. 현재날짜시간
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # 2. 실행 파일명 (Jupyter 환경 등에서는 파일명이 다르게 나올 수 있음)
    file_name = os.path.basename(sys.argv[0])
    
    # 3. 모델 이름
    model_name = model.name
    
    # 4. 모델 구조 (동일한 레이어 타입이 연속되면 출력 노드 수/shape만 표기)
    structure_parts = []
    prev_layer_type = None

    # 5. epochs 추출
    epochs = len(history.history['loss'])

    structure_parts = []
    prev_layer_type = None

    for layer in model.layers:
        # 1. 레이어 타입 (예: Dense, Conv2D, Dropout, PReLU 등)
        layer_type = layer.__class__.__name__

        # 2. 출력 노드 수/차원 구하기
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

        # 3. Activation (활성화 함수) 및 레이어별 핵심 속성 추출
        extra_info = ""
        
        # Activation 확인 (안전한 이름 추출 로직)
        if hasattr(layer, 'activation') and layer.activation is not None:
            act = layer.activation
            # activation이 객체/클래스인 경우 (e.g. PReLU, LeakyReLU 객체)
            if hasattr(act, '__class__') and act.__class__.__name__ != 'function':
                act_name = act.__class__.__name__.lower()
            # 일반 함수인 경우 (e.g. relu, sigmoid)
            elif hasattr(act, '__name__'):
                act_name = act.__name__.lower()
            else:
                act_name = str(act).lower()

            # 'linear'가 아닌 경우 표기
            if act_name != 'linear':
                extra_info = f"({act_name})"

        # Dropout 레이어인 경우 비율(rate) 표시
        if layer_type == 'Dropout' and hasattr(layer, 'rate'):
            dim_str = f"({layer.rate})"

        # Conv2D 레이어인 경우 커널 사이즈 표시 (예: 32[3x3])
        if 'Conv' in layer_type and hasattr(layer, 'kernel_size'):
            k_size = "x".join(map(str, layer.kernel_size))
            extra_info = f"[{k_size}]{extra_info}"

        # 최종 노드 및 속성 조합
        node_desc = f"{dim_str}{extra_info}".strip()

        # 4. 이전 레이어와 비교하여 문자열 생성
        if layer_type == prev_layer_type:
            structure_parts.append(node_desc)
        else:
            if node_desc:
                structure_parts.append(f"{layer_type} {node_desc}")
            else:
                structure_parts.append(layer_type)
            prev_layer_type = layer_type

    # " -> " 로 연결
    model_structure = " -> ".join(structure_parts)

    total_params = model.count_params()
    
    # 5. loss 함수 및 optimizer
    # (문자열로 컴파일된 경우와 객체로 컴파일된 경우를 모두 처리)
    loss_func = model.loss if isinstance(model.loss, str) else model.loss.__name__
    optimizer_name = model.optimizer.name if hasattr(model.optimizer, 'name') else type(model.optimizer).__name__
    
    # 6. first loss, last loss
    first_loss = history.history.get('val_loss')[0] if history.history.get('val_loss') else history.history['loss'][0]
    last_loss = history.history.get('val_loss')[-1] if history.history.get('val_loss') else history.history['loss'][-1]
    
    # CSV에 기록할 데이터 리스트
    log_data = [
        current_time, file_name, data_shape, random_num, model_name, model_structure, 
        loss_func, optimizer_name, epochs, batch_size, round(training_time, 4), 
        first_loss, last_loss, test_loss, sub_score, train_ration, total_params
    ]
    
    # 파일이 존재하지 않으면 헤더(Header)를 먼저 작성
    csv_file_path="C:/study/_data/record/"+csv_file_path
    file_exists = os.path.isfile(csv_file_path)
    
    with open(csv_file_path, mode='a', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "time", "pyfile", "data_shape", "random_num", "model", "model_structure", 
                "loss", "optimizer", "epochs", "batch_size", "train_second", 
                "first_loss", "last_loss","test_loss", "sub_score", "train_ration","total_params"
            ])
        writer.writerow(log_data)
    
    # print(f"모델 학습 정보가 '{csv_file_path}'에 기록되었습니다.")

def RMSE(y_test, y_predict): #rmse 함수 정의
    return np.sqrt(mean_squared_error(y_test,y_predict))

def leaveTop(path, prefix, subfix, count, mode = "min"):
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
            candidates.append((float(match.group("score")), filename))

    candidates.sort(key=lambda item: (item[0], item[1]), reverse=mode == "max")
    deleted_files = []

    for _, filename in candidates[count:]:
        filepath = os.path.join(path, filename)
        os.remove(filepath)
        deleted_files.append(filepath)
    
    return deleted_files
