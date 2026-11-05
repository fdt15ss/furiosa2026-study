# ML 4일차 정리

> 녹취 기반 정리입니다. 음성 인식 오류가 많아 일부 용어·수치는 문맥으로 보정했고, 코드는 수업 흐름을 재구성한 예시입니다.

## 오늘의 흐름

1. 양자화와 엣지(sLLM) 개념 복습
2. LangChain으로 전환: `ChatOpenAI`, `invoke`, `stream`, TPS 측정
3. 파이썬 기초: 모듈·패키지·`__name__`
4. 머신러닝: StratifiedKFold + 모델 44개 일괄 비교
5. 하이퍼파라미터 튜닝: 수동 반복문 → GridSearchCV(XGBoost)
6. 환경 이슈: SSH/WSL, 키 파일 문제

---

## 1. 양자화(Quantization)와 엣지 컴퓨팅

- 양자화는 모델 크기를 줄이는 방법이다. 예: float16 → float8(또는 int)로 변환.
- **파라미터 수 × 정밀도(바이트)** 로 메모리를 대략 계산한다.
  - 8B(80억) 파라미터 × 8bit(1바이트) ≈ **8GB**
- 라즈베리파이 같은 소형 기기(엣지)에 LLM(sLLM)을 올리려면 양자화를 해도 **기기 메모리보다 작게** 만들어야 한다. 8GB가 안 들어가는 장비에는 더 줄여야 한다.

---

## 2. LangChain으로 전환하기

### 설치와 import

```bash
pip install langchain-openai   # 패키지 이름은 하이픈(-)
```

> 팁: 패키지 이름을 모르겠으면 `langchain-xxx`처럼 하이픈으로 먼저 시도하고, 안 되면 검색한다.

### OpenAI SDK vs LangChain

| 구분 | OpenAI SDK | LangChain |
|---|---|---|
| 객체 | `client = OpenAI(base_url, api_key)` | `llm = ChatOpenAI(base_url, api_key, model, ...)` |
| 호출 | `client.chat.completions.create(model, messages=[{role, content}])` | `llm.invoke("질문")` |
| 출력 | `response.choices[0].message.content` | `response.content` |
| 스트리밍 | `stream=True` 후 청크의 `delta` 처리 | `for chunk in llm.stream(...)` |

- 모델명은 서버에서 자동으로 가져올 수 있다: `client.models.list().data[0].id` (한 줄로 축약 가능).
- 코드가 훨씬 간결해지는 것이 장점. 어느 쪽을 쓸지는 취향/회사 환경에 따라 바뀌므로 둘 다 자유롭게 오갈 수 있으면 된다.

### invoke 예시

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(base_url=URL, api_key=KEY, model=MODEL,
                 max_tokens=1000, temperature=0.9, top_p=1.0)

response = llm.invoke("MCP와 API의 차이점을 설명해줘")
print(response.content)
```

### stream 예시 (청크 단위 출력)

```python
for chunk in llm.stream("MCP와 API의 차이점을 설명해줘"):
    print(chunk.content, end="", flush=True)
```

- ChatGPT처럼 글자가 조금씩 나오는 방식이 스트리밍이다. 결과가 한 번에 오지 않고 **청크(chunk) 단위로 분할**되어 온다.
- OpenAI SDK에서 `stream=True`만 주고 기존처럼 `choices[0].message`를 읽으면 에러가 난다. 스트림은 청크를 순회하며 읽어야 한다. **에러는 겁먹지 말고 읽는다.**

### TPS(Tokens Per Second) 측정

- `TPS = 생성된 토큰 수 ÷ 걸린 시간(초)`. 예: 1000토큰 / 10초 = 100 TPS.
- 시간은 **응답을 받는 구간만** 잰다(요청 직전 ~ 응답 직후).

```python
import time

start = time.perf_counter()
response = llm.invoke(prompt)
end = time.perf_counter()

elapsed = round(end - start, 2)
tps = round(max_tokens / elapsed, 1)
```

- `time.time()`은 OS 시스템 시계를 따라가서 시간이 바뀌면 틀어질 수 있다. 측정용으로는 **`time.perf_counter()`** 가 더 적합하다.
- 수업 결과 예: 6.93초, 약 144 TPS (max_tokens 1000 기준).

### 생성 파라미터

| 파라미터 | 의미 |
|---|---|
| `max_tokens` | 생성할 최대 토큰 수 |
| `temperature` | 높을수록 다양/무작위, 낮을수록 일관 |
| `top_p` | 확률 높은 토큰부터 **누적 확률이 p가 될 때까지만** 후보로 사용 |

- top_p 예: 토큰 확률이 0.4, 0.2, 0.1 … 순일 때 누적 0.65~0.8까지만 후보로 쓰고 나머지는 버린다.

### 공통 설정을 모듈로 분리 (패키지화)

반복되는 접속 코드를 파일(수업에서는 `king_keras.py`)로 빼서 함수로 만든다.

```python
# king_keras.py (예시)
from langchain_openai import ChatOpenAI

def get_llm(max_tokens=1000, temperature=0.7, top_p=1.0):
    return ChatOpenAI(base_url=URL, api_key=KEY, model=MODEL,
                      max_tokens=max_tokens, temperature=temperature, top_p=top_p)
```

```python
# 사용하는 쪽
from king_keras import get_llm
llm = get_llm(max_tokens=1000, temperature=0.9, top_p=1.0)
print(llm.invoke("질문").content)
```

- 기본값(default)을 두면 인자를 생략했을 때 기본값이 쓰인다.
- **내가 만든 모듈은 같은 폴더(또는 import 경로)에 있어야 한다.**

---

## 3. 파이썬 기초: 모듈 · 패키지

### import는 "파일을 실행해서 가져오는 것"

- `import P01_car` 하는 순간 그 파일의 **최상위 코드가 전부 실행**된다. 그래서 파일 안에 `drive()` 호출이 있으면 import만 해도 출력된다.
- `import 파일` → 그 파일의 함수/변수/클래스를 `파일.함수()` 형태로 사용.
- `from 파일 import 함수` → 바로 `함수()`로 사용.

### `__name__` 과 `if __name__ == "__main__":`

- 파일을 **직접 실행**하면 `__name__`은 `"__main__"`.
- 다른 파일에서 **import**하면 `__name__`은 그 **모듈 이름**(예: `"P11_car"`)이 된다.
- 따라서 import될 때 실행되길 원치 않는 코드는 아래처럼 감싼다.

```python
def drive():
    print("운전하다")

if __name__ == "__main__":
    drive()          # 이 파일을 직접 실행할 때만 동작
```

- 이렇게 해두면 다른 곳에서 import해도 자동 실행되지 않고, 필요할 때 `drive()`를 직접 호출해서 쓴다.

### 같은 이름의 변수가 있을 때

- 가져온 `x`와 내 파일의 `x`가 있으면, 내 파일에서 정의한 값이 우선이다. 이름 충돌은 위치(스코프)를 보고 판단한다. (보통은 이런 식으로 값을 겹치게 두지 않는다.)

### 폴더 구조 = 패키지 경로

```
python/
 ├─ machine/
 │   ├─ car.py
 │   └─ tv.py
 └─ P41_import_module.py
```

```python
import machine.car
machine.car.drive()           # 전체 경로 필요

from machine import car, tv
car.drive(); tv.watch()

from machine.car import drive
from machine.tv import watch
drive(); watch()              # 이름만으로 사용
```

- `from tensorflow.keras.layers import Dense`, `from sklearn.metrics import ...` 가 **폴더 구조를 따라가는 같은 문법**이다.
- 하위 폴더(`machine/test/`)도 같은 방식으로 `machine.test.car` 처럼 가져온다.
- 노란 밑줄(경고)이 뜨면 대개 경로가 틀렸다는 뜻 → 어느 폴더에 있는지 확인.

### tensorflow는 내 폴더에 없는데 왜 import되나?

- 파이썬이 **경로(path)** 를 따라 설치된 패키지 폴더(site-packages 등)에서 찾기 때문이다. 가상환경 안에 설치되어 있고 그 경로가 걸려 있다.

### 실무 팁

- 파일 하나에 5,000줄을 몰아넣지 말고 **함수/클래스 단위로 파일을 분리**해서 import로 조합한다.
- 파이썬 기초가 약하면 매일 조금씩 보충한다.

---

## 4. 머신러닝: K-Fold 복습 + 모델 44개 일괄 비교

### StratifiedKFold

- **분류**: 라벨(y) 비율을 유지하며 분할하는 `StratifiedKFold` 사용 (`train_test_split`의 `stratify=y`와 같은 개념).
- **회귀**: y가 연속값이라 층화 분할을 쓸 수 없다 → 일반 `KFold`.
- `cross_val_score(model, X, y, cv=5)`처럼 `cv`에 숫자만 줘도 동작한다. (분류기에서는 기본적으로 층화 분할이 적용된다.) 분할 방식을 직접 통제하고 싶으면 `StratifiedKFold` 객체를 넘긴다.

```python
from sklearn.model_selection import StratifiedKFold, cross_val_score

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X_train, y_train, cv=kfold)
```

### 평가 흐름 (중요)

1. `train_test_split`으로 train/test 분리
2. **train 데이터만** 교차검증(`cross_val_score`)
3. 최종 test 데이터로 `model.score(X_test, y_test)` 확인 → 과적합 여부 판단

### `cv=5` 와 KFold 객체를 넘길 때의 차이

둘 다 "5등분 교차검증"이라 폴드 수는 같지만, **분할 방식을 누가 정하느냐**가 다르다.

| 구분 | `cv=5` (정수) | `cv=StratifiedKFold(...)` / `KFold(...)` (객체) |
|---|---|---|
| 분할기 선택 | sklearn이 자동 결정 | 내가 직접 지정 |
| 분류 문제 | `StratifiedKFold(n_splits=5)` 사용 | 내가 넘긴 객체 그대로 |
| 회귀 등 그 외 | `KFold(n_splits=5)` 사용 | 내가 넘긴 객체 그대로 |
| 셔플(`shuffle`) | **안 함**(`False`) | `shuffle=True/False` 선택 가능 |
| `random_state` | 지정 불가 | 지정 가능 → 재현성 확보 |
| 데이터 순서 영향 | 원본 순서대로 잘라서 받음 | 셔플하면 순서 영향 제거 |
| 코드 길이 | 짧고 간단 | 한 줄 더 필요 |

#### 핵심 차이 3가지

1. **셔플 여부**
   - `cv=5`는 데이터를 **섞지 않고** 앞에서부터 순서대로 5조각으로 자른다.
   - 데이터가 라벨/값 순서로 정렬되어 있으면(예: iris는 클래스별로 정렬) 폴드마다 분포가 치우칠 수 있다. 분류기는 층화 분할이 비율을 맞춰 주지만, 회귀에서 정렬된 데이터를 `cv=5`로 돌리면 폴드별 편차가 커질 수 있다.
   - KFold 객체에 `shuffle=True`를 주면 먼저 섞은 뒤 자른다.
2. **재현성**
   - `random_state`를 고정한 객체를 쓰면 **누가 몇 번 돌려도 같은 폴드 구성**이 나온다. 모델 간 비교(예: 44개 모델 일괄 비교)나 튜닝 결과 비교에서 "같은 조건"을 보장하려면 객체를 쓰는 것이 안전하다.
   - 같은 객체를 `cross_val_score`와 `GridSearchCV`에 똑같이 넘기면 평가 조건이 통일된다.
3. **제어 범위**
   - 정수는 "5등분" 외에 조절할 수 있는 것이 없다.
   - 객체를 쓰면 `shuffle`, `random_state`를 정할 수 있고, `RepeatedStratifiedKFold`, `GroupKFold`, `TimeSeriesSplit` 같은 다른 분할기로 바꾸기도 쉽다.

#### 코드로 비교

```python
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)
model = RandomForestClassifier(random_state=42)

# 1) 정수: 자동으로 StratifiedKFold(n_splits=5), shuffle 없음
scores_int = cross_val_score(model, X, y, cv=5)

# 2) 객체: 셔플 + 시드 고정
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores_obj = cross_val_score(model, X, y, cv=kfold)

print(scores_int.mean(), scores_obj.mean())   # 평균은 비슷하지만 폴드 구성이 다름
```

#### 어떻게 쓰나

- 빠르게 확인만 할 때: `cv=5`로 충분하다.
- 모델 비교, 튜닝, 보고서/제출처럼 **결과를 다시 재현해야 할 때**: 객체를 만들어 `shuffle=True, random_state=...`를 명시한다.
- 수업에서 "`cv=5`만 써도 먹히지만 제대로 먹히게 하고 싶으면 K-Fold를 따로 적용"이라고 한 말은 이 차이를 가리킨다. 분류에서는 `cv=5`도 층화 분할이 자동 적용되므로 **핵심 차이는 셔플과 시드 지정 여부**다.

### 모델 전체 일괄 실행

- 사이킷런 `all_estimators`로 분류 모델 약 44개(회귀는 50여 개)를 전부 돌려본다.
- 데이터셋 4개: `load_iris`, `load_breast_cancer`, `load_wine`, `load_digits`.
- 실무에서 쓰는 모델들이라 빠르다(전부 CPU). 보팅/배깅/스태킹 같은 앙상블에서 보조로 쓰기도 한다. 그래도 주력은 부스팅 3대장(XGBoost, LightGBM, CatBoost).

### 알고리즘 구조 (이중 for문)

```python
import numpy as np
from sklearn.utils import all_estimators

models = all_estimators(type_filter="classifier")

for idx, (X, y) in enumerate(datasets):          # 바깥 for: 데이터셋
    # 분할 → 스케일링
    names, accs = [], []
    for name, Model in models:                    # 안쪽 for: 모델 44개
        try:
            model = Model()
            scores = cross_val_score(model, X_train, y_train, cv=kfold)
            ...
            names.append(name)
            accs.append(round(acc, 4))
        except Exception:
            continue                              # 오류 나는 모델은 건너뜀

    best = np.argmax(accs)                        # 최댓값 위치
    print(data_names[idx], names[best], accs[best])
```

- 안쪽 for에서는 일일이 출력하지 않고, 바깥 for에서 **데이터셋마다 최고 모델 1개**만 출력한다.
- 에러가 나는 모델(입력 형식 불일치 등)은 `try/except` + `continue`로 제외 → 리스트에는 정상 모델만 쌓인다.
- 최댓값 갱신 알고리즘(`best_score`)과 `np.argmax`는 같은 개념: 이름 리스트와 점수 리스트의 **인덱스가 대응**되므로 `names[idx]`, `accs[idx]`로 꺼낸다.
- 시간 측정(`perf_counter`)을 처음과 끝에 넣어 총 소요 시간도 출력.

### 수업 결과 (요약)

| 데이터 | 최고 모델 | 정확도 |
|---|---|---|
| iris | ExtraTreesClassifier | 약 0.96 |
| breast_cancer | AdaBoostClassifier | 약 0.98 |
| wine | ExtraTreesClassifier | 약 0.96 |
| digits | ExtraTreesClassifier | 약 0.95 |

- 전체 소요 시간 약 84초. 작은 데이터 위주로 선정했는데도 시간이 걸린다.

---

## 5. 하이퍼파라미터 튜닝

### 5-1. 수동 반복문 (원리 이해)

- 모델: `HistGradientBoostingClassifier`, 튜닝 대상: `learning_rate`, `max_depth`.
- 각 5개씩이면 5 × 5 = **25번** 학습. 이중 for문으로 직접 구현한다.
- 최댓값 알고리즘으로 최고 점수와 그때의 파라미터를 함께 저장한다.

```python
best_score = 0
best_params = ""

for i, lr in enumerate(learning_rates):
    for j, md in enumerate(max_depths):
        model = HistGradientBoostingClassifier(learning_rate=lr, max_depth=md)
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        print(i, j, "번째 점수 계산 중")        # 진행 상황 확인용
        if score > best_score:
            best_score = score
            best_params = f"lr={lr}, max_depth={md}"
```

- 이 방식이 곧 Grid Search의 본질이다(제시한 조합을 전부 돌림).
- 후보값을 늘리고 싶으면 리스트에 값만 추가하면 된다. 단 후보가 늘면 실행 횟수가 곱으로 증가한다.

### 5-2. 튜닝에 대한 현실 인식

- 파라미터 조합이 많아지면 사실상 다 돌릴 수 없다.
- "완벽한 튜닝"은 없다. 어떤 값이 진짜 중요한지 완전히 알기는 어렵고, 자동화 도구(Optuna 등)도 알고리즘에 의지해 **좋은 값을 어느 정도 찾아주는 것**이다.
- 모델별로 영향이 큰 파라미터(예: 부스팅의 `learning_rate`, `n_estimators`, 트리의 `max_depth`) 위주로 튜닝하면 효과가 크다.

### 5-3. GridSearchCV

- Grid(촘촘한 격자) + Search + **CV(Cross Validation)**: 제시한 모든 조합을 교차검증까지 포함해 전부 학습한다.
- 총 학습 횟수 = **파라미터 조합 수 × CV 폴드 수**.
  - 예: 조합 1000개 × cv 5 = 5000번
- 이 수업의 예: 후보 42개 × 5폴드 = **210번** 학습 (iris 150행이라 약 16초).

```python
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from xgboost import XGBClassifier

param_grid = [
    {"n_estimators": [100, 500], "max_depth": [6, 10, 12]},
    {"max_depth": [6, 8, 10, 12]},
    {"learning_rate": [...], "min_child_weight": [...]},   # 수업에서 구성한 사전
]

model = XGBClassifier()
grid = GridSearchCV(model, param_grid, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                    verbose=1, n_jobs=-1)
grid.fit(X_train, y_train)
```

- `param_grid`는 **리스트 안에 딕셔너리**, 딕셔너리의 각 값은 **리스트**로 준다.
  - 한 딕셔너리 안의 파라미터끼리는 **곱**(모든 조합), 딕셔너리끼리는 **합**으로 후보 수가 계산된다.
  - 지정하지 않은 파라미터는 모델의 **기본값(default)** 이 사용된다.
- `n_jobs=-1`: 사용 가능한 **모든 CPU 코어**를 사용. 일부만 쓰려면 개수를 지정.
- `verbose`를 주면 진행 상황(`Fitting 5 folds for each of 42 candidates, totalling 210 fits`)이 출력된다.
- GridSearchCV는 모델이 아니라 **모델을 감싼(wrapping) 탐색 알고리즘**이다. 모델만 바꾸면(예: LightGBM) 같은 틀로 쓸 수 있다.

### 결과 속성

| 속성 | 의미 |
|---|---|
| `best_estimator_` | 최적 파라미터로 학습된 평가자(모델). 지정하지 않은 값은 기본값 |
| `best_params_` | 제시한 후보 중 가장 좋았던 파라미터 조합 |
| `best_score_` | 그 조합의 **교차검증 점수(학습 데이터 기반)** |
| `grid.score(X_test, y_test)` | 테스트 데이터 점수 |

- `best_score_`는 train 데이터에 대한 교차검증 결과라 **과적합 쪽으로 낙관적**일 수 있다. 일반화 성능은 **test 점수**로 확인하는 것이 좋다.
- 수업 결과 예: best_score ≈ 0.96, test score ≈ 0.94. 최적 조합은 `learning_rate=0.1`, `n_estimators=500` 근처.
- 평가 파트(`grid.score`)는 CV가 train에서만 이뤄졌다는 점을 기억하고 해석한다.

### 현역 방법

- Grid Search는 가장 무식하지만 지금도 **현역**이다. 좋은 후보값을 잘 선정하는 것이 핵심이며, 이를 보완한 방법(Random Search, Optuna 등)이 이어서 나온다.

---

## 6. 모델 가중치 저장 (대회/제출 팁)

- 머신러닝 모델(XGBoost 등)은 `joblib`이나 `pickle`로 저장하는 것이 일반적이다.
- 제출할 때는 **심사자가 가중치를 열 수 있는지**(로드 방법, 포맷)를 미리 확인하고 안내한다. 특이한 포맷으로 저장하면 열기 어려울 수 있다.

```python
import joblib
joblib.dump(model, "model.pkl")
model = joblib.load("model.pkl")
```

---

## 7. 환경 · 리눅스 · SSH 메모

- 서버 접속: PowerShell 또는 **WSL**로 SSH 접속. VS Code에서 연결이 자주 끊기면 VS Code를 완전히 종료 후 재접속.
- VS Code 터미널 열기: **Ctrl + Shift + `** (백틱, 숫자 1 옆 키).
- 가상환경 활성화 후 설치: `source <venv>/bin/activate` → `pip install ...`
- 접속이 안 되는 경우 점검 순서
  1. 키 파일 권한/경로 확인 (Windows는 `chmod` 대신 권한 설정 방식이 다름)
  2. 기존 키를 지우고 새로 받은 키로 교체
  3. 그래도 안 되면 키 재발급 문의
- 서버 안에서 Furiosa(NPU) 서버를 띄워 로컬 OpenAI 호환 엔드포인트로 LangChain을 연결해 사용.

---

## 8. 오늘 배운 핵심 한눈에 보기

- LangChain `ChatOpenAI` → `invoke()`(한 번에), `stream()`(청크 단위)
- TPS = 토큰 수 / 시간, 시간은 `perf_counter()`로 측정
- import는 파일 실행, `if __name__ == "__main__":`으로 import 시 자동 실행 방지
- 패키지 = 폴더 구조. `from 폴더.파일 import 함수`
- 분류는 StratifiedKFold, 회귀는 KFold
- `cv=5`는 자동 분할(셔플 없음), KFold 객체는 `shuffle`·`random_state`를 직접 제어 → 재현이 필요하면 객체 사용
- 평가는 train으로 CV → test로 최종 확인
- GridSearchCV 횟수 = 조합 수 × CV 폴드 수, 결과는 `best_params_`, `best_estimator_`, `best_score_`
- 튜닝은 완벽할 수 없으니 중요한 파라미터 위주로

## 다음 시간 예고

- Random Search, Optuna 등 자동화 튜닝
- 회귀 데이터셋 실습 (California, Santander, Kaggle Bike, covtype 등)
- 논문 리뷰/프로젝트 발표 준비가 병행됨
