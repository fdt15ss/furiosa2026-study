# ML 3일차 정리

> 녹취(STT) 기반 정리입니다. 음성 인식 오류가 많아 용어·명령어는 문맥으로 복원했고, 확실하지 않은 부분은 **(추정)** 으로 표시했습니다.
> '필요사' = **퓨리오사(FuriosaAI) NPU**, 'QN3' = **Qwen3**, 'FXB' = 퓨리오사 컴파일 산출물/빌드 명령으로 추정됩니다.

---

## 0. 오늘 한눈에 보기

| 순서 | 주제 | 핵심 |
|---|---|---|
| 1 | 리눅스/환경 | conda activate, `source ~/.bashrc`, apt 설치, SSH, 터미널 여러 개 운용 |
| 2 | NPU 컴파일 | Qwen3-8B-FP8을 NPU용으로 컴파일(약 2시간), 캐시·로컬 등록 |
| 3 | 양자화 개념 | 가중치 16bit → 8bit/4bit, 용량 계산, KV 캐시 |
| 4 | LLM 서버 + 툴 콜링 | 로컬 서버(포트 8000) + OpenAI 클라이언트 + tools |
| 5 | K-Fold 심화 | train_test_split 먼저 → train에만 K-Fold, 스케일러, cross_val_predict |
| 6 | all_estimators | 사이킷런 분류기 44개 일괄 평가, try/except, warnings |

---

## 1. 리눅스 / 환경 설정

### 1-1. conda 가상환경
- `conda activate <환경명>` / `conda deactivate` 가 원래 형태. 윈도우 명령 프롬프트와 똑같고, 설정이 되어 있으면 앞의 `conda`가 생략되어 보이는 것뿐.
- 터미널 앞에 `(base)`가 보이면 기본 환경에 있는 상태. `deactivate`를 해도 base는 그대로 남는다.
- 환경이 안 먹히면 새 터미널을 열고 `conda activate 환경명`을 직접 입력.

### 1-2. 환경변수/셸 설정 다시 읽기
```bash
source ~/.bashrc     # .bashrc에 적힌 설정(환경변수 등)을 지금 바로 실행
```
- "환경을 강제로 적용한다"는 의미. 앞으로 자주 쓰게 되니 외워둘 것.
- 숨김 파일 확인: `ls -a` (파일이 생성됐는지 확인할 때 사용)

### 1-3. 패키지 설치 (apt)
```bash
sudo apt update
sudo apt-get install <패키지명>
# 옵션 예: --no-install-recommends (권장 패키지 제외하고 설치)
```
- `sudo`가 안 되는 계정도 있다(권한 문제). 계정이 root인지 일반 계정인지 확인.
- 강의 중엔 컴파일 전에 필요한 패키지 2개를 apt로 먼저 설치했음(패키지명은 녹취에서 불명확).

### 1-4. SSH / 터미널 운용
- PowerShell 창을 **여러 개** 열고 역할을 나눔
  1. 상태 모니터링용 (NPU 사용 상태를 계속 띄워둠, `watch` 사용 — **(추정)** `watch -n 1 <상태조회 명령>`)
  2. 모델 서버용 (켜놓고 닫지 않음)
  3. 코드 실행/실습용
- **처음 접속이 제일 어렵고**, 한 번 붙고 나면 그다음은 쉽다. 새 도구/언어는 "Hello World까지가 고비".
- 긴 명령을 붙여넣을 때 줄 끝의 `\`(역슬래시)는 **줄 이어쓰기** 표시. 붙여넣기 하다 `\`나 공백이 꼬이면 에러가 나니 `echo` 앞뒤 등 줄바꿈 위치를 확인.
- 명령어는 다 외우는 게 아니라 **메모해두고 쓴다**(강사도 그렇게 함).

---

## 2. NPU 컴파일 (퓨리오사)

### 2-1. 컴파일이 뭔가
- 허깅페이스 모델(예: `Qwen/Qwen3-8B-FP8`)을 **NPU에서 돌릴 수 있게 맞춰주는 작업**.
- 결과물은 `.fxb`(추정) 파일로 생성됨 → `ls -a`로 루트 디렉터리에 만들어졌는지 확인.
- 소요 시간: **최소 2시간**. 진행률이 59%에서 멈춘 것처럼 보여도 **끄지 말 것** (원래 느림, 끄면 처음부터).
- 컴파일은 CPU 작업이라 다운로드 속도와 무관.

### 2-2. 캐시와 로컬 등록
- 이미 컴파일된 모델은 퓨리오사가 올려둔 것을 **캐시 폴더**로 받아 씀 (그래서 어제 캐시 폴더에 들어 있던 것).
- 직접 컴파일한 결과물은 `add` 명령으로 캐시/저장소에 **등록**하고, 이후 로컬 경로 또는 등록된 이름으로 모델을 지정해 사용.
- 캐시 확인: 캐시 폴더에서 `ls`
- **성능 차이**: 직접 컴파일 vs 캐시 모델 → 거의 없음(속도는 약간 차이 있을 수 있음). 다만 허깅페이스가 내려가도 로컬에 있으면 안전.
- 경로 에러가 나면 **루트(절대 경로)를 앞에 붙여** 지정.

### 2-3. 제약 사항
- NPU에서 모든 모델이 도는 게 아님. 퓨리오사가 **지원하는 모델 리스트**가 있고, 신모델은 업체에 요청하면 몇 주~두 달 뒤에 올려줌.
- 수업에서는 NPU 한 장(디바이스 번호는 사람마다 다름, 예: 1번이 아닌 7번)을 사용. 번호는 **상태 모니터링 화면에서 확인**.
- 큰 모델(20B급)은 컴파일은 되더라도 **메모리 부족으로 실행 불가**.
- 본 프로젝트에서는 NPU 여러 장(수업 환경 기준 96GB급 언급) 사용 가능 → 모델 크기에 맞게 선택.

---

## 3. 양자화(Quantization) 개념

### 3-1. 정의
- "대단한 게 아니라 **모델 크기를 줄이는 것**". 대신 **성능이 떨어질 수 있음**.
- 가중치 하나를 몇 비트로 표현하느냐를 줄이는 방식 (16bit → 8bit → 4bit).

### 3-2. 용량 계산 (핵심 공식)
> **용량 ≈ 파라미터 수 × 가중치 1개당 바이트 수**

| 정밀도 | 1개당 | 8B(80억) 모델 용량 |
|---|---|---|
| FP16 / BF16 (기본, 양자화 안 됨) | 2 byte | 약 **16GB** |
| FP8 / INT8 | 1 byte | 약 **8GB** |
| INT4 | 0.5 byte | 약 **4GB** |

- "8B"의 B는 **Billion = 가중치(파라미터) 개수**. "8B면 8GB"는 **FP8일 때만 맞는 말**.
- 모델 이름 끝의 `FP8`, `INT8`, `INT4` 등은 이미 양자화된 모델이라는 뜻. 숫자 16이 붙으면 보통 양자화 안 된 원본.
- 이미 양자화된 모델도 **더 줄일 수 있다** (예: FP8 → INT4).
- 엣지 컴퓨팅(작은 GPU 보드, 메모리 4GB급)에 4B 모델을 올리려면 INT4까지 줄여야 한다는 설명 (FP16 4B ≈ 8GB이므로 계산상 맞음).

### 3-3. KV 캐시
- 생성 길이/동시 요청이 늘수록 **KV 캐시가 메모리를 크게 차지**함. 모델 가중치만 보고 메모리를 계산하면 모자랄 수 있다.

### 3-4. 오늘의 결론
- 퓨리오사 쪽 사정으로 **양자화 실습은 어려움** → 내일 GPU에서 양자화 진행 예정, INT4 위주.
- 숙제(선택): 8B 말고 **4B 모델 컴파일**을 걸어놓고 퇴근.

---

## 4. 로컬 LLM 서버 + OpenAI 클라이언트 + 툴 콜링

### 4-1. 구조 (이번 수업의 3가지 방식 중 3번)
1. 코드 안에서 모델을 직접 로드
2. (이전) 모델을 코드에 넣어서 실행
3. **NPU에서 모델 서버를 켜놓고, 클라이언트가 API로 호출** ← 오늘

- 서버 옵션: 호스트 `0.0.0.0`(또는 `127.0.0.1`), **포트 8000** (수업 환경에서 열려 있는 포트는 8000번 하나).
- 서버 터미널은 **계속 켜둠**. 상태 터미널도 켜둠.
- 응답이 오는 방식이 ChatGPT API와 같아서 `openai` 라이브러리를 그대로 쓴다.

### 4-2. 왜 툴 콜링이 필요한가
- 로컬 모델은 **학습 시점까지의 지식**만 알고 있음(수년 전 모델이면 현재 정보 모름: 날씨, 시간 등).
- 그래서 **함수(툴)를 만들어 연결**해야 최신 정보를 쓸 수 있음.

### 4-3. 복습용 예시 코드 (수업 흐름을 바탕으로 재구성)
```python
import os
from openai import OpenAI

# 로컬 서버를 쓰므로 base_url만 바꿈. API 키는 형식적으로 아무 값이나 넣음
client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")

# 1) 실제 함수 2개 (이름 짓기가 알고리즘보다 어렵다!)
def get_weather(location, unit="celsius"):
    return f"{location}의 날씨: 맑음, 20도({unit})"

def get_time(timezone):
    return f"{timezone} 현재 시간: 12:00"

# 2) 모델에게 알려줄 tools 스키마
tools = [
    {"type": "function", "function": {
        "name": "get_weather",
        "description": "도시의 날씨를 알려준다",
        "parameters": {"type": "object",
                       "properties": {"location": {"type": "string"},
                                      "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}},
                       "required": ["location"]}}},
    {"type": "function", "function": {
        "name": "get_time",
        "description": "시간대의 현재 시간을 알려준다",
        "parameters": {"type": "object",
                       "properties": {"timezone": {"type": "string"}},
                       "required": ["timezone"]}}},
]

# 3) 호출
messages = [{"role": "user", "content": "서울 날씨랑 현재 시간 알려줘"}]
response = client.chat.completions.create(
    model="Qwen3-8B-FP8",        # 서버에 띄운 모델명 (환경마다 다름)
    messages=messages,
    tools=tools,
    max_tokens=1000,
)
msg = response.choices[0].message
print(msg.tool_calls)           # 모델이 "이 함수를 이 인자로 불러줘" 라고 요청한 내용
```
- 위 예시는 **모델이 어떤 툴을 부를지 결정하는 1단계**까지. 이후 툴 실행 결과를 `role: "tool"` 메시지로 넣고 다시 호출하면 최종 답변이 나옴.
- 수업에서는 `get_weather`와 `get_time` **두 개를 모두 불러서 하나의 응답으로** 합치는 것까지 확인(섭씨 단위가 어색하게 나오는 등 형태는 다듬을 것).

### 4-4. 메시지 구조 팁
- `messages`는 **리스트**(`[ ]`) 안에 **딕셔너리**(`{ }`) — `role`과 `content`.
- 대괄호/중괄호가 섞여 헷갈리면 일단 구조를 다 쓰고 실행해서 어디까지가 role이고 content인지 확인.
- `max_tokens`를 너무 작게(10) 잡으면 출력이 잘림 → 지우면 기본값, 보통 1000 정도로 설정.
- `with` 문: 열어놓은 리소스를 **자동으로 닫아주는 구문**. 보고 겁먹지 말 것.

---

## 5. K-Fold 심화 — train/test 분리와 함께 쓰기

### 5-1. 문제의식 (과적합의 순환)
1. `train_test_split`만 쓰면 → 소중한 데이터 20~30%를 테스트로 못 씀.
2. 그래서 K-Fold → 전체 데이터를 돌려 쓰지만, **K-Fold에 들어간 데이터 전체가 결국 학습에 관여** → 그 범위 밖 데이터에는 과적합 가능성.
3. 해결: **훈련에 절대 관여하지 않는 별도 테스트셋**을 먼저 떼어놓는다.
4. 그 테스트를 또 쪼개면 또 낭비... → 현실적으로는 **"train/test 분리 → train에만 K-Fold"** 가 통상적인 타협점. 귀찮으면 생략하기도 함.

### 5-2. 구조
```
전체 데이터
 ├─ X_test, y_test            (완전히 분리, 최종 평가용)
 └─ X_train, y_train          → 여기에만 K-Fold (cross_val_score)
```

### 5-3. 코드 흐름 (붓꽃 데이터, 다중분류)
```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

X, y = load_iris(return_X_y=True)

# 1) 먼저 분리: train 80%, 클래스 비율 유지(stratify)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.8, random_state=3, stratify=y)

# 2) 스케일링: train으로 fit, test는 transform만 (데이터 누수 방지)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# 3) train에만 K-Fold 교차검증
model = DecisionTreeClassifier()
kfold = KFold(n_splits=5, shuffle=True, random_state=3)
scores = cross_val_score(model, X_train, y_train, cv=kfold)
print(scores, scores.mean())

# 4) 예측 → 정확도
y_pred = cross_val_predict(model, X_test, y_test, cv=kfold)   # 수업에서 사용한 방식
print(accuracy_score(y_test, y_pred))
```
- `train_size` 기본값은 0.75 → 0.8로 조정.
- `stratify=y` → 클래스별로 8:2씩 분리됨.
- **스케일러**: 범위 밖 값 처리를 위해 적용. `fit_transform`은 train에만, test는 `transform`만.

> ⚠️ **주의(정리자 메모)**: 수업 중 `cross_val_predict(model, X_test, y_test, cv=kfold)`로 예측했는데, `cross_val_predict`는 **넘긴 데이터 안에서 폴드별로 다시 학습**합니다. 즉 X_test로 학습/예측이 같이 일어나므로, 엄밀한 "완전 분리된 최종 평가"는 아래가 정석입니다.
> ```python
> model.fit(X_train, y_train)
> y_pred = model.predict(X_test)
> print(accuracy_score(y_test, y_pred))
> ```
> 수업 흐름은 "예측 과정을 눈으로 확인하는 연습"으로 이해하고, 최종 평가는 위 방식을 병행해 보세요.

### 5-4. 기타 질문/답변
- K-Fold의 단점: 속도가 느림(폴드 수만큼 훈련), 때로는 검증 데이터 없이 전체로 학습한 결과가 더 좋게 나오기도 함.
- **시계열 데이터**는 순서가 중요 → 순서를 지키는 분할이 필요 (추가 참고: `TimeSeriesSplit`).
- **이미지(3·4차원) 데이터**도 K-Fold 가능: shape을 2차원으로 펴서 분할한 뒤 다시 원래 shape으로 복원.
- 같은 모델을 5번 돌리는 것이지 다른 모델 5개가 아님. 교차검증이므로 폴드별 점수 → 평균.
- 확인할 점: "정확도가 높다 = 좋다"가 아니라, **데이터 범위 밖에서도 맞는가**를 항상 고민.

### 5-5. 오늘 만든 실습 파일 (명명 규칙)
- 이름 규칙: `M03_KFOLD_TRAINTESTSPLIT_01.py` 형태(번호 증가, K-Fold 단독 버전과 구분).
- 기존 K-Fold 파일(아이리스, 캘리포니아, 당뇨, 산탄데르 등)을 복사해 **train_test_split + 스케일러 + K-Fold** 구조로 바꿔 3개 이상 만들기. 시간 남으면 10개까지.

---

## 6. all_estimators — 모델 일괄 평가

### 6-1. 개념
- `estimator` = 평가자/추정기, 즉 **사이킷런의 모델 클래스**. `all_estimators`는 사이킷런이 제공하는 **모든 모델을 한 번에** 가져오는 함수.

```python
from sklearn.utils import all_estimators

all_clf = all_estimators(type_filter='classifier')   # 분류기
all_reg = all_estimators(type_filter='regressor')    # 회귀기
print(len(all_clf), len(all_reg))
```
- 수업 환경 기준 **분류기 44개, 회귀기 55개** (사이킷런 버전마다 개수가 다름, 이전 버전은 43개). 버전 확인: `import sklearn; print(sklearn.__version__)` (녹취상 1.9.1 — **추정**).

### 6-2. 자료 구조 이해 (파이썬 기초 복습)
- 큰 `[ ]` = **리스트**
- 그 안의 `( )` = **튜플** → `(모델이름, 모델클래스)` 2개씩
- 그래서 `for`로 풀 때 두 개를 한 번에 받는다.
```python
for name, algorithm in all_clf:
    model = algorithm()          # 클래스를 인스턴스로 생성(기본 파라미터)
```

### 6-3. 전체 모델 평가 코드 흐름
```python
import warnings
from sklearn.utils import all_estimators
from sklearn.metrics import accuracy_score

warnings.filterwarnings('ignore')    # 경고 메시지 숨기기

for name, algorithm in all_estimators(type_filter='classifier'):
    try:
        model = algorithm()                 # #2 모델
        model.fit(X_train, y_train)         # #3 훈련
        acc = model.score(X_test, y_test)   # #4 평가
        print(name, '의 정답률:', acc)
    except:
        print(name, '모델 오류')            # 또는 pass
```
- 모델이 많아도 for문 한 번이면 44개를 돌릴 수 있음.

### 6-4. 에러 처리 포인트
- 기본 파라미터만으로는 **데이터 형태가 안 맞거나 필수 인자가 필요한 모델**은 에러가 남
  - 예: `CategoricalNB`(범주형 데이터 전용), 그 외 `VotingClassifier`·`StackingClassifier`처럼 `estimators` 인자가 필요한 앙상블 모델(추가 참고).
- 해결: **`try ~ except`** 로 에러 난 모델만 건너뛰고 계속 진행.
  - try 블록 안에서 에러 → 즉시 except 실행 → 다음 반복으로.
  - 에러 메시지도 보기 싫으면 `except: pass`.
- 경고(Warning)가 너무 많으면 `import warnings` + `warnings.filterwarnings('ignore')`.
- 결과 해석: 1.0(만점)이 하나도 없으면 과적합이 심하지 않다고 긍정적으로 볼 수 있음. `DummyClassifier`는 **기준선(baseline)** 으로 활용.

### 6-5. 오늘의 과제
- `M04_ALL_ESTIMATORS_01.py` : **분류기** 버전 (작성 완료)
- `M04_ALL_ESTIMATORS_02.py` : **회귀기** 버전 직접 만들기
  - 회귀는 정확도가 아니라 **R²** 사용 → 모델에 따라 **R²가 마이너스**로 나올 수 있음.
  - 44개(분류) 모델 전부 **cross_val** 처리까지 해보기. #2 모델부터는 직접 타이핑해서 연습.

---

## 7. 다음 수업 예고 / 숙제 체크리스트

- [ ] 4B 모델 NPU 컴파일 걸어두기 (8B는 완료)
- [ ] `M03_KFOLD_TRAINTESTSPLIT` 파일 3개 이상 작성
- [ ] `M04_ALL_ESTIMATORS` 분류/회귀 버전 작성 (회귀는 직접)
- [ ] 논문 스터디: 팀원 구하기 + 각자 논문 미리 찾아보기
- [ ] 개념 질문 대비 — 정확한 설명이 가능해야 함 (GPT를 써도 되지만 머릿속에 아키텍처를 넣을 것)
- 내일(예고): **하이퍼파라미터 자동화 튜닝**, **양자화(GPU, INT4)**, 퓨리오사 양자화는 재검토

---

## 8. 부록 — 수업 운영/일정 메모 (공부 외)

- **논문 스터디**: 4~5명 팀(24명 기준), 팀은 **12월 17일까지** 유지. 매주 **목요일 발표**.
  - 11/12: 2-stage 계열(Selective Search → R-CNN 계열 → **Mask R-CNN**까지)
  - 11/19: **Transformer**까지 (LSTM → Seq2Seq → Attention → Transformer 계보 이해 필요)
  - YOLO는 그다음 주 이후. 논문은 arXiv에서 찾고, 구현은 5~10줄 수준 코드면 충분하되 **아키텍처 이해가 핵심**.
- **팀 프로젝트**: 11월 말(26~27일경) 시작 ~ 12월 24일, 매주 **화요일 발표**, 목요일은 논문 발표.
- **참고 개념**: Selective Search는 슬라이딩 윈도우 방식의 연산량 문제(이미지 100×100을 3×3으로 훑으면 CNN을 엄청 많이 돌려야 함)를 해결하려고 나온 방식.
- **진로 조언 요약**
  - 학습(텐서플로·토치) 개념을 아는 사람은 추론/에이전트로 넘어가기 쉽지만, 반대는 어렵다 → **둘 다 하되 학습 개념(가중치, 역전파) 필수**.
  - GPU 가속/하드웨어 제어 쪽은 **C/C++** 이 필요할 수 있음.
  - 발표 실력과 포트폴리오(팀 프로젝트, 미니 프로젝트, 논문 6~7개)가 취업의 핵심.
