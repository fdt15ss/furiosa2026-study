# LLM 18일차 정리 — 로컬 LLM 양자화 (WSL + Ollama + llama.cpp)

> 녹취(STT) 기반 정리입니다. 음성 인식 오류가 많아 모델명·수치는 문맥으로 보정했고, 확실하지 않은 부분은 `(확인 필요)`로 표시했습니다.
> (예: 잼마/젬마 → **Gemma**, 필요사 → **퓨리오사(FuriosaAI)**, 올라마 → **Ollama**, 컨트라이제이션 → **Quantization**)

---

## 0. 오늘의 한 줄 요약

**HuggingFace의 BF16 원본 모델(Gemma 4 E2B)을 받아 → GGUF로 변환 → Q4_K_M으로 양자화 → Ollama에 등록해서 실행**까지 직접 해봤다.
핵심은 "**가중치(파라미터)의 표현 비트 수를 줄여서 용량·메모리를 줄이는 것**"이고, 명령어는 많아 보여도 결국 순서만 따라가면 되는 작업이다.

---

## 1. 양자화(Quantization) 개념

### 1-1. 왜 하는가
- 8GB짜리 장비(Jetson Nano, Raspberry Pi 8GB 등)에 원본(FP16/BF16) 모델은 **안 올라간다** → 줄여서 올려야 한다.
- 엣지 컴퓨팅, 오프라인(인터넷 불가) 환경, 소형 자율주행 장비 등에서 필요.
  - 강사 사례: 인터넷이 안 되는 지역(캄보디아 교육 사업)에서 로컬 AI 모델을 요구 → 100% 로컬 구동 필요
- 로컬 LLM은 **API 토큰 비용 0원**, 인터넷을 끊어도 동작.

### 1-2. 용량 계산법 (가장 중요)

```
모델 용량 ≈ 파라미터 개수 × 파라미터 1개당 바이트 수
```

| 자료형 | 비트 | 1개당 바이트 | 4B(40억) 모델 용량 |
|---|---|---|---|
| FP32 | 32bit | 4B | 약 16GB |
| FP16 / BF16 | 16bit | 2B | 약 **8GB** |
| FP8 / INT8 | 8bit | 1B | 약 4GB |
| INT4 | 4bit | 0.5B | 약 2GB |

- "반땡"이라고 생각하면 쉽다: 16 → 8 → 4 비트로 갈 때마다 절반씩 줄어든다.
- 모델 이름에 **BF16 / FP16 / F16**이 붙어 있으면 → **양자화 안 된 원본**이라고 보면 된다.
  - BF16 = Brain Float 16 (구글 브레인 방식), FP16 = Float 16 (부동소수점 16비트)
- 양자화 = 정밀도(표현력)를 깎는 압축. 예를 들어 소수점 아래를 날려버리는 식 → **정보 손실**이 생긴다.
- 이미 학습이 끝난 **최적 가중치**를 압축하는 것이므로, 어느 정도까지는 성능 저하가 작지만 비트를 많이 줄일수록 손실이 커진다.

### 1-3. 양자화할 때 주의할 점
- **한 번만 한다.** 이미 양자화된 모델을 또 양자화(재양자화)하면 성능이 많이 떨어져서 "안 쓰는 게 낫다" 수준이 될 수 있음. 항상 **FP16/BF16 원본에서 1회**.
- 보통 2의 배수(16→8→4)로 줄인다. (6비트 등은 애매함)
- 양자화 방식은 **NPU/GPU/CPU 종류마다 조금씩 다르다.** 퓨리오사 NPU는 별도 방식(FX 계열로 언급됨, 확인 필요), Jetson은 이번에 한 GGUF 방식으로 작업했다고 함.
- 반드시 성능이 좋아지는 게 아니라 **메모리·용량을 줄이는 대신 성능을 일부 양보**하는 것.

### 1-4. Ollama에 올라온 모델은 이미 양자화돼 있다
- `ollama run gemma4:e2b`로 받는 모델은 **기본적으로 양자화된 모델**.
- Gemma 4 기본(원본) 크기는 약 16GB(E4B 기준)인데, Ollama에 올라온 건 약 8GB — 양자화 덕분.
- Gemma 4의 **E2B / E4B**에서 E = **Effective(유효) 파라미터**. 실제 저장 파라미터는 이보다 더 많아서 "4B니까 8GB"로 딱 안 떨어진다(오차 감안).

---

## 2. 실습 환경 구성 (WSL2 + Ubuntu)

### 2-1. WSL 새 배포판 만들기 (Windows PowerShell)

```powershell
wsl --install -d Ubuntu-24.04
# 배포판 이름을 Quantization 등으로 지정, 계정/비밀번호 설정
```

- 가장 최신 버전보다 **직전 LTS(24.04)** 가 안정적이라 선호.
- 우분투는 **Debian 계열** → 패키지 설치는 `apt`. (RedHat/CentOS/Fedora 계열은 `yum`/`dnf`)
- 새로 설치하면 습관적으로 업데이트:

```bash
sudo apt update
```

### 2-2. WSL 관련 참고
- `nvidia-smi`가 WSL에서도 그대로 동작 → **Windows에 깔린 CUDA 드라이버를 WSL이 인식**한다. 리눅스용 CUDA를 따로 깔 필요 없음. (단, 리눅스 서버에서 작업한다면 별도 설치 필요)
- WSL은 Windows 폴더와 공유된다(`/mnt/c/...`). → **`/mnt/c/Users` 쪽을 함부로 지우면 Windows 파일이 날아갈 수 있음** 주의.
- WSL 여러 개 만들면 디스크가 꽉 찰 수 있으니 관리 필요.
- 리눅스에서 이름 앞에 `.`이 붙으면 **숨김 파일/폴더** (`.env`, `.venv` 등). `ls -a`로 확인.

### 2-3. 자주 쓴 리눅스 명령어

| 명령어 | 의미 |
|---|---|
| `cd`, `cd ..` | 폴더 이동 / 상위 폴더 |
| `ls -a`, `ls -l` | 숨김 포함 목록 / 상세 목록 |
| `mkdir 폴더명` | 폴더 생성 |
| `cat 파일명` | 파일 내용 화면에 출력 (보기만 함) |
| `sudo` | 관리자 권한으로 실행 |
| `apt install -y 패키지` | 패키지 설치 (`-y`: 확인 자동 Yes) |
| `\` (줄 끝) | 명령어가 다음 줄로 **이어진다**는 표시 |

> 강사 팁: 명령어는 한 번 정리해서 **노션/메모장에 매뉴얼로 저장**해두고 복붙해서 쓰는 게 실무에서도 흔한 방식. 에러가 나는 지점은 거의 항상 동일하니, 에러 지점을 리스트업해두면 다음엔 금방 한다. 에러가 뜨면 "아, 에러 떴네, 신난다" 하고 **메시지를 끝까지 읽는 습관**.

---

## 3. Ollama 설치와 기본 사용

### 3-1. 설치 (리눅스)

```bash
sudo apt install -y zstd          # 설치 스크립트가 zstd(압축 해제)를 요구함
curl -fsSL https://ollama.com/install.sh | sh
```

- `curl`로 `install.sh`를 받아서 `| sh`로 **바로 실행**한다는 뜻.
- 설치하면 서비스가 자동으로 실행됨. 상태 확인/시작:

```bash
systemctl status ollama      # active (running)이면 정상
sudo systemctl start ollama  # 꺼져 있으면 시작
```

### 3-2. 모델 실행 & 확인

```bash
ollama run gemma4:e2b   # 없으면 다운로드(약 3.5GB) 후 대화 시작
ollama ps               # 지금 "실행 중"인 모델만 표시
ollama list             # "설치된" 모델 목록
ollama --version
```

- **`ollama ps` vs `ollama list`** 구분해서 기억: 실행 중 / 설치됨. (Docker의 `docker ps` 느낌과 비슷)
- Ollama 자체는 모델이 아니라 **Gemma, Qwen 등 각종 로컬 모델을 돌리는 도구(런타임)**.
- Gemma 4는 구글 Gemini의 로컬(오픈) 버전. 학습 시점이 올해 초라 **최신 정보는 모른다** (예: 최근 인물/사건 질문에 약함).

### 3-3. 로컬 모델 활용 아이디어
- 지식 컷오프 한계 → **웹 검색 엔진 + 툴 콜링(Tool Calling)** 을 붙이면 비용 없이 에이전트 구현 가능. (실시간성은 검색으로 보완)
- 작은 모델(4B)로 먼저 구성 → 파라미터 큰 모델(예: Qwen3 30B)로 교체하면 성능 향상.
- GPU 메모리(8GB 등)를 넘더라도 **넘친 부분은 CPU로 처리**하기 때문에 돌아는 간다(느려질 뿐).
- 면접 팁(엣지 컴퓨팅 쪽): Jetson/라즈베리파이에 배터리·스피커를 붙여 **양자화한 로컬 모델이 말하는 시연**을 들고 가면 좋은 어필 포인트.

---

## 4. llama.cpp 빌드 (양자화 도구 준비)

### 4-1. 필요한 패키지

```bash
sudo apt update
sudo apt install -y build-essential cmake python3-pip git   # C++ 빌드 도구 등
```

### 4-2. 소스 받기 & 빌드

```bash
mkdir gemma4 && cd gemma4        # 작업 폴더 (이번 수업에서는 gemma4 폴더 사용)
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp

cmake -B build -DCMAKE_BUILD_TYPE=Release   # 빌드 환경 설정 (대소문자 주의!)
cmake --build build --config Release -j     # 실제 빌드 (-j: 병렬)

./build/bin/llama-quantize --help           # 도움말이 나오면 성공 → 이제 양자화만 하면 됨
```

- `llama.cpp` 이름대로 **C++ 프로젝트**이고 GitHub에서 `git clone`으로 가져옴.
- 빌드 중 경고/로그가 주르륵 나와도 정상. 오타만 조심 (`-D`, 대소문자).
- 강사 설명: C++ 코드 자체까지 파고들 필요는 없고, 양자화 도구로 쓰기 위한 환경 구축 단계. 나중에 시스템/NPU 쪽으로 가면 C++·행렬 연산 이해가 필요해짐.

---

## 5. Python 가상환경 + 변환 스크립트 준비

### 5-1. venv (파이썬 기본 가상환경)

```bash
sudo apt install -y python3-venv
python3 -m venv .venv            # .venv 폴더 생성 (앞의 . = 숨김)
source .venv/bin/activate        # 활성화 → 프롬프트 앞에 (.venv) 표시
```

- conda 대신 쓰는 **파이썬 기본 내장 가상환경**. 회사에서 `.venv`를 쓰는 곳도 많음 → 놀랄 필요 없음.
- (Miniconda를 설치하지 않았기 때문에 `conda` 명령은 아직 안 먹음)

### 5-2. 패키지 설치

```bash
python -m pip install --upgrade pip
pip install torch
pip install -r requirements.txt     # llama.cpp 폴더의 requirements 목록을 한 번에 설치
```

- `requirements.txt` = 설치할 패키지 목록 파일 → `pip install -r`로 **한 방에 설치**. 회사에서도 "이 목록대로 설치하세요" 하면 이렇게 하면 됨.
- `cat requirements...` 로 내용을 먼저 확인해보는 습관 (numpy, torch 2.x, transformers, gguf 등이 포함).

### 5-3. 에러 → 원인 → 해결 흐름 (오늘 실제로 겪은 것)

| 순서 | 에러 | 원인 | 해결 |
|---|---|---|---|
| 1 | 파일을 찾을 수 없음 | 모델(E2B/E4B) 경로·이름이 다름 | 폴더/파일명 확인 |
| 2 | `torch` 가 없다 | PyTorch 미설치 | `pip install torch` |
| 3 | `... object has no attribute ...special tokens` (토크나이저 호환 에러) | transformers 계열 패키지 미설치/버전 불일치 | `pip install transformers tokenizers sentencepiece` |
| 4 | HuggingFace 다운로드 중 인증/토큰 요구, `Chunked Encoding Error` | 로그인(토큰) 필요 + 24명이 동시에 다운로드해서 네트워크 병목 | 직접 다운로드한 파일을 공유(외장하드/USB/공유 드라이브)로 복사 |

> `pip check`로 설치된 패키지끼리 충돌이 없는지 확인할 수 있음 ("No broken requirements found").
> 토크나이저·트랜스포머는 언어모델 하려면 **반드시 알아야 하는 핵심** (20일 트랜스포머 수업 예고).

---

## 6. 변환 → 양자화 → Ollama 등록 (핵심 파이프라인)

```
HuggingFace 원본(BF16)  →  GGUF(BF16)  →  GGUF(Q4_K_M)  →  Ollama 등록  →  실행
        (다운로드)        (convert_hf_to_gguf.py)   (llama-quantize)   (ollama create)   (ollama run)
```

### 6-1. HF → GGUF 변환
- `convert_hf_to_gguf.py` = Hugging Face 포맷 모델을 **GGUF 파일 형식**으로 바꾸는 스크립트.
- 이번엔 허깅페이스 다운로드가 막혀서 **미리 받아둔 BF16 GGUF 파일을 공유받아** 사용. (수업에서 실제로 필요한 건 이 `.gguf` 파일 하나)
- GGUF: llama.cpp/Ollama가 쓰는 모델 파일 포맷 (`.gguf` 확장자). 양자화 전/후 모두 확장자는 `.gguf`.
- 이번에 받은 건 **E2B의 BF16** (약 9.3GB, 용량이 같은지 `ls -lh`로 확인). 

### 6-2. 양자화 실행 (이번 수업의 핵심 명령)

```bash
./build/bin/llama-quantize  ../gemma4-e2b-bf16.gguf  ../gemma4-e2b-Q4_K_M.gguf  Q4_K_M
#                           [입력: 원본 BF16]          [출력: 양자화 결과]         [양자화 방식]
```

- **경로가 가장 많이 틀린다** → `llama.cpp` 폴더에서 실행하는지, 입력/출력 파일 경로가 맞는지 확인. 파일명 대소문자도 구분.
- 결과: 용량이 **원본 약 8.8GB → 약 3.2GB** 수준으로 감소. (녹취상 수치가 불명확, 확인 필요) 양자화 자체는 오래 안 걸림.
- 앞에서 한 설치/빌드는 전부 "밑밥"이고, **양자화 작업은 이 명령 하나**.
- `Q4_K_M` 읽는 법: **Q4 = 4비트 양자화**, K = K-quant 방식, M = Medium (참고: llama.cpp 표기 관례).

### 6-3. Ollama에 등록하기

1) **Modelfile** 만들기 — 안에는 사실상 경로 한 줄:

```
FROM ./gemma4-e2b-Q4_K_M.gguf
```

```bash
echo "FROM ./gemma4-e2b-Q4_K_M.gguf" > Modelfile
ls -l            # Modelfile 생성 확인
```

2) 등록 & 실행:

```bash
ollama create gemma4-e2b-q4km -f Modelfile   # 이름은 자유, 소문자 권장
ollama list                                  # 등록된 모델 확인 (내 것 + 기존 것)
ollama run gemma4-e2b-q4km                   # 내가 양자화한 모델과 대화!
```

- 에러가 났다면 대부분 **모델 이름(대소문자)·경로 오타**. (강사: "Q4_K_M은 대문자로")

### 6-4. 결과 비교 & 해석
- Ollama 기본 모델(약 4.6GB) vs 내가 만든 모델(약 4.4GB, 녹취상 수치 확인 필요) — 크기가 비슷하거나 약간 작게 나옴.
- **"우리가 잘해서 작은 게 아니다."** Ollama 공식 모델은 **비전(이미지) 기능까지 포함**된 멀티모달이고, 이번에 직접 변환한 건 **텍스트(LLM) 부분만** 가져와서 작아진 것.
- 실행해보면 내가 만든 모델도 일반 모델과 똑같이 대화된다(말을 걸어 확인).
- 더 작은 장비에 넣어야 하면 → 더 작은 모델(E2B) / 더 낮은 비트 / 텍스트 전용 모델 선택.

---

## 7. 과제 & 실습 지시 (수업 운영)

- **오늘 저녁까지 양자화 테스트 완료**, 집에서도 처음부터 끝까지 한 번 직접 해볼 것 (명령어 매뉴얼 만들어서).
- 추가 실습: **Qwen3 4B** 모델 다운로드 → 같은 방식으로 양자화 (8B는 시간이 너무 오래 걸려 4B로 변경). 다음 시간엔 **Gemma E4B** 양자화 예정.
- 안드로이드/윈도우용 Ollama도 존재 (폰에서 Gemma 쓰는 사람도 많음).
- **논문 스터디**: 5인 1팀, 팀 업무 분담, 발표 시 PPT의 아키텍처뿐 아니라 **논문 속 아키텍처·코드 단위까지** 디테일하게 질문 예정. 발표 자세도 이전 지적 사항 확인.
  - 논문 구현 과제(Mask R-CNN): 전체를 다 구현할 필요는 없고 **마지막 Mask R-CNN만** HuggingFace 등의 사전 학습 가중치를 가져다 돌려보기(가중치 사용 허용).
  - 발표 흐름(스토리): **R-CNN → Fast R-CNN(문제점 개선) → Faster R-CNN → Mask R-CNN(세그멘테이션 추가)**, 각 단계의 문제점과 개선점을 약 10분간 설명. **Loss function(원)** 은 완벽히 이해할 것.
  - 비전/LLM 구분 없이 아키텍처는 서로 섞여 쓰임 (예: Vision Transformer).
- 20일: 트랜스포머 수업·발표 예정.

---

## 8. 특별 탭 — 취업/AI 업계 이야기 (참고)

- 채용공고를 같이 훑어봄: "AI 신입"으로 검색하면 수천 건, "AI" 전체는 만 단위. **일자리는 많은데 실력 있는 지원자가 부족**하다는 것이 회사 쪽 이야기.
- 공고에서 자주 보이는 키워드: LLM/Agent 개발, Python, VectorDB, Pandas/NumPy, 데이터 분석 등 — "사용 경험"만 적어도 일단 지원해볼 만하다.
- 회사가 면접에서 보는 것: **트랜스포머 아키텍처 설명, 실제 코드(PyTorch) 구현**. 열심히 하겠다는 말보다 설명·구현 능력.
- 학력/경력 조건은 "참고" 수준 → **지원 조건에 못 미쳐도 100~200군데 지원**하면 3~4곳 연락이 온다는 현실적인 조언.
- 첫 직장 선택 기준: **배울 수 있는 선임이 있는지** / 개발 환경, 그리고 어디든 **1년 버티면 선택지가 크게 늘어난다.**
- 면접 대비: 정장 준비.

---

## 9. 오늘 배운 것 체크리스트

- [ ] 모델 용량 = 파라미터 수 × 바이트 수 (BF16=2B)로 계산할 수 있다
- [ ] BF16/FP16 표기 = 양자화 안 된 원본임을 안다 / 재양자화는 지양한다
- [ ] WSL Ubuntu 설치, `apt`, `curl | sh` 패턴 이해
- [ ] Ollama: 설치 → `run` / `ps` / `list` / `create` 구분
- [ ] llama.cpp: `git clone` → `cmake` 빌드 → `llama-quantize`
- [ ] venv 가상환경 + `requirements.txt`로 한 번에 설치
- [ ] GGUF 변환(`convert_hf_to_gguf.py`) → Q4_K_M 양자화 → Modelfile → `ollama create` 흐름 설명 가능
- [ ] 에러 메시지를 끝까지 읽고 (torch 없음 / transformers·tokenizers 필요) 원인을 찾는 습관
