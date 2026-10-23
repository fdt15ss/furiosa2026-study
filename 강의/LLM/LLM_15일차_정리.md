# LLM 15일차 정리

> 녹취 기반 정리. 음성 인식이 불안정한 구간이 많아, 불확실한 부분은 `(확인 필요)`로 표시했습니다.

## 오늘의 흐름

1. Windows에 **WSL(리눅스 가상 PC)** 설치
2. 리눅스에서 **Hermes Agent** 설치 및 초기 설정
3. **게이트웨이 / 대시보드** 확인, 모델 변경(커스텀 엔드포인트)
4. Hermes에게 **작업 규칙 지시**하기 (프로젝트 활용)

---

## 1. WSL로 리눅스 환경 만들기

### 개념
- 지금까지는 **Windows PowerShell** 상태. 여기에 **WSL(Windows Subsystem for Linux)** 로 리눅스 PC를 하나 더 만든다.
- VMware / VirtualBox로 가상 PC를 만든 것처럼, **내 PC 안에 새로운 PC가 생긴다**고 이해하면 된다.
- PyTorch 등 딥러닝 실습도 이후 **WSL 환경**에서 진행할 예정.

### 주요 명령어 (PowerShell)

| 목적 | 명령어 |
|---|---|
| 설치된 리눅스 목록·버전 확인 | `wsl -l -v` |
| 온라인에서 설치 가능한 배포판 목록 | `wsl -l -o` (= `wsl --list --online`) |
| 배포판 설치 (이름 지정) | `wsl --install -d Ubuntu-24.04 --name <이름>` |
| WSL 상태 확인 | `wsl --status` (확인 필요) |
| WSL 전체 종료 | `wsl --shutdown` |
| 배포판 삭제(등록 해제) | `wsl --unregister <배포판 이름>` |

- `-d` 는 **distribution(배포판)** 의 약자.
- 설치 가능한 목록에는 Ubuntu(24.04, 22.04 등), openSUSE, Debian, AlmaLinux 등이 나온다. 강사 설명으로는 최신 Ubuntu 26.04도 나왔지만 **수업에서는 안정적인 24.04 사용**.
- **같은 버전을 여러 개 설치 가능**: `--name` 으로 이름만 바꿔주면 24.04를 2개, 3개… 계속 만들 수 있다. 이름을 지정하지 않으면 기본 이름(`Ubuntu-24.04`)으로 설치되며, 이미 있으면 충돌한다.
- 설치가 끝나면 **계정(사용자 이름) / 비밀번호**를 입력한다. (수업에서는 이름·비밀번호를 통일해서 설정)
- 설치 중 "다시 시작하라"는 메시지가 나오면 재부팅이 필요할 수 있다. 목록(`wsl -l -v`)에 리눅스가 보이면 일단 성공.

### 설치가 안 될 때 (트러블슈팅)
1. `wsl --shutdown` 으로 종료
2. `wsl --unregister Ubuntu-24.04` 로 기존 등록 해제
3. `wsl -l -v` 로 상태 확인 후, 이름을 바꿔 다시 설치
- 계정 생성 단계에서 멈추는 경우가 많았음 → 처음부터 다시 만들어 해결.
- 일부는 회사/집 PC 원격 접속 환경(IP 다름) 때문에 네트워크 이슈가 있었음.

### 리눅스 기본 명령어

| 명령어 | 설명 |
|---|---|
| `clear` | 화면 정리 |
| `cd <경로>` | 디렉터리 이동 (**C**hange **D**irectory) |
| `ls` | 폴더 내용 확인 |
| `explorer.exe .` | 현재 리눅스 폴더를 Windows 탐색기로 열기 (확인 필요) |

- **`/mnt/c`** = Windows의 **C 드라이브**. 예: `/mnt/c/Windows/System32` → Windows 폴더 구조를 리눅스에서 그대로 접근 가능.
- 리눅스는 **대소문자를 구분**한다. (`Windows` ≠ `windows`)
- 터미널에서 **붙여넣기는 `Ctrl+V` 대신 우클릭**하는 경우가 많다 (수업 환경 기준).
- 프롬프트 모양이 `PS C:\...` 이면 Windows PowerShell, `사용자@PC:~$` 이면 리눅스.

---

## 2. Hermes Agent 설치

- **Hermes Agent**: 프로젝트에 "직원을 하나 더 붙여주는" AI 에이전트. 강사 설명으로는 OpenClaw 계열의 기능을 대부분 가져온 것.
- 공식 홈페이지의 **Install Hermes** 에서 설치 명령어(`curl ... | bash` 형태)를 복사해 **리눅스(WSL) 터미널**에 붙여넣어 실행.
- 실행 중 `sudo` 관리자 권한 **비밀번호**를 물어보면 리눅스 계정 비밀번호 입력.
- 설치 후 초기 설정(setup) 화면에서 선택:
  - 모드: **Quick setup / Local** 중 선택 (Quick setup은 로그인 필요)
  - 로그인: 홈페이지(포털)에서 **구글 계정 등으로 로그인** → 인증 필요
  - 처음 모델: **Upstage Solar Mini (무료)** 로 통일해서 진행
  - 메신저 연동: **Telegram** 항목이 나옴 → 이번 시간은 건너뛰고 **오후에 별도로 연결**
  - 터미널 백엔드: **Local**
  - "WSL이 켜질 때마다 Hermes 게이트웨이 자동 실행" 옵션: **Yes**
  - 문제 보고(전송) 옵션: **No**

### 자주 쓰는 Hermes 명령어

| 명령어 | 설명 |
|---|---|
| `hermes` | 터미널에서 Hermes와 대화 시작 (종료: `Ctrl+C`) |
| `hermes gateway status` | 게이트웨이 동작 확인 (**Active: running** 이 파란색이면 정상) |
| `hermes gateway start` | 게이트웨이 시작 |
| `hermes gateway stop` | 게이트웨이 중지 |
| `hermes gateway install` | 게이트웨이 재설치 (설치 중 오류로 빠졌을 때) |
| `hermes dashboard` | 웹 대시보드 실행 → 출력된 주소를 브라우저에 입력 → **Chat** 메뉴에서 대화 |
| `hermes model` | 사용 모델/프로바이더 변경 |

**문제 해결 순서**: 뭔가 이상하면 `status` → 안 돌면 `stop` → `start` → 다시 `status`로 확인.

---

## 3. 모델 변경: 커스텀 엔드포인트 (OpenAI 호환 API)

`hermes model` 실행 후 순서:

1. 목록 아래쪽의 **Custom endpoint** 선택 (Custom direct API 아님)
2. **API Base URL** 입력 (수업용 스터디 사이트의 Base URL 복사)
3. **API Key** 입력
4. API 모드 선택: 자동 / **Chat Completions** / Responses 등 중 **Chat Completions(2번)** 선택
5. 제공되는 **모델 목록**이 나오면 수업에서 정한 번호의 모델 선택 (수업에서는 전원 7번, 고성능 모델은 비용이 빨리 소진되니 주의)
6. **토큰 길이 제한**: 제한 없이 Enter (제한하면 응답이 끊길 수 있음)
7. **Display name** 지정 (예: 수업용 이름)
8. 다시 `hermes dashboard` → Chat에서 모델이 바뀌었는지 확인

> 이 방식은 OpenAI 호환 엔드포인트라서, **LangChain에서 `base_url` + `api_key` 로 ChatOpenAI를 쓰는 구조와 같은 개념**이다.

---

## 4. 에이전트에게 작업 규칙 지시하기 (활용 팁)

강사가 시연한 지시 방식:

- 리포트를 수집해 **PDF로 만들어 지정한 폴더(예: `C:\...\hermes` 아래 report 폴더)** 에 저장
- **모든 작업 전에 예상 소요 시간을 먼저 말하도록** 규칙 지정
- 이 규칙을 **어디에 저장해서 앞으로도 지키게 할지**(메모리/규칙 파일)까지 함께 지시해야 함
- **폴더 범위를 제한**해야 함 (안 그러면 시스템 폴더까지 건드릴 위험)
- 말을 안 들으면 **다시 지시**하면 된다
- PDF 생성 라이브러리가 없으면 에이전트가 **"설치해도 되나요?"** 라고 묻고, 허락하면 알아서 설치하고 진행 → 예전처럼 직접 라이브러리를 설치할 필요가 줄어듦

**프로젝트 활용 아이디어**
- 해커톤/공모전 자료를 `input` 폴더에 넣고 "분석해서 예상 시간·필요 자료를 알려줘 → 필요한 자료는 네가 수집해 → 내가 줘야 할 자료는 물어봐" 식으로 반복 지시
- **Telegram 연동 후**: 이동 중에도 "이것도 추가해서 집에 가기 전까지 만들어놔" 식으로 원격 지시
- 메일 / 구글 드라이브 / 구글 캘린더 연동으로 자료 정리·일정 등록까지 확장 가능
- 팀원 역할을 일부 대신해 줄 수 있으니, 본인이 만든 에이전트와 **성능 비교**해 보라는 과제 아이디어

> **다음 시간 예고**: Telegram 연결 (오후)

---

## 오늘의 핵심 요약

- WSL = Windows 안에서 쓰는 리눅스. `wsl -l -v`, `wsl -l -o`, `wsl --install -d Ubuntu-24.04 --name ...`
- `/mnt/c` 로 Windows C 드라이브 접근, 리눅스는 대소문자 구분
- Hermes Agent: 설치(curl) → setup → `hermes gateway status`로 정상 확인 → `hermes dashboard` 로 웹 채팅
- 모델 변경은 `hermes model` → **Custom endpoint** → Base URL / API Key / Chat Completions
- 에이전트에게는 **작업 폴더 제한, 예상 시간 보고, 규칙 저장**을 지시

## 복습 체크리스트

- [ ] WSL에 Ubuntu 24.04 설치 후 `wsl -l -v` 로 확인
- [ ] `hermes gateway status` 가 Active(running)인지 확인
- [ ] `hermes dashboard` 로 웹 채팅 접속
- [ ] `hermes model` 로 커스텀 엔드포인트 설정 연습
- [ ] 오후: Telegram 연결 준비
