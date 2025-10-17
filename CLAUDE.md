# CLAUDE.md

**⚠️ 중요 사항:**
1. **이 파일의 목적**: 릴리즈 직전 코드 정리 및 문서화를 위한 개발자 참고 자료
2. **Git 커밋 시 주의**: 이 파일을 git push 할 때는 "Claude" 관련 내용을 제거하고 일반 개발 문서로 변환할 것
   - 파일명을 `DEVELOPMENT.md` 또는 `ARCHITECTURE.md`로 변경 권장
   - Claude Code 관련 언급은 모두 삭제

---

## 프로젝트 개요

LDM-EKI는 대기 확산 모델링과 앙상블 칼만 역산을 결합한 방사능 오염원 추정 시스템입니다:
- **CUDA/C++ 순방향 모델**: GPU 가속 입자 확산 시뮬레이션
- **Python 역산 모델**: 앙상블 칼만 기법을 이용한 오염원 최적화
- **IPC 통신**: POSIX 공유 메모리를 통한 고성능 프로세스 간 데이터 교환

## 빌드 및 실행

### 프로젝트 빌드

```bash
# 메인 EKI 실행파일 빌드 (기본 타겟)
make

# 모든 타겟 빌드 (ldm, ldm-eki, ldm-receptor-debug)
make all-targets

# 빌드 산출물 정리
make clean
```

**필수 요구사항:**
- CUDA 툴킷 (nvcc) 경로: `/usr/local/cuda/bin/nvcc`
- GPU 컴퓨팅 성능: SM 6.1 이상
- OpenMP 지원 (병렬 빌드용)

**빌드 최적화:**
- 자동 병렬 빌드 활성화 (CPU 코어 수만큼)
- 최적화 레벨: `-O2` (빠른 컴파일, 충분한 성능)
- 빌드 시간: ~30초-1분 (시스템에 따라 다름)

### 데이터 정리

```bash
# 이전 실행 데이터 정리 (수동 실행 시)
python3 util/cleanup.py

# 옵션:
# --dry-run         실제 삭제 없이 미리보기
# --no-confirm      확인 없이 즉시 삭제
# --logs-only       logs만 정리
# --output-only     output만 정리
# --shm-only        공유 메모리만 정리
```

**참고:** `./ldm-eki` 실행 시 자동으로 cleanup.py가 호출되어 이전 데이터를 정리합니다.

### 시뮬레이션 실행

```bash
# EKI 최적화 실행
./ldm-eki

# 실행 시 자동으로:
# 1. 이전 데이터 정리 (확인 프롬프트 표시)
# 2. 시뮬레이션 수행
# 3. 결과 시각화 자동 생성

# 출력 파일 위치:
# - logs/ldm_eki_simulation.log - 메인 시뮬레이션 로그
# - logs/python_eki_output.log - Python EKI 프로세스 로그
# - output/plot_vtk_prior/ - 사전 시뮬레이션 VTK 파일
# - output/plot_vtk_ens/ - 앙상블 시뮬레이션 VTK 파일
# - output/results/all_receptors_comparison.png - 시각화 결과 (자동 생성)
```

## 시스템 아키텍처

### 2-프로세스 설계

시스템은 **두 개의 협력 프로세스**로 동작합니다:

1. **LDM (C++/CUDA)**: `ldm-eki` 실행파일
   - 순방향 입자 확산 시뮬레이션 수행
   - GPU에서 CUDA 커널 실행
   - 관측값을 공유 메모리에 기록 (`/dev/shm/ldm_eki_*`)
   - 공유 메모리에서 앙상블 상태 읽기
   - 빠른 반복을 위한 기상 데이터 사전 로딩

2. **EKI (Python)**: `src/eki/RunEstimator.py`
   - LDM에 의해 백그라운드에서 자동 실행
   - 공유 메모리에서 관측값 읽기
   - 앙상블 칼만 역산 수행
   - 업데이트된 앙상블 상태를 공유 메모리에 기록
   - 수렴 또는 최대 반복 횟수까지 반복

### IPC 통신 흐름

```
[초기 실행]
LDM (C++) → 관측값 → 공유 메모리 (/dev/shm/ldm_eki_data)
                         ↓
Python EKI ← 관측값 ← 공유 메모리
Python EKI가 사전 앙상블 생성

[반복 루프 - N회 반복]
Python EKI → 앙상블 상태 → 공유 메모리 (/dev/shm/ldm_eki_ensemble_*)
                              ↓
LDM (C++) ← 앙상블 상태 ← 공유 메모리
LDM이 N개의 앙상블 시뮬레이션 실행 (각 멤버당 1개)
LDM (C++) → 앙상블 관측값 → 공유 메모리
                              ↓
Python EKI ← 앙상블 관측값 ← 공유 메모리
Python EKI가 칼만 이득을 사용해 앙상블 업데이트
[루프 계속...]
```

### 주요 IPC 모듈

**C++ 측:**
- `src/ipc/ldm_eki_writer.cuh`: IPC writer 클래스
  - `EKIWriter::writeObservations()` - 초기 관측값 기록
  - `EKIWriter::writeEnsembleObservations()` - 앙상블 관측값 기록
- `src/ipc/ldm_eki_reader.cuh`: IPC reader 클래스
  - `EKIReader::waitForEnsembleData()` - Python 상태 대기
  - `EKIReader::readEnsembleStates()` - 앙상블 상태 읽기

**Python 측:**
- `src/eki/eki_ipc_reader.py`: C++로부터 관측값 읽기
  - `receive_gamma_dose_matrix_shm()` - 초기 관측값
  - `receive_ensemble_observations_shm()` - 앙상블 관측값
- `src/eki/eki_ipc_writer.py`: C++로 앙상블 상태 쓰기
  - `write_ensemble_to_shm()` - 앙상블 상태 전송

### 설정 시스템

**LDM 설정** (`input/setting.txt`):
- 시뮬레이션 파라미터: time_end, dt, 입자 수
- 물리 모델: 난류, 침적, 붕괴
- 파일 경로 및 그리드 차원

**EKI 설정** (`input/eki_settings.txt`):
- 수용체 위치 및 포착 반경
- 참값/사전 방출량 시계열
- EKI 알고리즘 파라미터 (앙상블 크기, 반복 횟수, adaptive/localized 옵션)
- GPU 설정

**공유 메모리 설정:**
- 두 프로세스 모두 `input/eki_settings.txt` 읽기
- C++가 전체 설정을 `/dev/shm/ldm_eki_full_config`에 기록 (128 바이트)
- Python이 `Model_Connection_np_Ensemble.py::load_config_from_shared_memory()`를 통해 설정 읽기

### 핵종 시스템

모델은 CRAM (Chebyshev Rational Approximation Method)을 사용한 방사성 붕괴 체인을 지원합니다:

- 핵종 정의: `input/nuclides_config_1.txt` (또는 60-핵종 체인용 `nuclides_config_60.txt`)
- CRAM 행렬: `cram/A60.csv`
- 붕괴 체인 처리: `src/physics/ldm_nuclides.cuh`

## EKI 최적화 알고리즘

Python EKI 구현 (`src/eki/Optimizer_EKI_np.py`)은 다음 알고리즘들을 지원합니다:

- **EnKF**: 표준 앙상블 칼만 필터
- **Adaptive_EnKF**: 적응형 스텝 크기 조절
- **EnKF_with_Localizer**: 공분산 국소화 (거짓 상관관계 제거)
- **EnRML**: 앙상블 랜덤화 최대우도법
- **EnKF_MDA**: 다중 데이터 동화
- **REnKF**: 제약조건을 가진 정규화 EnKF

`input/eki_settings.txt`에서 제어:
```
EKI_ADAPTIVE=On/Off
EKI_LOCALIZED=On/Off
EKI_REGULARIZATION=On/Off
```

## 중요 구현 세부사항

### 기상 데이터 사전 로딩

앙상블 모드에서는 반복 전에 **모든 기상 데이터를 사전 로딩**합니다:
- 함수: `LDM::preloadAllEKIMeteorologicalData()` (`ldm.cuh`)
- 반복 중 파일 I/O를 피하기 위해 모든 타임스텝을 병렬로 로드
- `eki_meteo_cache` 멤버 변수에 저장
- 성능에 필수적: 빠른 앙상블 반복 가능

### 입자 초기화

**단일 모드** (초기 참값 시뮬레이션):
- `LDM::initializeParticlesEKI()`: 설정 파일의 `true_emissions` 사용

**앙상블 모드** (각 반복마다):
- `LDM::initializeParticlesEKI_AllEnsembles()`: 모든 앙상블 멤버용 입자 생성
- 각 앙상블은 고유한 `ensemble_id`를 가진 입자 세트 보유
- 입자 데이터 구조에 `ensemble_id`와 `timeidx` 필드 포함

### VTK 출력 제어

VTK 출력은 비용이 크므로 `ldm.enable_vtk_output`로 제어:
- 초기 참값 시뮬레이션(단일 모드)에서 활성화
- 중간 반복 중에는 **비활성화** (성능 최적화)
- 최종 반복에서만 **활성화**
- 선택된 앙상블 멤버(예: 앙상블 7)가 `output/plot_vtk_ens/`에 출력

### 관측 시스템

관측값은 **수용체 위치**에서 수집됩니다 (그리드 아님):
- 수용체는 `eki_settings.txt`에서 위도/경도로 정의
- 포착 반경: `RECEPTOR_CAPTURE_RADIUS` (도 단위)
- GPU 배열: `d_eki_receptor_observations`가 각 수용체의 타임스텝별 선량 저장
- 형태: `[num_ensemble][num_timesteps][num_receptors]`

### 데이터 재배열 규칙

**중요**: Python과 C++는 서로 다른 배열 레이아웃 사용:

**Python (NumPy)**: 상태에 대해 열-우선(Column-major)
- 앙상블 상태: `(num_states, num_ensemble)`
- 관측값: `(num_receptors, num_timesteps)`

**C++**: 행-우선(Row-major)
- 앙상블 상태: `[ensemble][state]`
- 관측값: `[ensemble][timestep][receptor]`

공유 메모리에 쓸 때는 **항상 행-우선 순서로 평탄화**해야 C++에서 올바르게 읽을 수 있습니다.

### Memory Doctor 모드

IPC 통신 문제 디버깅용:
```
MEMORY_DOCTOR_MODE=On
```
C++와 Python 간 모든 데이터 전송을 `/tmp/eki_debug/`에 로깅하여 비교 가능.

## 일반적인 개발 패턴

### 새로운 EKI 알고리즘 추가

1. `src/eki/Optimizer_EKI_np.py`의 `Inverse` 클래스에 메서드 추가
2. `input/eki_settings.txt`에 설정 옵션 추가
3. `Model_Connection_np_Ensemble.py`의 `load_config_from_shared_memory()` 업데이트
4. `Optimizer_EKI_np.py::Run()`에 새 메서드 호출 케이스 추가

### 관측 수집 방식 수정

1. C++ 측: `src/ipc/ldm_eki_writer.cuh` 및 `ldm_eki_reader.cuh` 수정
2. Python 측: `src/eki/eki_ipc_reader.py` 수정
3. writer/reader 간 데이터 형식 일치 확인
4. 필요시 공유 메모리 버퍼 크기 업데이트

### 물리 모델 추가

1. `src/kernels/ldm_kernels.cuh`에 CUDA 커널 추가
2. `src/simulation/ldm_func_simulation.cuh`에서 커널 호출하도록 업데이트
3. `input/setting.txt`에 설정 추가
4. `src/init/ldm_init_config.cuh`에서 설정 파싱

## 파일 구조 (모듈화된 구조)

```
src/
├── main_eki.cu              - EKI 실행파일 진입점
├── main.cu                  - 표준 시뮬레이션 진입점
├── main_receptor_debug.cu   - 그리드 수용체 디버그 도구
├── colors.h                 - 범용 ANSI 색상 정의
├── core/                    - 핵심 클래스
│   ├── ldm.cuh             - 메인 LDM 클래스 정의
│   └── ldm.cu              - LDM 클래스 구현
├── data/
│   ├── config/             - 설정 구조체
│   │   ├── ldm_config.cuh  - 설정 파일 파서
│   │   └── ldm_struct.cuh  - 데이터 구조체 정의
│   └── meteo/              - 기상 데이터 관리
│       ├── ldm_mdata_loading.cuh/cu
│       ├── ldm_mdata_processing.cuh/cu
│       └── ldm_mdata_cache.cuh/cu
├── physics/                 - 물리 모델
│   ├── ldm_cram2.cuh/cu    - CRAM48 방사성 붕괴
│   └── ldm_nuclides.cuh/cu - 핵종 체인 관리
├── kernels/                 - CUDA 커널
│   ├── ldm_kernels.cuh     - 커널 메인 헤더
│   ├── device/             - 디바이스 함수
│   ├── particle/           - 입자 업데이트 커널
│   ├── eki/                - EKI 관측 커널
│   └── dump/               - 그리드 덤프 커널
├── ipc/                     - 프로세스 간 통신
│   ├── ldm_eki_writer.cuh/cu
│   └── ldm_eki_reader.cuh/cu
├── simulation/              - 시뮬레이션 함수
│   ├── ldm_func_simulation.cuh/cu
│   ├── ldm_func_particle.cuh/cu
│   └── ldm_func_output.cuh/cu
├── visualization/           - VTK 출력
│   ├── ldm_plot_vtk.cuh/cu
│   └── ldm_plot_utils.cuh/cu
├── init/                    - 초기화
│   ├── ldm_init_particles.cuh/cu
│   └── ldm_init_config.cuh/cu
├── debug/                   - 디버깅 도구
│   ├── memory_doctor.cuh/cu
│   └── kernel_error_collector.cuh/cu
└── eki/                     - Python EKI 프레임워크
    ├── RunEstimator.py      - 메인 EKI 실행기
    ├── Optimizer_EKI_np.py  - 칼만 역산 알고리즘
    ├── Model_Connection_np_Ensemble.py - 순방향 모델 인터페이스
    ├── eki_ipc_reader.py    - C++로부터 읽기
    └── eki_ipc_writer.py    - C++로 쓰기

util/                        - 유틸리티 스크립트
├── cleanup.py               - 데이터 정리 스크립트
├── compare_all_receptors.py - 결과 시각화 (자동 실행)
├── compare_logs.py          - 로그 비교 도구
└── diagnose_convergence_issue.py - 수렴 진단 도구

input/                       - 입력 설정 파일 (data/ 폴더 제거됨)
├── setting.txt              - LDM 시뮬레이션 설정
├── eki_settings.txt         - EKI 알고리즘 설정
├── nuclides_config_*.txt    - 핵종 정의
└── gfsdata/                 - 기상 데이터 (GFS 형식)

output/
├── plot_vtk_prior/          - 참값 시뮬레이션 VTK 파일
├── plot_vtk_ens/            - 앙상블 실행 VTK 파일
└── results/                 - 그래프 및 분석 출력
```

## 디버깅 팁

**공유 메모리 문제:**
```bash
# 공유 메모리 파일 목록
ls -lh /dev/shm/ldm_eki*

# 필요시 수동 정리
rm -f /dev/shm/ldm_eki_*
```

**프로세스 통신 확인:**
```bash
# Python 프로세스 모니터링
ps aux | grep RunEstimator

# 로그 확인
tail -f logs/ldm_eki_simulation.log
tail -f logs/python_eki_output.log
```

**GPU 사용 확인:**
```bash
nvidia-smi
```

**Memory Doctor 진단:**
`MEMORY_DOCTOR_MODE=On` 활성화 후 `/tmp/eki_debug/`에서 상세한 데이터 전송 로그 확인.

**Kernel Error Collector:**
CUDA 커널 에러를 자동으로 수집하여 시뮬레이션 종료 시 일괄 보고합니다:
- 에러 자동 수집: 시뮬레이션 중 발생한 모든 커널 에러를 메모리에 저장
- 중복 제거: 동일한 위치의 동일한 에러는 카운트만 증가
- 일괄 보고: 시뮬레이션 종료 시 에러 요약을 빨간색/굵게 출력
- 로그 저장: `logs/error/kernel_errors_YYYY-MM-DD_HH-MM-SS.log`에 타임스탬프 로그 생성
- 상세 문서: `docs/KERNEL_ERROR_COLLECTOR.md` 참조

**참고:** 이 시스템은 **비동기 커널 에러**만 수집합니다 (`cudaGetLastError()`). 동기 CUDA API 에러(예: `cudaMemcpyToSymbol` 실패)는 기존 `fprintf` 핸들러로 즉시 출력됩니다.

## 최근 변경사항 (2025)

### 코드 정리 및 최적화
- MPI 제거: 단일 프로세스 모드로 단순화
- 빌드 최적화: `-O3` → `-O2`, 자동 병렬 빌드, 시간 ~2-3분 → ~30초
- 유틸리티 스크립트 `util/` 폴더로 이동, 자동 정리 기능 추가

### 출력 및 국제화 (2025-01-15)
- 모든 출력 메시지 영어 변환 (한국어 → 영어)
- ANSI 색상 코딩 시스템 도입 (에러/성공/경고/헤더)
- 앙상블 관측 로깅 개선: 평균값만 출력하도록 최적화
- 시각화 시간 축 정렬 및 다중 수용체 지원

### 병렬 리팩토링 및 모듈화 (2025-10-15)
- 6개 에이전트로 코드베이스 분할하여 동시 리팩토링 완료
- 23개 모듈화된 파일로 재구성 (`src/simulation/`, `src/data/meteo/`, `src/init/`, `src/visualization/`, `src/ipc/`, `src/physics/`, `src/kernels/`)
- `src/include/` 폴더 제거, 각 모듈에 헤더 배치
- **상세 보고서**: `PARALLEL_REFACTORING_MASTER.md` 참조

### 터미널 출력 및 로그 시스템 개선 (2025-10-16)
- ORANGE 색상 추가, 체크마크 사용 최적화
- ColorStripStreambuf 클래스로 로그 파일에서 ANSI 코드 자동 제거
- 로그 전용 디버그 스트림 (`logonly`) 구현: 터미널은 깔끔, 로그는 상세
- 전역 로그 파일 포인터로 크로스 컴파일 유닛 로깅 지원

### CRAM T Matrix 및 Flex Height 리팩토링 (2025-10-16)
- `__constant__`/`__device__` 메모리 → 일반 GPU 메모리로 변경
- Non-RDC 컴파일 모드와 완전 호환
- KernelScalars를 통한 포인터 전달 패턴 확립
- "invalid device symbol" 및 "illegal memory access" 에러 완전 제거

### Kernel Error Collection System (2025-10-16)
- 커널 에러를 메모리에 수집하여 시뮬레이션 종료 시 일괄 보고
- CHECK_KERNEL_ERROR() 매크로로 20+ 곳에 자동 체크
- 타임스탬프 로그 파일 생성 (`logs/error/kernel_errors_*.log`)
- **상세 문서**: `docs/KERNEL_ERROR_COLLECTOR.md` 참조

### Input File Modernization (2025-10-17)
- 5개 새로운 config 파일 생성 (simulation, physics, source, nuclides, advanced)
- 자기 문서화, 일관된 `KEY: value` 형식, 물리적 의미 및 예제 포함
- 모듈화된 파서 함수 구현 (하나의 거대한 함수 → 5개 특화 파서)
- 완전한 하위 호환성 유지 (legacy 파일 자동 fallback)
- **상세 문서**: `docs/INPUT_MODERNIZATION_PLAN.md` 참조

### Comprehensive Input Validation (2025-10-17)
- 세계 최고 수준의 입력 검증 로직 (~600 lines)
- Fail-fast 철학: 잘못된 값 발견 시 즉시 종료
- Educational errors: 문제, 요구사항, 권장값, 예제, 수정 위치 모두 포함
- 물리적/통계적/지리적 타당성 검증 (단순 타입 체크 초월)
- 색상 코딩된 에러 메시지로 가시성 확보

### Configuration Simplification (2025-10-17)
- v1.0 릴리즈를 위한 실험적 기능 제거
- GPU 설정, source location 등 프로덕션 값으로 하드코딩
- IPC 구조체 크기 감소: 128 bytes → 80 bytes
- 설정 파일 복잡도 40% 감소, 코드베이스 단순화 (~100 lines 제거)

### True Emissions and Decay Constant IPC (2025-10-17)
- Python 하드코딩 완전 제거: true_emissions 배열 및 decay_constant 값
- C++에서 config 파일 읽어 공유 메모리로 전송
- EKIConfigFull 구조체 확장: 80 → 84 bytes (decay_constant 필드 추가)
- 별도 공유 메모리 세그먼트로 가변 길이 배열 지원 (`/dev/shm/ldm_eki_true_emissions`)
- 완전한 설정 파일 기반 시스템 구축 (NO 하드코딩)

### Python EKI 모듈 리팩토링 (2025-10-17)
- `Model_Connection_np_Ensemble.py` 모듈화 (600+ lines → 300+ lines)
- 공유 메모리 기능 분리: `eki_shm_config.py` 생성
  - `load_config_from_shared_memory()` - 설정 로드
  - `EKIConfigManager` 클래스 - 설정 캐싱
  - `receive_gamma_dose_matrix_shm_wrapper()` - 관측값 읽기
  - `send_tmp_states_shm()` - 앙상블 상태 전송
- 디버그 로깅 분리: `eki_debug_logger.py` 생성
  - 항상 활성화 (사용자 선택 불필요)
  - 단일 NPZ 파일로 통합 (`logs/debug/eki_debug_data.npz`)
  - 압축된 바이너리 형식으로 효율적 저장
  - 메모리 내 누적 후 디스크 저장

### Detailed Post-Processing Utility (2025-10-17)
- `util/detailed_postprocess.py` 생성 - 선택적 상세 분석 도구
- 디버그 데이터 추출:
  - NPZ 아카이브에서 모든 배열 텍스트로 변환
  - 통계 정보 (shape, dtype, min/max/mean/std) 자동 계산
  - 첫 100개 값 출력으로 데이터 검증 용이
- 개별 플롯 생성:
  - `compare_all_receptors.py`의 동일 함수 재사용
  - 원본 데이터로부터 고해상도 개별 플롯 생성
  - 수용체별 입자/선량 플롯 + 방출량 추정 플롯
- 입력 설정 요약:
  - 모든 config 파일의 핵심 값만 추출
  - Markdown 형식의 간결한 요약 생성
- 사용법: `python3 util/detailed_postprocess.py`
