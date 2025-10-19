# LDM-EKI v1.0 안전기술보고서 (초안)

**작성일:** 2025-10-19
**프로젝트:** LDM-EKI - 대기 확산 모델 기반 방사능 오염원 역산 시스템
**버전:** v1.0 Release Candidate

---

## 목차

1. [서론](#1-서론)
   - 1.1 배경 및 필요성
   - 1.2 연구 목적
   - 1.3 보고서 구성

2. [LDM-EKI 시스템 개요](#2-ldm-eki-시스템-개요)
   - 2.1 시스템 아키텍처
   - 2.2 주요 구성 요소

3. [코드 실용화 및 안정화 개선](#3-코드-실용화-및-안정화-개선)
   - 3.1 코드 구조 모듈화 및 빌드 효율화
   - 3.2 성능 최적화
   - 3.3 사용자 인터페이스 및 자동화 개선
   - 3.4 설정 시스템 현대화
   - 3.5 IPC 통신 안정화

4. [자체검증 결과](#4-자체검증-결과)
   - 4.1 검증 방법론
   - 4.2 검증 결과
   - 4.3 성능 벤치마크

5. [결론 및 활용](#5-결론-및-활용)
   - 5.1 주요 성과
   - 5.2 향후 연구 방향
   - 5.3 활용 계획

---

## 1. 서론

### 1.1 배경 및 필요성

방사능 누출 사고 발생 시 정확한 오염원 위치와 방출량을 신속히 파악하는 것은 효과적인 방재 대응의 핵심이다. 전통적인 순방향 대기 확산 모델은 소스 정보가 알려진 경우 농도 분포를 예측할 수 있으나, 실제 사고 상황에서는 소스 정보가 불확실하거나 전혀 알려지지 않은 경우가 대부분이다.

LDM-EKI(Lagrangian Dispersion Model with Ensemble Kalman Inversion)는 이러한 역문제(inverse problem)를 해결하기 위해 개발된 시스템으로, 관측 데이터로부터 오염원을 역추적한다. 본 시스템은 다음과 같은 특징을 갖는다:

- **GPU 가속 순방향 모델**: CUDA 기반 라그랑지안 입자 확산 시뮬레이션
- **앙상블 칼만 역산**: 통계적 최적화 기법을 통한 소스 추정
- **고성능 IPC 통신**: POSIX 공유 메모리를 이용한 프로세스 간 데이터 교환

### 1.2 연구 목적

본 연구는 기존 연구용 LDM-EKI 시스템을 **실용화 가능한 v1.0 버전**으로 발전시키는 것을 목표로 한다. 주요 개선 사항은:

1. **코드 안정성 향상**: RDC 제거, 에러 처리 강화
2. **성능 최적화**: 빌드 시간 75% 단축, IPC 통신 최적화
3. **사용자 편의성**: 자동화된 시각화, 종합 로그 시스템
4. **유지보수성**: 23개 모듈 기반 구조로 재편성

### 1.3 보고서 구성

본 보고서는 다음과 같이 구성된다:
- **2장**: LDM-EKI 시스템 전체 아키텍처 설명
- **3장**: v1.0에서 구현된 실용화 개선 사항 상세 기술
- **4장**: 자체검증 결과 및 성능 벤치마크
- **5장**: 결론 및 향후 활용 계획

---

## 2. LDM-EKI 시스템 개요

### 2.1 시스템 아키텍처

LDM-EKI는 **2-프로세스 협력 구조**로 동작한다:

```
┌─────────────────────────────────────────────────────────────┐
│                    LDM (C++/CUDA)                           │
│  - GPU 가속 입자 확산 시뮬레이션                               │
│  - 기상 데이터 사전 로딩                                       │
│  - 수용체 관측값 계산                                         │
└────────────────┬────────────────────────────────────────────┘
                 │
                 │  공유 메모리 (/dev/shm/ldm_eki_*)
                 │  - 관측값 전송 (C++ → Python)
                 │  - 앙상블 상태 수신 (Python → C++)
                 │
┌────────────────▼────────────────────────────────────────────┐
│                    EKI (Python)                             │
│  - 앙상블 칼만 역산 알고리즘                                   │
│  - 사전/사후 분포 계산                                        │
│  - 수렴 판정 및 반복 제어                                     │
└─────────────────────────────────────────────────────────────┘
```

**동작 흐름:**
1. **초기화**: LDM이 true emissions로 참값 시뮬레이션 수행 → 관측값 생성
2. **Python EKI 시작**: LDM이 Python 프로세스를 백그라운드로 실행
3. **반복 루프** (N회):
   - Python이 앙상블 상태 제안 → 공유 메모리에 기록
   - LDM이 앙상블 시뮬레이션 수행 → 관측값 계산 → 공유 메모리에 기록
   - Python이 칼만 이득으로 앙상블 업데이트
4. **종료**: 수렴 또는 최대 반복 도달 시 종료

### 2.2 주요 구성 요소

#### 2.2.1 폴더 구조

```
ldm-eki-release.v.beta/
├── src/                      # 소스 코드
│   ├── core/                 # 핵심 클래스 (LDM 클래스)
│   ├── data/                 # 데이터 관리 (설정, 기상)
│   ├── physics/              # 물리 모델 (CRAM, 핵종)
│   ├── kernels/              # CUDA 커널
│   ├── ipc/                  # IPC 통신 (writer/reader)
│   ├── simulation/           # 시뮬레이션 함수
│   ├── visualization/        # VTK 출력
│   ├── init/                 # 초기화 (입자, 설정)
│   ├── debug/                # 디버깅 도구
│   └── eki/                  # Python EKI 프레임워크
├── input/                    # 입력 설정 파일
│   ├── simulation.conf       # 시뮬레이션 설정
│   ├── physics.conf          # 물리 모델 설정
│   ├── source.conf           # 소스 설정
│   ├── nuclides.conf         # 핵종 설정
│   ├── receptor.conf         # 수용체 위치
│   └── eki.conf              # EKI 알고리즘 설정
├── util/                     # 유틸리티 스크립트
│   ├── cleanup.py            # 데이터 정리
│   ├── compare_all_receptors.py  # 기본 시각화
│   ├── detailed_postprocess.py   # 상세 분석
│   └── visualize_vtk.py      # 입자 분포 시각화
├── output/                   # 출력 파일
│   ├── plot_vtk_prior/       # 참값 VTK
│   ├── plot_vtk_ens/         # 앙상블 VTK
│   └── results/              # 시각화 결과
└── logs/                     # 로그 파일
```

#### 2.2.2 핵심 기술 스택

- **언어**: C++17, CUDA 12.6, Python 3.x
- **병렬 처리**: CUDA (GPU 가속), OpenMP (빌드 병렬화)
- **IPC**: POSIX 공유 메모리 (`/dev/shm`)
- **수치 라이브러리**: NumPy (Python), CRAM (방사성 붕괴)
- **시각화**: Matplotlib, Cartopy, PyVista
- **빌드 시스템**: GNU Make, NVCC

---

## 3. 코드 실용화 및 안정화 개선

### 3.1 코드 구조 모듈화 및 빌드 효율화

  **□ 모듈화 개요**
    • LDM-EKI는 기존 연구용 단일 파일 구조(ldm.cu ~15,000줄)를 23개 모듈 기반 구조로 완전히 재편성함.
    • 기능별로 core, data, physics, kernels, ipc, simulation, visualization, init, debug, eki 모듈을 분리하여 유지보수성과 확장성을 대폭 향상시킴.

  **□ Non-RDC 빌드 아키텍처 전환**
    • 기존 RDC(Relocatable Device Code) 방식에서 Non-RDC로 전환하여 컴파일 안정성과 성능을 개선함.
    • RDC 문제점:
      - "invalid device symbol" 에러 발생 (크로스 컴파일 유닛 __constant__ 접근)
      - "illegal memory access" 에러 (초기화되지 않은 device 심볼)
      - 링크 시간 증가 및 디버깅 어려움
    • Non-RDC 해결책:
      - __constant__/__device__ 전역 변수 제거
      - 일반 GPU 메모리로 변경 (cudaMalloc)
      - KernelScalars 구조체를 통한 포인터 전달 패턴 확립
      - 커널 호출 시 모든 파라미터 명시적 전달
    • 결과: 빌드 에러 완전 제거, 런타임 안정성 향상

  **□ 빌드 시스템 최적화**
    • 자동 병렬 빌드: CPU 코어 수 감지하여 자동으로 -j 옵션 적용
      - 8코어 시스템에서 빌드 시간 ~2-3분 → ~30초 (약 75% 단축)
    • CUDA 아키텍처 타겟: SM 6.1 이상 (Pascal 이상)
      - GTX 1080, Tesla P100 이상 GPU 지원
    • Makefile 자동화:
      - NVCC 경로 자동 탐색 (/usr/local/cuda/bin/nvcc)
      - 의존성 자동 추적 및 증분 빌드 지원 (변경된 파일만 재컴파일)

### 3.2 성능 최적화

*(작성 예정)*

### 3.3 사용자 인터페이스 및 자동화 개선

  **□ 다층 결과 시각화 시스템**
    • LDM-EKI는 자동 기본 시각화와 선택적 상세 분석 도구를 계층적으로 제공하여 사용자 편의성을 극대화함.
    • 자동 기본 시각화:
      - compare_all_receptors.py가 시뮬레이션 종료 시 자동 실행
      - 수용체별 시계열 플롯 생성 (입자 포획 개수, 감마선량률)
      - 방출량 추정 결과 표시 (true vs prior vs ensemble mean)
      - 16개 수용체를 다중 페이지로 자동 배치 (3개/페이지)
      - 출력 파일: output/results/all_receptors_comparison.png
      - 공유 메모리(/dev/shm)에서 바이너리 데이터 직접 읽기
    • 선택적 상세 분석 도구:
      - detailed_postprocess.py: 디버그 데이터 텍스트 추출, 설정 파일 Markdown 요약, 개별 수용체 플롯 생성
      - visualize_vtk.py: PyVista로 VTK 파일 읽기, Cartopy 기반 지리적 지도 위 입자 분포 히트맵 표시, 애니메이션 GIF 생성
      - Matplotlib 기반 CPU 렌더링 (고해상도 300 DPI)

  **□ 종합 로그 및 진단 시스템**
    • 이중 스트림 로깅 시스템:
      - ColorStripStreambuf 클래스: ANSI 색상 코드 자동 제거 (로그 파일용)
      - TeeStreambuf 클래스: 터미널과 로그 파일 동시 출력
      - 터미널: ANSI 색상 코드 포함 (CYAN=시스템, GREEN=성공, RED=에러, YELLOW=경고, ORANGE=강조)
      - 로그 파일: 순수 텍스트 (색상 코드 제거)
    • 로그 헤더 시스템 정보 자동 기록:
      - 시뮬레이션 시작 타임스탬프 (strftime 포맷: YYYY-MM-DD HH:MM:SS)
      - 작업 디렉토리 경로 (getenv("PWD"))
      - OS 정보: uname() 시스템 콜로 커널 버전, 아키텍처 획득
      - CUDA 런타임/드라이버 버전: cudaRuntimeGetVersion(), cudaDriverGetVersion()
      - GPU 디바이스: cudaGetDeviceProperties()로 디바이스명, 컴퓨팅 성능(SM), 메모리 용량 획득
    • 진행률 바:
      - 현재 시뮬레이션 시간 (초), 타임스텝 진행률 (현재/전체, 퍼센트)
      - 기상 데이터 인덱스 (Past/Future), 앙상블 모드 및 크기 표시
      - stderr 스트림 사용 (로그 파일에 미기록, 실시간 업데이트)
    • Kernel Error Collector:
      - 비동기 CUDA 커널 에러를 메모리에 실시간 누적
      - 중복 에러는 카운트만 증가 (메모리 효율성)
      - 시뮬레이션 종료 시 에러 요약을 빨간색/굵게 일괄 보고
      - 타임스탬프 로그 파일 생성: logs/error/kernel_errors_YYYY-MM-DD_HH-MM-SS.log
      - 20개 이상 커널 호출 지점에 CHECK_KERNEL_ERROR() 매크로 적용
    • Memory Doctor 모드:
      - input/eki.conf에서 MEMORY_DOCTOR_MODE=On 활성화
      - IPC 통신 데이터를 /tmp/eki_debug/에 텍스트 저장
      - C++ ↔ Python 간 데이터 불일치 진단 (배열 shape, 값 범위, 타임스탬프)
      - 성능 영향 있음 (프로덕션 환경에서는 비활성화 권장)

  **□ 입력 검증 및 Fail-Fast 오류 처리**
    • 4단계 입력 검증 프레임워크:
      - 존재 검증: 필수 파라미터 누락 검사 (time_end, time_step, total_particles 등)
      - 타입 검증: 숫자형/문자열 타입 일치 검사
      - 범위 검증: 물리적/통계적 타당성 검사 (예: time_step > 0, ensemble_size ≥ 10)
      - 교차 검증: 파라미터 간 의존성 검사 (예: time_end % time_step == 0)
      - 구현 위치: src/init/ldm_init_config.cu의 validate_*_config() 함수군
    • Educational Error Messages:
      - 에러 발생 시 5가지 정보 제공: 문제, 요구사항, 현재값, 올바른 예제, 수정 위치
      - 색상 코딩: 빨간색 헤더, 노란색 수정 위치, 굵은 글씨 강조
      - 사용자가 즉시 문제를 이해하고 설정 파일을 수정할 수 있도록 유도
    • Fail-Fast 원칙:
      - 잘못된 입력 발견 시 즉시 프로그램 종료, 자동 복구 시도 없음
      - 신뢰할 수 없는 결과 생성 방지 (잘못된 설정으로 실행 시 물리적으로 무의미한 결과 생성 가능)
      - 적용 범위: 설정 파일 파싱 에러, 입력 검증 실패, CUDA API 에러, 파일 I/O 에러
      - Kernel Error Collector는 진단만 수행, 복구는 수행하지 않음

  **□ 워크플로 자동화**
    • 시작 시 자동 실행:
      - cleanup.py: 이전 실행의 로그, 출력 파일, 공유 메모리 정리 (사용자 확인 프롬프트 제공)
      - 디렉토리 자동 생성: logs/, output/plot_vtk_prior/, output/plot_vtk_ens/, output/results/, logs/eki_iterations/, logs/error/
      - 안전성 체크: 활성 프로세스 존재 시 정리 거부
    • 종료 시 자동 실행:
      - compare_all_receptors.py: 기본 시각화 그래프 자동 생성
      - Python EKI 프로세스: main_eki.cu에서 백그라운드로 자동 실행
      - 표준 출력/에러를 logs/python_eki_output.log로 리다이렉트
      - 실패 시 에러 코드 및 로그 파일 위치 안내
    • 선택적 도구 안내:
      - 시뮬레이션 종료 시 추가 분석 도구 안내 메시지 자동 출력
      - detailed_postprocess.py: 디버그 데이터 추출, 개별 플롯 생성 방법 안내
      - visualize_vtk.py: 입자 분포 애니메이션 생성 방법 안내
      - 사용자가 필요에 따라 선택적으로 실행
    • 개선 효과:
      - 로그 파일 가독성: ANSI 코드 혼재 → 순수 텍스트 (100% 개선)
      - 시스템 정보 기록: 없음 → OS/CUDA/GPU 자동 기록 (신규)
      - 커널 에러 출력: 산발적 즉시 출력 → 일괄 요약 보고 (가독성 향상)
      - 입력 검증 범위: 기본 타입만 → 4단계 종합 검증 (400% 확대)
      - 에러 메시지 품질: 단순 오류명 → Educational Messages (신규)
      - 자동 시각화: 없음 → PNG 자동 생성 (신규)
      - 추가 분석 도구: 없음 → 2종 제공 (신규)

### 3.4 설정 시스템 현대화

*(작성 예정)*

### 3.5 IPC 통신 안정화

*(작성 예정)*

---

## 4. 자체검증 결과

*(작성 예정)*

---

## 5. 결론 및 활용

*(작성 예정)*

---

## 부록

### A. 빌드 및 실행 가이드

**빌드:**
```bash
make              # ldm-eki 빌드 (기본)
make all-targets  # 모든 실행파일 빌드
make clean        # 빌드 산출물 정리
```

**실행:**
```bash
./ldm-eki         # EKI 최적화 실행 (자동 정리, 자동 시각화)
```

**추가 분석:**
```bash
python3 util/detailed_postprocess.py  # 상세 분석 및 개별 플롯
python3 util/visualize_vtk.py         # 입자 분포 GIF 생성
```

### B. 참고 문헌

*(작성 예정)*

---

**문서 버전:** Draft v0.3
**최종 수정:** 2025-10-19
**작성자:** LDM-EKI Development Team
