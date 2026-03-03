# Plan: Step-by-step Experiment Roadmap (No Code Yet)

본 문서는 'project_Copernicus_FM/research.md`에 정리된 방법론/데이터셋 명세를 바탕으로, **코드 작성 전에** “무엇을 어떤 순서로” 진행할지 매우 세세하게 정의한 실행 계획서입니다.  
원칙: **한 번에 모든 실험을 하지 않고**, **단계별(steps)로**, **데이터셋도 전체를 바로 쓰지 않고 점진적으로** 확장합니다.

---

## 0) 프로젝트 규칙 (Ground Rules)

### 0.1 지금 단계에서 하지 않는 것

- 모델/로더/학습 코드 구현은 **아직 하지 않는다**.
- 성능 최적화/튜닝(하이퍼파라미터 탐색, 대규모 분산 학습)은 **나중 단계**로 미룬다.

### 0.2 진행 방식(필수)

- 모든 실행은 **Step 단위**로 진행하며, 각 Step은 아래를 반드시 포함한다.
  - **목표(Goal)**: 이 Step에서 “확인”할 것 1~3개
  - **입력(Input)**: 사용 데이터 범위(부분/전체), 채널/마스크 정의
  - **산출물(Deliverables)**: 저장 파일/리포트/그림/로그
  - **성공 기준(Exit Criteria)**: 다음 Step으로 넘어갈 조건
  - **리스크 & 대응(Risks & Mitigation)**: 실패 가능성과 대안
- 모든 실험은 `config/*.yaml`로 정의하고, 실행 결과는 `output/`에 **실험 ID** 기준으로 축적한다.

---

## 1) 목표/범위 (Goals / Scope)

### 1.1 최종 목표(High-level)

- Korea GEE-filtered 7채널 GeoTIFF 데이터셋을 활용해,
  - **Stage 1: SAR-to-RGB “단순 복원”** (상대적으로 구름 적은 타일 중심) 성능 비교
  - **Stage 2: Cloud removal** (Fmask 기반 마스킹) 성능 비교
- 비교 대상 모델 계열:
  - **GAN 계열(pix2pix/CycleGAN 계열 접근)** vs **Copernicus-FM (ViT/MIM 계열 접근)**

### 1.2 제약/가정(Assumptions)

- 데이터 경로(로컬): `~/data_2/SARtoRGB/Korea/`
- 파일: `*.tif` (BigTIFF GeoTIFF, multi-band). **여러 개의 scene 파일**이 존재하며(현재 로컬에서 확인된 예: 12개), 학습 샘플 수는 파일 개수와 별개로 **window 기반 patch 인덱싱 결과로 결정**된다.
- 기본 샘플 단위: 한 scene에서 추출한 **256×256 patch(window)** (`patch_size=256`, `stride=256`)
- 텐서(패치 로딩 결과): `(7, 256, 256)` / 10m per pixel
- 채널 순서(인덱스):
  - 0..3: S2 (B4, B3, B2, B8)
  - 4..5: S1 (VV, VH)
  - 6: Fmask (0~100 cloud probability, `s2cloudless` 기반)
- Copernicus-FM은 `Copernicus-FM/` 아래에 **.pth 가중치가 존재**한다. 다만 Python 구현은 레포 내에 없을 수 있으므로(Step 5), 실제 모델 인스턴스화 코드 경로를 먼저 확정한다.

---

## 2) 폴더 구조(계획) 및 책임 분리

아래 구조를 “최종적으로” 갖추되, Step별로 필요한 최소 파일부터 추가한다.

```
loader/
  __init__.py
  scenes.py             # scene(.tif) 탐색 + first_scene 선택(._ sidecar 제외)
  tiff_window.py        # BigTIFF window read(meta/window)
  indexing.py           # patch index(jsonl) 생성 + 통계
  preprocess.py         # cleaning/normalization + valid/cloud mask

model/
  __init__.py
  gan_pix2pix.py        # (나중) pix2pix 계열
  copernicus_fm.py      # Copernicus-FM builder placeholder(가중치는 존재, 구현은 별도 확정)
  heads.py              # (나중) task-specific head
  losses.py             # (나중) losses

scripts/
  01_discover_scenes.py        # Step 1: scene 목록 생성 + first_scene 고정
  02_build_index_first_scene.py# Step 2: first_scene patch index 생성
  02_train_stage1.py    # (나중) Stage 1 학습
  03_eval_stage1.py     # (나중) Stage 1 평가/그림
  04_train_stage2.py    # (나중) Stage 2 학습
  05_eval_stage2.py     # (나중) Stage 2 평가/그림
  06_ablation.py        # (나중) ablation

config/
  dataset.yaml
  stage1_gan.yaml
  stage1_copfm.yaml
  stage2_gan_inpaint.yaml
  stage2_copfm_mim.yaml
  eval.yaml
  logging.yaml

output/
  <exp_id>/
    config_resolved.yaml
    logs/
    checkpoints/
    metrics.json
    figures/
    samples/
```

**폴더별 원칙**

- `loader/`: “데이터를 읽고 텐서로 만드는” 책임만 가진다(모델/학습 로직 금지).
- `model/`: 네트워크 구조/forward/loss 최소 단위만 둔다(데이터 경로 접근 금지).
- `scripts/`: 실행 엔트리 포인트(학습/평가/시각화). `config/*.yaml`을 받아 동작.
- `config/`: 실험 재현성의 핵심. 모든 임계값(T), 정규화 방식, split seed, subset 크기를 여기로.
- `output/`: 어떤 Step에서든 결과는 “항상 output 아래”로.

---

## 3) 데이터셋 사용 계획 (Subset Strategy)

“한 번에 전부” 금지 원칙을 강제하기 위해, 모든 Step에서 **(1) 파일 범위(file_scope)** 와 **(2) patch 범위(patch_scope)** 를 명시한다.

### 3.0 데이터 단위 정의(고정)

- **scene(file)**: `~/data_2/SARtoRGB/Korea/*.tif` 중 하나
- **patch(sample)**: scene에서 `patch_size=256`, `stride=256`으로 추출한 window
- patch는 다음 schema로 인덱싱한다(고정):
  - `scene_path`, `x`, `y`, `w=256`, `h=256`, `patch_id`
  - 통계: `mean_fmask`, `cloud_ratio = mean(Fmask>=60)`, `valid_ratio = mean(Fmask<60)`, `nan_ratio`

### 3.1 샘플 규모 단계(권장)

- **Tiny**: first_scene 1개 + patch 32개 (인덱스 상위 N개) — 디버깅/형상 확인
- **Small**: first_scene 1개 + patch 256개 — 첫 학습/추론 파이프라인 확인
- **Medium**: first_scene 1개 + patch 전체 — 첫 의미 있는 비교
- **Large**: 모든 scene + patch 일부 → patch 전체 — 최종 보고용(마지막에만)

### 3.2 샘플링 원칙

- 항상 seed 고정(재현성).
- 가능하면 타일 크기(H/W)와 Fmask 평균값 분포가 고르게 들어가도록 층화(가능하면).
- Stage 1에서는 “상대적으로 구름이 적은(clean)” 타일을 우선 샘플링.
- Stage 2에서는 “구름이 충분히 있는(cloudy)” 타일을 우선 샘플링.

### 3.3 Cloud 정책(고정)

- **학습/평가에서 구름 픽셀 제외**: `valid_pixel = (Fmask < 60)`만 loss/metric에 사용
- Stage 2 “구름 영역” 정의: `cloud_pixel = (Fmask >= 60)`

---

## 4) Step-by-step 로드맵 (매우 세세한 실행 계획)

아래 Step은 “진짜로” 순서대로 진행한다.  
각 Step은 완료 기준을 통과해야 다음으로 이동한다.

### Step 0 — 리포/환경 준비(코드 작성 전 정리)

**Goal**

- 실험이 “step-by-step”로 누적되도록, 폴더/설정/출력 규칙을 확정한다.

**To-do**

1. `output/` 디렉토리 네이밍 규칙 확정: 예) `YYYYMMDD_stepXX_<model>_<subset>_<seed>`
   - (v2 고정) `YYYYMMDD_stepXX_<stage>_<model>_<fileN>_<patchN>_seed<seed>`
2. config 관리 규칙 확정:
   - 모든 config는 `config/*.yaml`
   - 실행 시 “resolved config(병합된 최종값)”를 `output/<exp_id>/config_resolved.yaml`로 저장
3. metric 저장 포맷 확정:
   - `metrics.json`: scalar metrics (PSNR/SSIM/RMSE 등)
   - `figures/`: 정성 비교(확대 포함)
4. 실험 로그 기준 확정:
   - stdout 로그 + 최소한의 학습곡선(가능하면)

**Deliverables**

- 이 문서(`plan.md`)를 기준으로 진행.

**Exit Criteria**

- “어떤 Step이 끝나면 어디에 뭐가 남는지”가 모호하지 않음.

---

### Step 1 — 데이터 인벤토리(파일 존재/형상/범위) 점검

**Goal**

- 실제 데이터가 명세대로 존재하는지 확인한다(경로/파일 수/채널 수/크기 분포/NaN/Inf 존재).

**Input**

- 로컬 데이터 경로: `~/data_2/SARtoRGB/Korea/` (전체를 쓰되, 처음엔 “읽기만”)

**To-do (정밀 체크리스트)**

1. 파일 리스트 수집
   - `.tif` 파일 개수(여러 scene 파일; **patch 개수와 다름**)
   - `._*.tif` macOS sidecar 파일은 **무조건 제외**
   - 파일명 규칙/중복/깨진 파일 여부
2. 랜덤 샘플(예: 16개)로 빠른 로드 검증
   - 밴드 수가 7인지
   - dtype이 float32로 읽히는지(또는 변환 필요 여부)
   - 값 범위가 대략 명세와 부합하는지(S2 0~10000, S1 dB 범위, Fmask 0~100)
3. NaN/Inf 비율 측정
   - 타일별 NaN/Inf 픽셀 비율
   - “극단값(outlier)” 존재 여부(예: S1 dB가 비정상적으로 큼/작음)
4. scene 크기(W,H) 확인 및 patch 정책 검증
   - patch_size(256)로 stride(256) window를 생성했을 때 patch 개수 예측
   - patch read 결과가 항상 `(7,256,256)`이 되도록 window read 전략 확정
5. Fmask 통계
   - 타일별 `mean(Fmask)` 분포
   - 픽셀별 `Fmask >= T` 비율 분포(T 후보: 60/80/90)

**Deliverables**

- `output/step01_dataset_inventory/` 아래:
  - `summary.json` (파일수, H/W 분포, NaN/Inf 통계, Fmask 통계)
  - `figures/` (분포 히스토그램: H,W, mean(Fmask), cloud-pixel-ratio)

**Exit Criteria**

- 7채널/범위/결측치 특성이 “예상과 얼마나 다른지”가 숫자로 요약됨.
- “입력 크기 통일 전략”에 대한 결론 후보가 최소 1개 도출됨.

**Risks & Mitigation**

- (Risk) 밴드 순서가 명세와 다름 → (Mitigation) 몇 개 타일을 RGB 합성해서 육안 검증(후속 Step에 포함).
- (Risk) H/W가 너무 다양함 → (Mitigation) 최소 입력 정책(예: 256 crop)부터 시작 후 확장.

---

### Step 2 — Patch Index 생성(단, first_scene 1개) + schema 확정

**Goal**

- “patch/sample”을 window 기반으로 고정하고, 재현 가능한 index를 생성한다.

**Decisions (고정)**

- `patch_size = 256`
- `stride = 256` (non-overlap; 가장 단순)
- index record schema:
  - `scene_path`, `x`, `y`, `w`, `h`, `patch_id`
  - `mean_fmask`, `cloud_ratio`, `valid_ratio`, `nan_ratio`
- 저장 포맷: `jsonl` (1 patch = 1 line)
- 저장 위치: `output/index/`

**To-do**

1. `first_scene`를 정렬 기준으로 1개 선택(Discover 단계에서 고정)
2. first_scene를 window로 순회하며 index 생성
   - 구현 시 BigTIFF window read가 필요하므로 `rasterio`가 필요할 수 있음(환경 의존성).
3. 통계/히스토그램 산출: `mean_fmask`, `cloud_ratio`, `valid_ratio`, `nan_ratio`
4. 초기 patch 필터링 규칙(권장 시작값)
   - drop: `nan_ratio > 0`
   - Stage 1 train candidate: `valid_ratio >= 0.95`
   - Stage 2 inference candidate: `cloud_ratio >= 0.10`

**Deliverables**

- `output/index/first_scene.index.jsonl`
- `output/step03_index_stats/summary.json`
- `output/step03_index_stats/decision.md`
- `output/step03_index_stats/figures/*.png` (가능한 경우)

**Exit Criteria**

- patch pool이 “추측이 아닌 index 결과”로 확정된다.

---

### Step 3 — 전처리 정의(클리닝/정규화/마스크)

**Goal**

- 모델이 “수치 스케일 차이”에 압도되지 않도록 전처리를 고정한다.

**To-do**

1. Cleaning(필수)
   - `NaN`, `+/-Inf`를 0으로 치환
   - (선택) 극단값 clip 규칙 후보 정의
2. Normalization(필수)
   - S2: `/10000.0` (0~1)
   - S1: `(x + 20) / 20` (근사; clip 범위는 실험으로 확정)
   - Fmask: `/100.0` (0~1)
3. Mask 생성 규칙(Stage 2용)
   - pixel mask: `cloud = (Fmask >= T)`
   - boundary ring 평가용: `dilate(cloud, k) - cloud` (k=3~5 후보)
4. “clean 타일” 정의(Stage 1용)
   - tile-level: `mean(Fmask) < t_clean` 후보(예: 20)
   - exclude rule: `mean(Fmask) >= 80` 제외(사용자 규칙)

**Deliverables**

- `config/dataset.yaml`에 들어갈 파라미터 리스트(초안) 작성:
  - `nan_policy`, `s2_scale`, `s1_norm_offset`, `s1_norm_scale`, `fmask_scale`
  - `cloud_threshold_T`, `tile_clean_threshold`, `tile_drop_threshold`

**Exit Criteria**

- Stage 1/2에서 “데이터 선정과 마스크”가 같은 규칙으로 재현 가능.

---

### Step 4 — (코드 없이) 모델 입력/출력 정의 확정

**Goal**

- 각 실험에서 모델이 실제로 무엇을 입력받고 무엇을 예측할지 “명확히” 고정한다.

**Stage 1 (고정)**

- 입력: `S1(VV,VH)` 2채널
- 출력(타깃): `S2(B4,B3,B2,B8)` = **RGB+NIR 4채널**
- Cloud exclusion(고정): **구름 픽셀은 학습/평가에서 제외**
  - `valid_pixel = (Fmask < 60)`만 loss/metric에 사용

**Stage 2 (초기 고정: inference 중심, GT 없음)**

- 입력: `S1(VV,VH) + S2_observed(B4,B3,B2,B8) + cloud_pixel mask`
- 출력: `S2_reconstructed(B4,B3,B2,B8)`
- 합성(final): `final = where(cloud_pixel, pred, observed)` (구름 영역만 교체)

**To-do**

- Stage 1/2는 위 “고정 spec”으로 시작하고, 입력 채널 확장은 ablation으로 분리한다.
- Copernicus-FM이 요구하는 입력 포맷(토큰화/채널)과 맞추는 계획을 문서화.

**Deliverables**

- `output/step04_io_spec/io_spec.md`:
  - 입력 텐서 shape, 채널, 정규화, 출력 범위, loss 적용 영역(전체/마스크)

**Exit Criteria**

- 학습/평가 스크립트가 “입출력 혼선” 없이 작성 가능할 만큼 명확해짐.

---

### Step 5 — Copernicus-FM 코드(이미 있는 파일) 읽고 “적용 방식” 확정

**Goal**

- Copernicus-FM을 “이 프로젝트에서 어떻게 쓸지”를 결정한다(그냥 가져다 쓰는지, head만 붙이는지, 어떤 입력 채널을 받을지).

**To-do**

1. 레포 내 Copernicus-FM 구현 파일 위치 확인
2. 아래 항목을 문서로 정리(코드 작성 금지, 읽기/요약만)
   - 입력 채널 수 제한/가정(3채널? 다채널?)
   - patch size / tokenizer 규칙
   - MIM 학습 또는 fine-tuning 인터페이스(가능한지)
   - 메타데이터 입력을 실제로 받는지(받는다면 어떤 포맷인지)
   - output head가 무엇인지(분류/복원/회귀)
3. “Stage 1/2에 쓰기 위한 최소 래퍼 설계”를 텍스트로 정의

**Deliverables**

- `output/step05_copfm_reading/notes.md`:
  - Copernicus-FM 적용 설계(채널/헤드/손실/학습 방식) 초안

**Exit Criteria**

- Copernicus-FM 실험이 “어떤 코드 구조로” 진행될지 청사진이 완성됨.

**Risks & Mitigation**

- (Risk) Copernicus-FM이 다채널 입력을 바로 못 받음 → (Mitigation) (a) 채널 프로젝션 레이어 계획, (b) RGB만 넣고 SAR를 다른 경로로 조건화하는 계획 중 택1.

---

### Step 6 — 실험 로깅/평가 산출물(PSNR/SSIM/정성 비교) 정의

**Goal**

- 어떤 모델이든 동일한 기준으로 비교할 수 있게 “평가 산출물”을 고정한다.

**To-do**

1. 공통 scalar metrics 정의
   - Stage 1: PSNR, SSIM (+ 선택 RMSE/MAE)
   - Stage 2: 전체/ROI(마스크)/경계 링 별 PSNR/SSIM/RMSE
2. 정성 비교 템플릿 정의(figure layout)
   - 입력(SAR VV/VH) 시각화 규칙(dB→정규화→그레이)
   - GT RGB / Pred RGB / |diff| / mask overlay
   - 확대 crop(특정 관심영역) 2~3개 고정
3. 결과 저장 규칙 정의
   - `metrics.json` 스키마
   - `figures/` 파일명 규칙: `<tile_id>_<view>.png`

**Deliverables**

- `output/step06_eval_spec/eval_spec.md`

**Exit Criteria**

- “평가 루틴이 하나로 고정”되어 이후 Step에서 혼란이 없음.

---

### Step 7 — Stage 1 “Tiny” 실행 계획(학습이 돌아가는지 확인)

**Goal**

- 아주 작은 데이터로 end-to-end 파이프라인이 성립하는지 확인한다(나중에 코드로).

**Input**

- file_scope: `first_scene` 1개
- patch_scope: Tiny = index 상위 **32 patches** (Stage 1 train candidate 조건 만족하는 patch만)

**Model Plan**

- GAN: 가장 단순한 pix2pix 계열(조건부)로 계획
- Copernicus-FM: Step 5에서 정의한 최소 래퍼로 계획

**To-do**

1. seed 고정 split(예: 12/2/2)
2. 학습 epoch/iteration 최소값으로 “한 번” 돌아가게 설정
3. overfit 체크(의도적으로 tiny에서 훈련 loss가 내려가는지)

**Deliverables**

- `output/step07_stage1_tiny_plan/runbook.md`:
  - 실행 config(초안), 예상 runtime, 관찰 포인트(학습곡선/샘플)

**Exit Criteria**

- “코드 구현 후” tiny에서 최소한의 학습/추론이 가능하다는 구체 runbook이 존재.

---

### Step 8 — Stage 1 “Small→Medium” 비교 실험 계획(첫 의미 있는 비교)

**Goal**

- GAN vs Copernicus-FM의 **첫 정량 비교**를 가능하게 한다.

**Input**

- file_scope: `first_scene` 1개
- patch_scope:
  - Small = **256 patches**
  - Medium = **first_scene 전체 patches**

**To-do**

1. clean 기준(t_clean) 확정(예: mean(Fmask)<20)
2. 동일 split/seed로 두 모델 비교(공정성)
3. 보고서 템플릿 정의:
   - 표: PSNR/SSIM 평균±표준편차
   - 그림: 대표 8장(성공/실패 케이스 포함)
   - 관찰: “질감 뭉개짐” vs “전역 구조” 사례 서술

**Deliverables**

- `output/step08_stage1_compare_plan/report_template.md`

**Exit Criteria**

- “Stage 1에서 누가 어떤 면에서 이겼는지”를 쓸 수 있는 최소 결과가 나올 준비가 됨.

---

### Step 9 — Stage 2 마스킹/ROI 평가 계획 확정(Cloud removal 준비)

**Goal**

- Stage 2는 Stage 1과 다르게 “어디를 복원하는지(ROI)”가 핵심이므로, 마스크/평가가 흔들리지 않게 고정한다.

**To-do**

1. cloud threshold 고정: `T = 60`
2. ROI(구름 영역) 정의 고정
   - `cloud_pixel = (Fmask >= 60)`
   - `valid_pixel = (Fmask < 60)`
3. Stage 2는 GT가 없으므로, 초기 목표는 “inference 산출물 확보 + 정성 분석”으로 고정

**Deliverables**

- `output/step09_stage2_masking_plan/masking_plan.md`

**Exit Criteria**

- Stage 2 실험에서 “구름 제거 성능”을 객관적으로 비교 가능.

---

### Step 10 — Stage 2 “Tiny” 실행 계획(인페인팅 파이프라인 검증)

**Goal**

- GT 없이도 “구름 영역만 교체하는” inference 산출물이 나오는지 확인할 runbook을 만든다.

**Input**

- file_scope: `first_scene` 1개
- patch_scope: Tiny = index 상위 **32 patches** (Stage 2 inference candidate 조건 만족하는 patch만)

**To-do**

1. Stage 2 inference candidate 정의 확정
   - 예: `cloud_ratio >= 0.10` (구름 픽셀이 충분히 있는 patch만)
2. 입력/출력/합성 고정(spec 재확인)
   - 입력: `S1(VV,VH) + S2_observed(B4,B3,B2,B8) + cloud_pixel`
   - 출력: `S2_reconstructed(B4,B3,B2,B8)`
   - final: `where(cloud_pixel, pred, observed)`
3. 저장 산출물(정성 중심) 템플릿 확정
   - observed / mask overlay / pred / final / boundary zoom

**Deliverables**

- `output/step10_stage2_tiny_plan/runbook.md`

**Exit Criteria**

- Stage 2 코드를 만들기 위한 상세 실행 계획이 완성.

---

### Step 11 — Stage 2 “Small→Medium” 비교 실험 계획(첫 Cloud removal 비교)

**Goal**

- “경계 자연스러움(seamless boundary)” 및 ROI 복원 품질을 첫 비교한다.

**Input**

- file_scope: `first_scene` 1개 → (통과 후) all scenes 확장
- patch_scope:
  - Small = **256 patches**
  - Medium = **first_scene 전체 patches**

**To-do**

1. 동일 split/seed 및 동일 T로 비교(공정성)
2. Failure case 카테고리 정의(예시)
   - ROI 내부 색 번짐/채도 붕괴
   - 경계 seam
   - SAR 텍스처가 RGB로 잘못 투영(가짜 패턴)
3. 정성 비교 figure를 “경계 확대” 중심으로 구성

**Deliverables**

- `output/step11_stage2_compare_plan/report_template.md`

**Exit Criteria**

- Cloud removal에서의 구조적 장단점(GAN inpainting vs MIM)이 관찰 가능한 수준의 결과가 나올 준비.

---

### Step 12 — Ablation 계획(필요할 때만)

**Goal**

- 성능 차이의 원인을 분해한다(단, 비교 실험이 먼저).

**Ablation 후보**

- 입력 채널 변화:
  - (VV,VH)만 vs (VV,VH,Fmask) vs (VV,VH + cloudy RGB + mask)
- 출력 채널 변화:
  - RGB만 vs RGB+NIR
- 전처리 변화:
  - S1 정규화 offset/scale 변화, clipping 여부
- 마스크 임계값 변화:
  - T=60 vs 80 vs 90

**Deliverables**

- `output/step12_ablation_plan/ablation_matrix.md` (실험 조합 표)

**Exit Criteria**

- “왜 Copernicus-FM이 더 낫다/왜 GAN이 더 낫다”를 설명 가능한 최소 근거 확보.

---

### Step 13 — 최종 확장 계획(Large, all scenes + all patches) 및 보고서 구성

**Goal**

- 마지막에만 전체 데이터로 확장하여, 최종 성능을 보고한다.

**To-do**

1. 최종 실험 선정(무조건 다 하지 않음)
   - Stage 1 최상 1개 설정
   - Stage 2 최상 1개 설정
2. 계산 자원/시간 추정
3. 최종 결과물 구성(논문/보고서 Methodology 섹션에 바로 반영 가능 형태)

**Deliverables**

- `output/step13_final_plan/final_report_outline.md`

**Exit Criteria**

- “어떤 두 개 실험을 최종으로 가져갈지”가 결정됨.

---

## 5) 즉시 확인할 질문(모호성 제거용)

아래는 코드 작성 전에, 계획의 빈칸을 메우기 위해 확인이 필요한 항목들이다(답을 받으면 plan에 고정).

1. **Copernicus-FM 코드가 들어있는 파일 경로/모듈명**은 무엇인가?
2. Stage 1의 “정답 RGB”는 어떤 채널 조합인가?
   - 기본: S2(B4,B3,B2) = RGB
3. Stage 2에서 “cloudy RGB” 입력이 실제로 데이터에 존재하는가?
   - 현재 데이터는 S2 4채널 + Fmask를 포함하므로, cloudy/clean 구분은 Fmask 기반으로 만들 계획(정확한 정의 필요).
4. 타일 크기(256~512px) 처리 정책:
   - crop 256로 고정할지, pad/resize할지 최종 선택
