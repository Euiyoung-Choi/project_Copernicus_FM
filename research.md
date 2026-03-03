# Research: SAR-to-Optical Image Translation for Cloud Removal

## 1. Methodology: Model Comparison

본 연구에서는 Sentinel-1(SAR)과 Sentinel-2(Optical) 융합 입력을 활용하여 **구름 제거(Cloud removal)** 및 **SAR-to-RGB 생성**을 수행하기 위해, 전통적인 생성 모델인 **GAN(pix2pix/CycleGAN)** 과 자기지도학습 기반의 **Copernicus-FM**을 비교 분석한다. 본 문서는 **GEE(Google Earth Engine)로 필터/타일링하여 구축한 한국 지역 패치 데이터셋**을 기준으로, 두 모델 계열의 **구조적 차이**와 그로부터 파생되는 **강점/제약**, 그리고 재현 가능한 실험 프로토콜을 정리한다.

### 1.1 Baseline: Generative Adversarial Networks (GAN)

#### 모델 구조

- **pix2pix(conditional GAN)**: (입력, 정답) 쌍이 존재하는 **지도학습 paired translation**을 가정한다. 일반적으로 U-Net 계열의 **Encoder–Decoder 생성기**와 PatchGAN 계열 **판별기** 조합을 사용한다.
- **CycleGAN(unpaired translation)**: paired 데이터가 부족한 경우를 위해 **Cycle-consistency**를 도입하여, 두 도메인 간 양방향 매핑을 동시에 학습한다.

#### 작동 원리 (학습 신호)

- 생성기(Generator)가 입력(예: SAR 또는 SAR+Cloudy Optical)을 받아 **가짜 광학(RGB) 영상**을 생성한다.
- 판별기(Discriminator)가 생성 결과와 실제 광학 영상의 분포 차이를 학습하고, 생성기는 이를 속이도록 최적화된다(**adversarial learning**).
- 보통 **재구성 손실($\ell_1/\ell_2$)**, **perceptual loss**, (필요 시) **mask-weighted loss** 등을 결합하여 안정성과 시각적 품질을 확보한다.

#### 구조적 강점

- **지역 텍스처/경계 복원에 유리**: CNN 기반 지역 수용영역(local receptive field)과 adversarial loss가 결합되어, 타일 단위 입력에서 **선명한 디테일**을 얻기 쉽다.
- **실험 베이스라인 구축 용이**: 구현/학습 파이프라인이 성숙하여 빠른 비교 실험에 적합하다.

#### 구조적 한계

- **픽셀/로컬 패턴 편향**: 손실 설계가 픽셀 수준 대응에 치중되기 쉬워, 넓은 장면 문맥(landscape-level context)이나 물리적 일관성 반영이 제한될 수 있다.
- **학습 불안정 및 아티팩트**: adversarial training 특성상 모드 붕괴, checkerboard/texture artifact가 발생할 수 있어 세심한 튜닝이 필요하다.
- **일반화 이슈**: 지역/계절/관측 조건 변화에 대해 성능 변동이 나타날 수 있다.

### 1.2 Proposed: Copernicus Foundation Model (Copernicus-FM)

#### 모델 구조d

- **ViT(Vision Transformer) 백본** 기반 파운데이션 모델을 가정한다.
- **Dynamic Hypernetworks**로 센서 조건(예: 밴드/파장 정보)에 따라 표현/파라미터를 **동적으로 조건화(conditioning)** 한다.

#### 핵심 구성 요소 (구조적 차별점)

- **Sensor-Aware Embedding**: 센서의 파장/대역폭 정보를 학습에 반영하여, 다양한 센서 조합/채널 구성에서 표현 학습이 가능하도록 한다.
- **Metadata Integration**: 위치(Geolocation), 시간(Time) 등 메타데이터를 Fourier Encoding 등으로 입력에 통합하여, 계절/지역성에 따른 복원 일관성을 강화한다.
- **Patch tokenization**: Transformer 입력은 패치 기반 토큰화를 사용한다. 본 데이터는 타일 크기가 256~512px 범위이므로, 패치 크기(예: $16\times16$) 선택에 따라 토큰 수가 변하며, 학습/추론 시 메모리 예산에 맞게 설정한다.

#### 학습 방식 (self-supervised signal) !이미 되어있어서 이 부분은 빼야할듯

- **Masked Image Modeling (MIM)** 기반 자기지도 사전학습으로 문맥적 표현을 학습한 뒤,
- 다운스트림 태스크(Cloud removal / SAR-to-RGB translation)로 **미세조정(fine-tuning)** 한다.

#### 구조적 강점

- **광역 문맥 이해**: 전역 attention으로 장거리 의존성을 포착하여, 결손 영역 복원 시 **장면 수준 구조 일관성**을 강화할 수 있다.
- **조건/센서 변화에 대한 견고성**: sensor-aware 및 메타데이터 조건화를 통해 관측 조건 변화에 대한 일반화 가능성이 높다.
- **지식 전이(transfer)**: 사전학습 표현을 활용해 비교적 적은 태스크 데이터로도 성능을 확보할 여지가 있다.

---

## 2. Comparison Matrix

| 비교 항목 | GAN (pix2pix/CycleGAN) | Copernicus-FM (Proposed) |
| --- | --- | --- |
| **기본 아키텍처** | CNN (Encoder–Decoder) | ViT (Transformer) |
| **데이터 활용** | 이미지 픽셀 정보 위주 | 이미지 + 메타데이터(위치/시간 등) |
| **학습 방식** | 적대적 학습 + 재구성/마스크 손실 | 자기지도(MIM) 사전학습 + 다운스트림 미세조정 |
| **확장성** | 도메인/조건 변화에 민감 | 전이/일반화에 유리 |
| **복원 관점 강점** | 로컬 디테일 선명도 | 전역 구조/문맥 일관성 |

---

## 3. Dataset: Korea SAR-to-RGB (GEE-filtered)

### 3.1 데이터 저장소 및 파일 구조 (Data Storage)

- **로컬 경로**: `/home/ey/data_2/SARtoRGB/Korea/`
- **파일 형식**: 개별 `.tif` (GeoTIFF, multi-band raster)
- **파일 개수**: 총 **1,232** tiles
- **데이터 타입**: Float32 (연산 효율을 위해 로드 시 `float32` 변환 권장)

### 3.2 이미지 크기 및 해상도 (Shape & Resolution)

- **해상도(Scale)**: **10 m/pixel** (1 pixel = 10 m)
- **텐서 형태**: `(7, H, W)`
  - GEE에서 ROI를 타일링하여 내려받았으므로, 각 타일의 가로/세로는 대략 **256px ~ 512px** 범위로 가정한다.
- **채널 수**: 총 **7 channels** (S2 4개 + S1 2개 + Fmask 1개)

### 3.3 채널별 상세 명세 (Channel Ordering)

`input_channels = 7`로 설정할 때 인덱스별 의미는 다음과 같다.

| Index | Name | Sensor | Description | Range |
| --- | --- | --- | --- | --- |
| 0 | B4 (Red) | S2 | 광학 적색 파장 (665nm) | 0 ~ 10,000 |
| 1 | B3 (Green) | S2 | 광학 녹색 파장 (560nm) | 0 ~ 10,000 |
| 2 | B2 (Blue) | S2 | 광학 청색 파장 (490nm) | 0 ~ 10,000 |
| 3 | B8 (NIR) | S2 | 근적외선 파장 (842nm) | 0 ~ 10,000 |
| 4 | VV | S1 | SAR 수직 편파(구름 투과) | -45 ~ +35 dB |
| 5 | VH | S1 | SAR 교차 편파(구름 투과) | -55 ~ +25 dB |
| 6 | Fmask | - | 구름 확률(Cloud probability) | 0 ~ 100 (%) |

### 3.4 Fmask란 무엇인가? (The Role of Channel 6)

**Fmask(Function of Mask)** 는 위성 영상에서 구름/구름 그림자/눈 등을 식별하는 알고리즘 계열의 출력이다. 본 데이터셋에서는 `s2cloudless` 기반의 **Cloud Probability(0~100)** 값을 채널 6에 저장한다.

- **용도 1 (Data Cleaning)**: 훈련 전, Fmask 채널의 **평균값이 80 이상**인 타일은 “너무 흐려서 유효한 정답(가시 지표면 정보)이 부족한 샘플”로 간주하여 train set에서 제외한다.
- **용도 2 (Inference / Attention Mask)**: 구름 제거 모델링 시, Fmask가 높은 영역을 중심으로 SAR(VV/VH) 정보를 사용해 RGB를 복원하도록 유도하는 **attention / loss mask**로 활용한다.

---

## 4. Developer Guide: Preprocessing & Loading

### 4.1 No-Data 및 Outlier 처리

GEE로 내려받은 래스터에는 `-inf` 또는 `NaN`이 포함될 수 있으므로, 로드 직후 전처리가 필요하다.

```python
import numpy as np

data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
```

### 4.2 채널별 정규화 (Normalization)

센서/단위가 다르므로 채널별로 별도 정규화한다.

- **S2 (0~3)**: `data[:4] / 10000.0`  (0~1 스케일)
- **S1 (4~5)**: `(data[4:6] + 20) / 20`  (dB 단위를 고려한 근사 정규화; clip 범위는 실험으로 확정)
- **Fmask (6)**: `data[6] / 100.0`  (0~1 확률)

### 4.3 PyTorch Dataset 예시

```python
import numpy as np
import torch
from torch.utils.data import Dataset
import rasterio


class KoreaSatelliteDataset(Dataset):
    def __init__(self, file_paths):
        self.file_paths = file_paths

    def __getitem__(self, idx):
        with rasterio.open(self.file_paths[idx]) as src:
            img = src.read().astype(np.float32)  # (7, H, W)

        img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)

        # Channel-wise normalization (example)
        img[:4] = img[:4] / 10000.0
        img[4:6] = (img[4:6] + 20.0) / 20.0
        img[6] = img[6] / 100.0

        return torch.from_numpy(img)

    def __len__(self):
        return len(self.file_paths)
```

---

## 5. Experimental Protocol (Two-Stage)

본 절에서는 Korea 데이터셋(`(7, H, W)`, 10m/pixel) 환경에서 **(1) 단순 이미지 복원(베이스라인 검증)** 과 **(2) 실제 구름 제거(Cloud removal)** 의 2단계 실험을 통해 GAN 계열과 Copernicus-FM을 비교하는 프로토콜을 정의한다.

### 5.1 Stage 1: 단순 이미지 복원 (Baseline 검증)

#### 목적

- **상대적으로 구름이 적은(clean) 타일**에서, SAR 입력이 광학(RGB)과 얼마나 유사한 출력을 생성하는지 확인한다.

#### 데이터 구성 (clean split)

- Fmask 채널을 사용하여 타일 단위로 필터링한다.
  - 예: `mean(Fmask) < 20`을 clean 후보로 사용(임계값은 데이터 분포를 보고 확정).
  - `mean(Fmask) >= 80` 타일은 학습에서 제외(데이터 정리 규칙).

#### 실험 설정

- **입력/출력**: `(VV, VH)` 또는 `(VV, VH, Fmask)` $\rightarrow$ `(R, G, B)` 생성.
- **모델 비교**: GAN(pix2pix/CycleGAN) vs Copernicus-FM(fine-tuned).

#### 평가 지표

- **PSNR / SSIM**: 생성 RGB와 GT RGB의 전반적 유사도.
- **Texture 보존**: 지표면 질감(도로/농지 패턴/수변 경계 등)이 **뭉개지는지(oversmoothing)** 를 확대 시각화로 비교하고, 필요 시 주파수/스펙트럼 기반 보조 지표로 확인한다.

#### 비교 포인트(관찰 가설)

- **GAN**: 시각적으로 그럴듯해도 로컬 손실/판별기 편향으로 인해 **지형 질감이 뭉개지거나 반복 패턴 아티팩트**가 발생할 수 있다.
- **Copernicus-FM**: ViT 전역 문맥 파악으로 **전체 구조 일관성**을 더 잘 유지할 가능성이 있다(단, texture 평활화 여부는 보조 평가로 확인).

### 5.2 Stage 2: 실제 구름 제거 (Cloud Removal)

#### 목적

- 구름이 낀 광학 영상에서 **구름 영역을 마스크로 정의**하고, SAR(VV/VH) 및 주변 문맥을 활용해 **자연스럽게 채우기(inpainting / reconstruction)** 성능을 비교한다.

#### 실험 설정

- **마스크 정의**: 픽셀 단위로 `Fmask >= T`를 구름 영역으로 간주(예: `T=80`, 임계값은 실험으로 확정).
- **비교 모델**
  - **GAN 기반 Inpainting**: `(SAR + Cloudy RGB + mask)` $\rightarrow$ `Clean RGB` 형태로 조건부 생성기를 학습하고, 마스크 영역 중심으로 손실을 가중한다.
  - **Copernicus-FM(MIM) 복원**: MIM 스타일로 마스크 영역을 복원하며, SAR 및(가능 시) 메타데이터 조건화를 함께 사용한다.

#### 평가 지표

- **ROI 기반 PSNR/SSIM/RMSE**: 전체가 아닌 **마스크 영역(구름 영역)** 중심으로 분리 보고(비마스크 영역도 함께 보고 가능).
- **경계 자연스러움**: 마스크 경계(seam) 여부를 정성 비교하고, 가능하면 경계 주변 링(예: 3~5px dilate ring)의 오류를 별도 집계한다.

#### 비교 포인트(관찰 가설)

- **GAN Inpainting**: 경계가 그럴듯해 보이도록 만들 수 있으나, 마스크 경계에서 **불연속/텍스처 불일치**가 나타날 수 있다.
- **Copernicus-FM(MIM)**: 문맥 기반 복원으로 **경계 연속성**과 장면 수준 일관성이 더 자연스러울 가능성이 있으며, ROI/경계 기반 평가로 검증한다.

---

## 6. Dataset Summary (for Codex / Research Log)

- **Total Samples**: 1,232 tiles for tile
- **Local Path**: `/home/ey/data_2/SARtoRGB/Korea/`
- **Input Dimension**: $7 \times H \times W$ (각 타일은 대략 256~512px 범위)
- **Resolution**: 10 m/pixel
- **Channels**: S2(B4,B3,B2,B8) + S1(VV,VH) + Fmask(cloud probability)
- **Key Metadata**: Geolocation(WGS84), Timestamp(2023-08), Sensor fusion(S1+S2)
- **Target Tasks**: Cloud removal / SAR-to-RGB translation (필요 시 land cover 관련 다운스트림 태스크로 확장 가능)
