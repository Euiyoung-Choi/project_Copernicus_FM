# Research: SAR-to-Optical Image Translation for Cloud Removal

## 1. Methodology: Model Comparison

본 연구에서는 Sentinel-1(SAR) 데이터를 활용하여 Sentinel-2(광학) 영상을 복원(Cloud removal / cloud-free optical reconstruction)하기 위해, 전통적인 생성 모델인 **GAN(pix2pix/CycleGAN)** 과 최신 자기지도학습 기반의 **Copernicus-FM**을 비교 분석한다. 특히 **SEN1-2 데이터셋의 $256\times256$ 패치 단위** 입력을 가정하고, 두 계열 모델의 **구조적 차이(architecture-level differences)** 와 그로부터 파생되는 **강점/제약**을 방법론 관점에서 정리한다.

### 1.1 Baseline: Generative Adversarial Networks (GAN)

#### 모델 구조

- **pix2pix(conditional GAN)**: SAR $\rightarrow$ Optical의 **지도학습 paired translation**을 가정한다. 일반적으로 U-Net 계열의 **Encoder–Decoder 생성기**와 PatchGAN 계열 **판별기** 조합을 사용한다.
- **CycleGAN(unpaired translation)**: (SAR, Optical) 쌍이 부족한 경우를 위해 **Cycle-consistency**를 도입하여, SAR $\leftrightarrow$ Optical의 양방향 매핑을 동시에 학습한다.

#### 작동 원리 (학습 신호)

- 생성기(Generator)가 SAR를 입력으로 **가짜 광학 영상**을 생성한다.
- 판별기(Discriminator)가 생성 결과와 실제 광학 영상의 분포 차이를 학습하고, 생성기는 이를 속이도록 최적화된다(**adversarial learning**).
- 보통 **재구성 손실($\ell_1/\ell_2$)**, **perceptual loss**, **cycle loss**(CycleGAN) 등을 함께 사용하여 시각적 유사성과 안정성을 확보한다.

#### 구조적 강점

- **지역적 텍스처/경계 복원에 강함**: CNN 기반의 지역 수용영역(local receptive field)과 adversarial loss가 결합되어, 256×256과 같은 고정 크기 패치에서 **선명한 시각적 디테일**을 얻기 쉽다.
- **구현/학습 파이프라인이 성숙**: SEN1-2와 같은 패치 데이터셋에 대해 빠르게 베이스라인을 구성할 수 있다.

#### 구조적 한계 (SEN1-2, 256×256 관점)

- **픽셀 수준 대응에 치중**: 손실 설계가 픽셀/로컬 패턴에 집중되기 쉬워, SAR–Optical 간의 **물리적 상관관계(산란 특성 vs 반사율)** 나 **광역 문맥(context)** 을 충분히 반영하지 못할 수 있다.
- **도메인/센서 확장성 제한**: 특정 센서/밴드 조합에 맞춰 학습된 가중치가 다른 조건(지역/계절/센서)으로 일반화될 때 성능 저하가 나타날 수 있다.
- **학습 불안정 및 아티팩트**: adversarial training 특성상 모드 붕괴/격자 아티팩트 등 품질 이슈가 발생할 수 있으며, 이를 완화하기 위해 세심한 튜닝이 필요하다.

### 1.2 Proposed: Copernicus Foundation Model (Copernicus-FM)

#### 모델 구조

- **ViT(Vision Transformer) 백본**을 기반으로, 다양한 Copernicus 센서/태스크를 포괄하도록 설계된 **파운데이션 모델**로 가정한다.
- **Dynamic Hypernetworks**를 결합하여, 입력 센서 특성(예: 밴드/파장 정보)에 따라 네트워크의 일부 파라미터/표현을 **동적으로 조건화(conditioning)** 하는 구조를 사용한다.

#### 핵심 구성 요소 (구조적 차별점)

- **Sensor-Aware Embedding**: 하이퍼네트워크를 통해 센서의 **파장(Wavelength)** 및 **대역폭(Bandwidth)** 정보를 학습 과정에 반영하여, 서로 다른 센서/밴드 조합에서도 표현 학습이 가능하도록 한다.
- **Metadata Integration**: 위치(Geolocation), 시간(Time), 면적(Area) 등 메타데이터를 **Fourier Encoding** 등으로 임베딩하여 입력에 통합함으로써, **계절/지역성**에 따른 광학 복원의 일관성을 강화한다.
- **Dynamic Patching**: FlexiViT 계열 아이디어를 적용해 입력 해상도/패치 설정에 유연하게 대응한다. SEN1-2의 $256\times256$ 입력에서는 보편적으로 $16\times16$ 패치 토큰화가 자연스럽다(총 $16\times16=256$ 토큰).

#### 학습 방식 (self-supervised signal)

- **Masked Image Modeling (MIM)** 기반 자기지도 사전학습을 통해, 대규모 위성 데이터에서 **문맥적 표현(Contextual Representation)** 을 먼저 학습하고,
- 이후 SAR-to-Optical(Cloud removal) 다운스트림 태스크로 **미세조정(fine-tuning)** 하여 복원 성능을 확보한다.

#### 구조적 강점 (SEN1-2, 256×256 관점)

- **광역 문맥 이해**: Transformer의 전역 attention은 패치 기반 입력에서 **장거리 의존성**을 포착하기 유리하여, 구름 제거/복원 시 **대규모 구조(토지 피복 패턴, 장면 구성)** 의 일관성에 강점을 가진다.
- **센서/조건 일반화**: sensor-aware 및 메타데이터 조건화를 통해, 지역/계절/센서 조건 변화에 대해 더 견고한 복원이 가능하다.
- **지식 전이(transfer)**: 기초 모델의 사전학습 표현을 활용하여, 제한된 paired 데이터에서도 성능을 확보할 여지가 있다.

---

## 2. Comparison Matrix

| 비교 항목 | GAN (pix2pix/CycleGAN) | Copernicus-FM (Proposed) |
| --- | --- | --- |
| **기본 아키텍처** | CNN (Encoder–Decoder) | ViT (Transformer) |
| **데이터 활용** | 이미지 픽셀 정보 위주 | 이미지 + 메타데이터(위치/시간 등) |
| **학습 방식** | 적대적 학습(Adversarial) + 재구성 손실 | 자기지도(MIM) 사전학습 + 다운스트림 미세조정 |
| **확장성** | 특정 센서/도메인에 맞춤 학습 경향 | 센서/태스크 전반으로 지식 전이 가능 |
| **복원 관점 강점** | 시각적 선명도/디테일 확보 용이 | 문맥적/물리적 일관성 강화 가능 |

---

## 3. Implementation Note for SEN1-2 Dataset ($256\\times256$)

### 3.1 Input Resolution & Patching

- **Input resolution**: SEN1-2의 $256\times256$ 패치는 ViT 계열에서 대표적으로 **$16\times16$ 패치**로 토큰화하기 적합하다(토큰 수 256).
- **Practical note**: GAN은 컨볼루션으로 고정 해상도 처리가 자연스럽고, Copernicus-FM은 dynamic patching을 통해 **해상도/패치 설정 변화**를 흡수할 수 있다.

### 3.2 Data Augmentation

- 학습 시 **Random Resized Crop**, **Horizontal Flipping** 등을 적용해 일반화 성능을 높인다.
- SAR/Optical 정합이 중요한 paired 설정에서는, **동일한 공간 변환을 입력/타깃에 함께 적용**하여 정합(align)을 유지한다.

### 3.3 Evaluation Metric

- 복원 품질 평가는 Copernicus-Bench의 전처리/평가 규약(Cloud-S2 task)을 따르는 것을 가정한다.
- 주요 지표로 **RMSE**를 사용하며, 태스크 정의에 따라 세그멘테이션/클래스 기반 평가가 포함될 경우 **mIoU**를 함께 보고한다.

