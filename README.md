# :raised_hand: [Boostcamp-AI-Tech-Level2] HiBoostCamp :raised_hand:

## Contributors ?

<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
[![All Contributors](https://img.shields.io/badge/all_contributors-5-orange.svg?style=flat-square)](#contributors-)
<!-- ALL-CONTRIBUTORS-BADGE:END -->
<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tr>
    <td align="center"><a href="https://github.com/ejrtks1020"><img src="https://github.com/ejrtks1020.png" width="100px;" alt=""/><br /><sub><b>강나훈</b></sub></a><br /><a href="https://github.com/ejrtks1020" title="Code"></td>
    <td align="center"><a href="https://github.com/lijm1358"><img src="https://github.com/lijm1358.png" width="100px;" alt=""/><br /><sub><b>이종목</b></sub></a><br /><a href="https://github.com/lijm1358" title="Code"></td>
    <td align="center"><a href="https://github.com/fneaplle"><img src="https://github.com/fneaplle.png" width="100px;" alt=""/><br /><sub><b>김희상</b></sub></a><br /><a href="https://github.com/fneaplle" title="Code"></td>
    <td align="center"><a href="https://github.com/KimGeunUk"><img src="https://github.com/KimGeunUk.png" width="100px;" alt=""/><br /><sub><b>김근욱</b></sub></a><br /><a href="https://github.com/KimGeunUk" title="Code"></td>
    <td align="center"><a href="https://github.com/jshye"><img src="https://github.com/jshye.png" width="100px;" alt=""/><br /><sub><b>정성혜</b></sub></a><br /><a href="https://github.com/jshye" title="Code"></td>    
  </tr>
</table>
<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->
<!-- ALL-CONTRIBUTORS-LIST:END -->

## Project Overview

뼈는 우리 몸의 구조와 기능에 중요한 영향을 미치기 때문에, 정확한 뼈 분할은 의료 진단 및 치료 계획을 개발하는 데 필수적입니다.
Bone Segmentation은 인공지능 분야에서 중요한 응용 분야 중 하나로, 특히, 딥러닝 기술을 이용한 뼈 Segmentation은 많은 연구가 이루어지고 있으며, 다양한 목적으로 도움을 줄 수 있습니다.
따라서 우리는 질병 진단, 수술 계획, 의료 장비 제작, 의료 교육 등에 사용될 수 있도록 우수한 성능의 모델을 개발합니다.

## Experiments

### 1. EDA

<p align="center"><img src="https://github.com/boostcampaitech5/level2_cv_semanticsegmentation-cv-08/assets/74355042/471470c5-c320-49d6-b2b6-236caf264a36" width="500" height="500"/></p>

<br>

<p align="center"><a href="https://kr.freepik.com/free-vector/bone-inside-the-human-hand_22725580.htm#query=%EC%86%90%EB%AA%A9%20%EB%BC%88&position=0&from_view=keyword&track=ais">작가 brgfx</a> 출처 Freepik</p>

<br>

- 이미지 크기 : 2048 * 2048
- 이미지 픽셀 평균 값: 0.1239
- 이미지 픽셀 표준 편차 값 : 0.1683
- 전체 학습 이미지 수 : 800장 (400 Subjects * 2 hands)
- 전체 테스트 이미지 수 : 300장
- 클래스 수 : 29
- 나이, 성별, 키, 몸무게이 따른 분포
  ![seg2](https://github.com/boostcampaitech5/level2_cv_semanticsegmentation-cv-08/assets/74355042/9fffd76d-0e44-468c-86ef-294ce0e710db)
- 손목이 꺾인 데이터 : train에는 약 10%, test에는 약 50%
- 기타 : 네일아트, 반지, 철심, 골절 데이터 소량
- Miss Label 데이터 소량

### 2. Training 속도 향상
1 Epoch (Train + Validation) 학습 시간
- 기본 : 300.842s
- json to numpy (label 데이터) : 240.4s
- pickle format : 167.231s
- hdf5 format : 147.998s

### 3. Model Finding
Encoder는 efficientnet-b0로 고정하였다.
| Model | Validation Dice Score | +/-(%) |
| :------: | :------: | :------: |
| UNet | 0.9495 | |
| UNet++ | 0.9508 | +0.13 |
| DeepLabV3 | 0.9326 | -1.69 |
| DeepLabV3+ | 0.9347 | -1.48 |
| MAnet | 0.949 | -0.05 |
| Linknet | 0.9289 | -2.04 |
| FPN | 0.9382 | -1.13 |
| PSPNet | 0.9256 | -2.39 |
| PAN | 0.9252 | -2.43 |

### 4. Encoder Finding
Model은 UNet++로 고정하였다.
| Encoder | Resize | Validation Dice Score | +/-(%) |
| :------:| :------: | :------: | :------: |
| Resnet18 | 512 | 0.9519 | |
| Resnet34 | 512 | 0.954 | +0.21 |
| Resnet50 | 512 | 0.9542 | +0.23 |
| Resnet101 | 512 | 0.9508 | -0.11 |
| Resnet152 | 512 | 0.9547 | +0.28 |
| resnext50_32x4d | 512 | 0.9526 | +0.07 |
| resnext101_32x8d | 512 | 0.9564 | +0.45 |
| resnext101_32x16d | 512 | 0.9551 | +0.32 |
| resnext101_32x8d | 1024 | 0.9725 | +2.06 |
| Swin-L | 1024 | 0.9715 | +1.96 |

### 5. Data Preprocessing
1. EDA에서 발견한 Noise 데이터 -> 제거
2. 골격과 신체부분에 대한 경계가 모호 -> Contrast

### 6. Data Augmentation
| Augmentation | Validation Dice Score | +/-(%) |
| :------: | :------: | :------: |
| None | 0.961484 | |
| Normalize | 0.961027 | -0.047 |
| RandomBrightnessContrast | 0.956270 | -0.52 |
| Rotate  | 0.962997 | +0.157 |
| Horizontal Flip | 0.962775 | +0.134 |
| RandomShiftScale | 0.962584 | +0.11 |
| Copy Paste | 0.962584 | +0.114 |
| Rotate + Horizontal Flip | 0.963012 | +0.158 |
| Rotate + Horizontal Flip + Copy Paste | 0.963026 | +0.16 |

### 7. Loss
- MS-SSIM Loss -> loss 감소 x
- Combined Loss (Binary Cross Entropy Loss + Dice Loss) -> 0.005 증가

## Change Log

**`2023-06-07`**: 
- create baseline (pytorch & smp)
- Add pickle dataset
- Add webhook

**`2023-06-09`**: 
- Refactoring
- Integration modules
- Add fp16, wandb logging, early stopping
- Add Rotate, Horizontal Flip Augmentation
- Add Github Actions for black formatting

**`2023-06-10`**: 
- Add train resume, save last checkpoint
- Add model modules

**`2023-06-11`**: 
- Fix loss calculation

**`2023-06-12`**: 
- Add Stratified Group KFold
- Fix loss calculation by GPU
- Add json to numpy converter for fast label load
- Add streamlit dashborad for inference

**`2023-06-13`**: 
- Add scheduler
- Fix wandb logging
- Add streamlit dashborad for validation

**`2023-06-14`**: 
- Add hdf5 dataset

**`2023-06-15`**: 
- Refactoring
- Add Adjust Contrast Augmentation
- Add HRNet

**`2023-06-16`**: 
- Add UNet3+
- Fix loss calculation

**`2023-06-17`**: 
- Add HRNet_OCR, Mask2Former

<br />

----
<br />
