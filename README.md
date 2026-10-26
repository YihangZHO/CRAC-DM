# CRAC-DM

**Class Relation-Aware Categorical Diffusion Model for Surgical Scene Segmentation**

Yihang Zhou, Chi Xu, Zaid Awad, and Stamatia Giannarou  
Imperial College London

## Overview

CRAC-DM performs pixel-wise segmentation of surgical images using conditional categorical diffusion. The proposed method combines:

- **Class relation-aware diffusion:** semantic class relationships guide label corruption.
- **S2D sampling:** step-skipping denoising reduces inference from 250 to 10 steps.
- **Confidence-adaptive TTA:** entropy-weighted ROI refinement improves uncertain predictions.

## Results

Results reported in the accompanying manuscript:

| Method | CholecSeg8k mIoU (%) | EndoVis18 mIoU (%) |
|---|---:|---:|
| CCDM | 81.81 | 47.28 |
| **CRAC-DM** | **84.79** | **48.74** |

The manuscript reports **30 FPS** with 10 reverse steps and TTA on an NVIDIA RTX A6000.

## Checkpoints

Pretrained checkpoints will be uploaded.

## Setup

Install a compatible PyTorch / torchvision build, then install the project dependencies:

```bash
pip install -r requirements.txt
```

Before installation, correct the historical `ignite` requirement to `pytorch-ignite` and include `cityscapesscripts` and `scikit-image`. DINO pretrained weights are downloaded through `torch.hub`.

## Data

Set `BASE_PATH` in `datasets/cholec.py`. Place RGB images and corresponding single-channel class-ID masks in `train/image`, `train/label`, `test/image`, and `test/label`. Cholec masks use **13 classes, indexed 0-12**. Image and mask filenames must sort into matching order.

## Training

Set the data path and output directory in the configuration. Set `load_from: null` when training from scratch.

```bash
python ddpm_train.py params_train_cdm_dino.yml
```

## Evaluation

Set `load_from` in the evaluation configuration to your trained checkpoint, then run:

```bash
python ddpm_eval.py params_eval_cdm_dino_256x512.yml
```

Predicted masks, color visualizations, and `evaluation_results.json` are saved relative to the checkpoint directory. Reported metrics include mIoU, per-class IoU, and pixel accuracy.

A more complete and organized code release will be available soon.

## Citation

```bibtex
@unpublished{zhou_crac_dm,
  title  = {{CRAC-DM}: Class Relation-Aware Categorical Diffusion Model for Surgical Scene Segmentation},
  author = {Zhou, Yihang and Xu, Chi and Awad, Zaid and Giannarou, Stamatia},
  note   = {Manuscript}
}
```

## Acknowledgements

Built upon **CCDM** (Zbinden et al., ICCV 2023), DINO, and an OpenAI-derived attention U-Net. Please retain upstream attribution and licenses.

**Contact:** [yihang.zhou23@imperial.ac.uk](mailto:yihang.zhou23@imperial.ac.uk)
