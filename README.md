# CHGS

Co-Speech Holistic 3D Motion Generation with Style from Video

## TODO

- [x] Gesture generation w/o style encoder
- [ ] Gesture generation w/ style encoder

## Setup

### 1. Install environment

```bash
conda create -n gesture python=3.8 -y
conda activate gesture
pip install -r requirements.txt
```

If the PyTorch version in `requirements.txt` does not match your CUDA version, install the correct PyTorch build first, then run:

```bash
pip install -r requirements.txt --no-deps
```

### 2. Download SMPL-X

Download the official SMPL-X model files from:

- [SMPL-X official website](https://smpl-x.is.tue.mpg.de/)

Place them under:

```text
smplx_models/
└── smplx/
    └── SMPLX_NEUTRAL_2020.npz
```


### 4. Download project checkpoints

You need to prepare the following project files:

- Diffusion checkpoint:
  - `experiments/DPT/<exp_name>/checkpoints/iter_<iteration>.pt`
- RVQ-VAE checkpoints:
  - `ckpt/beatx2_rvqvae/RVQVAE_upper/net_300000_1.pth`
  - `ckpt/beatx2_rvqvae/RVQVAE_hands/net_300000_1.pth`
  - `ckpt/beatx2_rvqvae/RVQVAE_lower_trans/net_300000_1.pth`
- Normalization stats:
  - `mean_std/beatx_2_330_mean.npy`
  - `mean_std/beatx_2_330_std.npy`
  - `mean_std/beatx_2_trans_mean.npy`
  - `mean_std/beatx_2_trans_std.npy`

Project checkpoint download link: https://drive.google.com/file/d/1xkseonnxHTTklSNCFj2LBhokYYqeTIQW/view?usp=sharing


## Inference

### Download checkpoints

Before inference, download or place:

- the diffusion checkpoint under `experiments/DPT/...`
- the RVQ-VAE checkpoints under `ckpt/beatx2_rvqvae/...`

### Generate a gesture video

```bash
python demo_vq.py \
  --exp_name gesture \
  --iter 155000 \
  -a /path/to/wave16k/example.wav \
  -o output.mp4
```

If the reference `.npz` directory or SMPL-X directory is not in the default location:

```bash
python demo_vq.py \
  --exp_name gesture \
  --iter 155000 \
  -a /path/to/audio.wav \
  -o output.mp4 \
  --gt_npz_dir /path/to/smplxflame_30 \
  --smplx_model_dir /path/to/smplx_models
```





