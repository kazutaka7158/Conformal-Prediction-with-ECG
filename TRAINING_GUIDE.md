# Training Configuration Guide

## Config Structure

```
config/
├── config.yaml                    # Main config file
├── classification/                # Classification dataset configs
│   ├── ptbdb_clean.yaml          # PTBDB clean data
│   ├── ptbdb_augmented.yaml      # PTBDB augmented data
│   ├── ptbxl_clean.yaml          # PTB-XL clean data
│   └── ptbxl_augmented.yaml      # PTB-XL augmented data
├── segmentation/                  # Segmentation dataset configs
│   └── default.yaml              # Default segmentation config
├── model/                         # Model configs
│   ├── mff_cnn.yaml              # MFF-CNN model
│   ├── ecg_transform.yaml        # ECG-Transform model
│   ├── mcdann.yaml               # MCDANN model
│   └── unet_3p_cgm.yaml          # UNet-3P-CGM model
└── trainer/                       # Training parameters
    └── default.yaml              # Default trainer config
```

## Usage Examples

### Classification

#### PTBDB Dataset - Clean Data - MIHealthy
```bash
python train.py task=classification dataset=ptbdb data_mode=clean data_type=mihealthy model=mcdann
```

#### PTBDB Dataset - Augmented Data
```bash
python train.py task=classification dataset=ptbdb data_mode=augmented model=mcdann
```

#### PTB-XL Dataset - Clean Data
```bash
python train.py task=classification dataset=ptbxl data_mode=clean model=mcdann
```

#### PTB-XL Dataset - Augmented Data
```bash
python train.py task=classification dataset=ptbxl data_mode=augmented model=mcdann
```

### Segmentation

```bash
python train.py task=segmentation
```

Note: Segmentation uses the `unet3pcgm` model by default.

## Available Models

### Classification Models
- `mff_cnn`: Multi-scale Feature Fusion CNN
- `ecg_transform`: ECG Transformer
- `mcdann`: Multi-Channel Domain Adaptation Neural Network

### Segmentation Models
- `unet3pcgm`: UNet 3+ CGM 1D (default)

## Training Parameters

The training parameters are defined in `config/trainer/default.yaml`:

- `max_epochs`: Maximum number of training epochs (default: 100)
- `accelerator`: Accelerator type (default: auto)
- `devices`: Number of devices (default: auto)
- `precision`: Precision mode (default: 16-mixed)
- `log_every_n_steps`: Logging frequency (default: 10)
- `gradient_clip_val`: Gradient clipping value (default: 1.0)
- `accumulate_grad_batches`: Gradient accumulation (default: 1)

## Resume Training

To resume training from a checkpoint:

```bash
python train.py task=classification dataset=ptbdb data_mode=clean model=mcdann resume_from_checkpoint=checkpoints/epoch=10-val_loss=0.1234.ckpt
```

Or resume from the last checkpoint:

```bash
python train.py task=classification dataset=ptbdb data_mode=clean model=mcdann resume_from_checkpoint=last
```

For segmentation:

```bash
python train.py task=segmentation resume_from_checkpoint=checkpoints/epoch=10-val_loss=0.1234.ckpt
```
