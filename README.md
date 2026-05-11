# Conformal Prediction with ECG

Project for ECG signal classification and segmentation using deep learning models with conformal prediction.

## Project Structure

```
Conformal Prediction with ECG/
├── config/                          # Configuration files
│   ├── config.yaml                  # Main configuration
│   ├── classification/              # Classification dataset configs
│   │   ├── ptbdb_clean_mihealthy.yaml      # PTBDB clean data (MI vs Healthy)
│   │   ├── ptbdb_clean_subtype.yaml        # PTBDB clean data (Subtype)
│   │   ├── ptbdb_augmented_mihealthy.yaml  # PTBDB augmented data (MI vs Healthy)
│   │   ├── ptbdb_augmented_subtype.yaml    # PTBDB augmented data (Subtype)
│   │   ├── ptbxl_clean_mihealthy.yaml      # PTB-XL clean data (MI vs Healthy)
│   │   ├── ptbxl_clean_subtype.yaml        # PTB-XL clean data (Subtype)
│   │   ├── ptbxl_augmented_mihealthy.yaml  # PTB-XL augmented data (MI vs Healthy)
│   │   └── ptbxl_augmented_subtype.yaml    # PTB-XL augmented data (Subtype)
│   ├── segmentation/              # Segmentation dataset configs
│   │   └── default.yaml            # Default segmentation config (LUDB)
│   ├── model/                       # Model configurations
│   │   ├── ecgtransform.yaml        # ECG Transformer model config
│   │   ├── mcdann.yaml              # MCDANN model config
│   │   ├── mffcnn.yaml              # MFF-CNN model config
│   │   └── unet3pcgm.yaml           # UNet 3+ CGM model config
│   ├── trainer/                     # Trainer configurations
│   │   └── default.yaml            # Default trainer config
│   └── hydra/                       # Hydra configurations
│       └── default.yaml            # Default Hydra config
├── data/                            # Data directory
├── src/                             # Source code
│   ├── conformal_prediction/        # Conformal prediction modules
│   ├── data/                        # Data processing modules
│   │   ├── classification.py        # Classification datasets
│   │   ├── get_data.py              # Data loading utilities
│   │   ├── preprocess.py            # Data preprocessing
│   │   └── segmentation.py         # Segmentation datasets
│   ├── model/                       # Model implementations
│   │   ├── ECGTransForm.py          # ECG Transformer
│   │   ├── MCDANN.py                # MCDANN model
│   │   ├── MFF_CNN.py               # Multi-scale Feature Fusion CNN
│   │   └── UNet1D.py                # 1D UNet variants
│   └── utils/                       # Utility modules
│       ├── augment.py               # Data augmentation
│       ├── denoise.py               # Signal denoising
│       ├── loss.py                  # Loss functions
│       └── transform.py             # Data transformations
├── main.py                          # Main entry point
├── train.py                         # Training script
├── evaluate.py                      # Evaluation script
├── patient_fold.csv                 # Patient fold assignments
├── ptb_fold.csv                     # PTB fold assignments
├── pyproject.toml                   # Project configuration
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

## Training Parameters

### Main Configuration (config/config.yaml)

| Parameter | Description | Default |
|-----------|-------------|---------|
| `task` | Task type: "classification" or "segmentation" | null |
| `dataset` | Dataset for classification: "ptbdb" or "ptbxl" | null |
| `data_mode` | Data mode for classification: "clean" or "augmented" | null |
| `data_type` | Data type for classification: "mihealthy" or "subtype" | null |
| `segmentation_model` | Segmentation model: "unet3pcgm" | unet3pcgm |
| `seed` | Random seed for reproducibility | 42 |
| `resume_from_checkpoint` | Path to checkpoint file for resuming training | null |

### Trainer Configuration (config/trainer/default.yaml)

| Parameter | Description | Default |
|-----------|-------------|---------|
| `max_epochs` | Maximum number of training epochs | 100 |
| `accelerator` | Accelerator type ("cpu", "gpu", "auto") | "auto" |
| `devices` | Number of devices to use | "auto" |
| `precision` | Training precision ("32", "16-mixed", "bf16") | "16-mixed" |
| `log_every_n_steps` | Log frequency in steps | 10 |
| `gradient_clip_val` | Gradient clipping value | 1.0 |
| `accumulate_grad_batches` | Gradient accumulation batches | 1 |

### Model Configurations

#### ECG Transformer (config/model/ecgtransform.yaml)

| Parameter | Description | Default |
|-----------|-------------|---------|
| `lr` | Learning rate | 0.001 |

#### MCDANN (config/model/mcdann.yaml)

| Parameter | Description | Default |
|-----------|-------------|---------|
| `lr` | Learning rate | 0.001 |
| `weight_decay` | L2 regularization | 0.0001 |

#### UNet 3+ CGM (config/model/unet3pcgm.yaml)

| Parameter | Description | Default |
|-----------|-------------|---------|
| `n_channels` | Number of input channels | 4 |
| `lr` | Learning rate | 0.001 |
| `focal_gamma` | Focal loss gamma parameter | 2.0 |
| `mask` | Mask configuration | true |

### Data Configurations

Classification dataset configs follow the naming convention: `{dataset}_{data_mode}_{data_type}.yaml`

#### PTBDB Classification (config/classification/ptbdb_*.yaml)

| Parameter | Description | Default |
|-----------|-------------|---------|
| `metadata_file` | Path to metadata file | data/ptb_fold.csv |
| `train_folds` | Training fold indices | [1, 2, 3, 4, 5, 6, 7, 8] |
| `test_folds` | Test fold indices | [9, 10] |
| `batch_size` | Batch size | 32 |
| `num_workers` | Number of data loading workers | 4 |
| `split_ratio` | Train/validation split ratio | 0.9 |
| `sample_before` | Samples before R-peak | 198 |
| `sample_after` | Samples after R-peak | 200 |
| `num_classes` | Number of classes | 3 (mihealthy) or 12 (subtype) |
| `use_cleaned_data` | Whether to use cleaned data | false (clean) / true (augmented) |

#### PTB-XL Classification (config/classification/ptbxl_*.yaml)

| Parameter | Description | Default |
|-----------|-------------|---------|
| `metadata_file` | Path to metadata file | data/ptb_fold.csv |
| `train_folds` | Training fold indices | [1, 2, 3, 4, 5, 6, 7, 8] |
| `test_folds` | Test fold indices | [9, 10] |
| `batch_size` | Batch size | 32 |
| `num_workers` | Number of data loading workers | 4 |
| `split_ratio` | Train/validation split ratio | 0.9 |
| `sample_before` | Samples before R-peak | 198 |
| `sample_after` | Samples after R-peak | 200 |
| `num_classes` | Number of classes | 3 (mihealthy) or 12 (subtype) |
| `use_cleaned_data` | Whether to use cleaned data | false (clean) / true (augmented) |

#### LUDB Segmentation (config/segmentation/default.yaml)

| Parameter | Description | Default |
|-----------|-------------|---------|
| `name` | Dataset name | ludb |
| `metadata_file` | Path to metadata file | data/ludb_metadata.csv |
| `split_ratio` | Train/validation split ratio | 0.8 |
| `batch_size` | Batch size | 16 |
| `num_workers` | Number of data loading workers | 4 |
| `seed` | Random seed for data split | 42 |

## Installation

This project uses [uv](https://github.com/astral-sh/uv) for fast Python package management.

### Prerequisites

- Python 3.10 or higher
- [uv](https://github.com/astral-sh/uv) installed

### Setup

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv sync

# Activate virtual environment (optional, uv run handles this automatically)
source .venv/bin/activate  # On Linux/Mac
# or
.venv\Scripts\activate  # On Windows
```

## Usage

### Download data

```bash
uv run download_data.py
```

### Training

#### Classification

Classification training requires three parameters:
- `dataset`: "ptbdb" or "ptbxl"
- `data_mode`: "clean" or "augmented"
- `data_type`: "mihealthy" (3 classes) or "subtype" (12 classes)

**PTBDB Dataset - Clean Data (MI vs Healthy):**
```bash
uv run train.py task=classification dataset=ptbdb data_mode=clean data_type=mihealthy model=mcdann
```

**PTBDB Dataset - Clean Data (Subtype):**
```bash
uv run train.py task=classification dataset=ptbdb data_mode=clean data_type=subtype model=mcdann
```

**PTBDB Dataset - Augmented Data (MI vs Healthy):**
```bash
uv run train.py task=classification dataset=ptbdb data_mode=augmented data_type=mihealthy model=mcdann
```

**PTBDB Dataset - Augmented Data (Subtype):**
```bash
uv run train.py task=classification dataset=ptbdb data_mode=augmented data_type=subtype model=mcdann
```

**PTB-XL Dataset - Clean Data (MI vs Healthy):**
```bash
uv run train.py task=classification dataset=ptbxl data_mode=clean data_type=mihealthy model=mcdann
```

**PTB-XL Dataset - Clean Data (Subtype):**
```bash
uv run train.py task=classification dataset=ptbxl data_mode=clean data_type=subtype model=mcdann
```

**PTB-XL Dataset - Augmented Data (MI vs Healthy):**
```bash
uv run train.py task=classification dataset=ptbxl data_mode=augmented data_type=mihealthy model=mcdann
```

**PTB-XL Dataset - Augmented Data (Subtype):**
```bash
uv run train.py task=classification dataset=ptbxl data_mode=augmented data_type=subtype model=mcdann
```

#### Segmentation

```bash
uv run train.py task=segmentation model=unet3pcgm
```

### Training Modes

**Classification tasks support two training modes:**

1. **`clean`**: Train only with cleaned data
   - Uses dataset config with `_clean` suffix
   - No augmentation applied
   - Best for baseline comparison

2. **`augmented`**: Train with normal data + cleaned data + augmentation
   - Uses dataset config with `_augmented` suffix
   - Applies data augmentation (Gaussian noise, blur, amplitude scaling, powerline noise)
   - Better for improving model robustness

**Classification data types:**

1. **`mihealthy`**: Binary classification (3 classes)
   - MI vs Healthy classification
   - 3 output classes

2. **`subtype`**: Multi-class classification (12 classes)
   - Detailed ECG subtype classification
   - 12 output classes

### Resume Training

```bash
# Resume from checkpoint
uv run python train.py task=classification dataset=ptbdb data_mode=clean data_type=mihealthy model=ecg_transform resume_from_checkpoint=outputs/classification/ecg_transform/ptbdb_clean/epoch=10-val_loss=0.1234.ckpt

# Resume from last checkpoint
uv run python train.py task=classification dataset=ptbdb data_mode=clean data_type=mihealthy model=ecg_transform resume_from_checkpoint=last
```

### Available Models

**Classification:**
- `ecg_transform` - ECG Transformer
- `mcdann` - Multi-Channel Domain Adaptation Neural Network

**Segmentation:**
- `unet3pcgm` - UNet 3+ with CGM (default)

### Available Datasets

**Classification:**
- `ptbdb` - Physikalisch-Technische Bundesanstalt database
- `ptbxl` - PTB-XL database

**Segmentation:**
- `ludb` - Lobachevsky University Database

### Logging

Training uses WandB for experiment tracking:
- Classification project: `demo_ecg_classification`
- Segmentation project: `demo_ecg_segmentation`
- Logs are saved to `logs/{task}/{data_type}/{model}_{dataset}_{data_mode}/`
- Checkpoints are saved to `outputs/{task}/{model_name}/{dataset}_{data_mode}/`

## Dependencies

Dependencies are managed via `pyproject.toml` and `uv.lock`. Key dependencies include:

- PyTorch
- PyTorch Lightning
- Hydra
- NumPy
- Pandas
- Scikit-learn
- WFDB (PhysioNet)
- ecgmentations (ECG data augmentation)
- Weights & Biases (WandB)

To add new dependencies:
```bash
uv add <package_name>
```
