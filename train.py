import random
from pathlib import Path

import ecgmentations as E
import lightning as L
import numpy as np
import torch
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger

import hydra
from omegaconf import DictConfig, OmegaConf

from src.data.classification import (
    MIHealthyDataset,
    ECGSubtypeDataset,
    ECGClassificationDataModule,
)
from src.data.segmentation import ECGSegmentationDataModule
from src.model.MCDANN import MCDANN
from src.model.ECGTransForm import ECGTransForm
from src.model.UNet1D import ECGUNet3pCGM


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_classification_model(model_name: str, model_cfg: DictConfig, num_classes: int) -> L.LightningModule:
    """Get classification model based on model name."""
    if model_name == "ecg_transform":
        return ECGTransForm(
            num_classes=num_classes,
            lr=model_cfg.lr,
        )
    elif model_name == "mcdann":
        return MCDANN(
            num_classes=num_classes,
            lr=model_cfg.lr,
            # weight_decay=model_cfg.weight_decay,
        )
    else:
        raise ValueError(f"Unknown classification model: {model_name}")


def get_segmentation_model(model_name: str,
                           model_cfg: DictConfig) -> L.LightningModule:
    """Get segmentation model based on model name."""
    if model_name == "unet3pcgm":
        return ECGUNet3pCGM(
            # n_channels=model_cfg.n_channels,
            lr=model_cfg.lr,
            focal_gamma=model_cfg.focal_gamma,
            # mask=model_cfg.mask,
        )
    else:
        raise ValueError(f"Unknown segmentation model: {model_name}")


def get_classification_dataset(dataset_name: str,
                               data_cfg: DictConfig,
                               data_mode: str,
                               data_type: str) -> L.LightningDataModule:
    """Get classification dataset based on dataset name and data mode."""
    # Determine transform based on data mode
    # if data_mode == "clean":
    #     transform = None  # No augmentation for clean mode
    if data_mode == "augmented":
        aug_cfg = data_cfg.get("augmentation", {})
        transform = E.Sequential([
            E.GaussNoise(
                p=aug_cfg.get("gauss_noise", {}).get("p", 0.2),
                mean=aug_cfg.get("gauss_noise", {}).get("mean", 0),
                variance=aug_cfg.get("gauss_noise", {}).get("variance", 0.0001),
            ),
            E.GaussBlur(p=aug_cfg.get("gauss_blur", {}).get("p", 0.2)),
            E.AmplitudeScale(p=aug_cfg.get("amplitude_scale", {}).get("p", 0.2)),
            E.PowerlineNoise(
                p=aug_cfg.get("powerline_noise", {}).get("p", 0.2),
                ecg_frequency=aug_cfg.get("powerline_noise", {}).get("ecg_frequency", 500),
                powerline_frequency=aug_cfg.get("powerline_noise", {}).get("powerline_frequency", 60),
                amplitude_limit=aug_cfg.get("powerline_noise", {}).get("amplitude_limit", 0.333),
            ),
        ])
    else:
        transform = None

    # Determine dataset class based on dataset name
    if data_type == "mihealthy":
        dataset_class = MIHealthyDataset
    elif data_type == "subtype":
        dataset_class = ECGSubtypeDataset
    else:
        raise ValueError(f"Unknown classification dataset: {dataset_name}")

    return ECGClassificationDataModule(
        dataset_class=dataset_class,
        metadata_file=data_cfg.metadata_file,
        # train_folds=data_cfg.train_folds,
        # test_folds=data_cfg.test_folds,
        batch_size=data_cfg.batch_size,
        num_workers=data_cfg.num_workers,
        split_ratio=data_cfg.split_ratio,
        sample_before=data_cfg.sample_before,
        sample_after=data_cfg.sample_after,
        use_cleaned_data=data_cfg.get("use_cleaned_data", False),
        transform=transform,
    )


def get_segmentation_dataset(dataset_name: str, data_cfg: DictConfig) -> L.LightningDataModule:
    """Get segmentation dataset based on dataset name."""
    if dataset_name == "ludb":
        return ECGSegmentationDataModule(
            metadata_file=data_cfg.metadata_file,
            split_ratio=data_cfg.split_ratio,
            batch_size=data_cfg.batch_size,
            num_workers=data_cfg.num_workers,
            seed=data_cfg.seed,
        )
    else:
        raise ValueError(f"Unknown segmentation dataset: {dataset_name}")


@hydra.main(config_path="config", config_name="config", version_base="1.3")
def train(cfg: DictConfig) -> None:
    """Main training function with Hydra config."""
    print(OmegaConf.to_yaml(cfg))

    # Set seed
    set_seed(cfg.seed)

    # Validate task type
    task = cfg.task
    if task not in ["classification", "segmentation"]:
        raise ValueError(f"Task must be 'classification' or 'segmentation', got: {task}")

    # Get model and trainer configs
    model_cfg = cfg.model
    trainer_cfg = cfg.trainer

    # Load data config based on task
    if task == "classification":
        dataset = cfg.dataset
        data_mode = cfg.data_mode
        data_type = cfg.data_type

        if dataset is None:
            raise ValueError("dataset must be specified for classification task (use 'ptbdb' or 'ptbxl')")
        if data_mode is None:
            raise ValueError("data_mode must be specified for classification task (use 'clean' or 'augmented' or 'default')")
        if data_type is None:
            raise ValueError("data_type must be specified for classification task (use 'mihealthy' or 'subtype')")

        # Load the appropriate dataset config file
        config_name = f"{dataset}_{data_mode}_{data_type}"
        from hydra.utils import get_original_cwd
        config_path = Path(get_original_cwd()) / "config" / "classification" / f"{config_name}.yaml"

        if not config_path.exists():
            raise ValueError(f"Dataset config file not found: {config_path}")

        data_cfg = OmegaConf.load(config_path)
        print(f"Loaded dataset config: {config_name}")
        print(OmegaConf.to_yaml(data_cfg))

        # Get model and datamodule
        num_classes = data_cfg.num_classes
        model = get_classification_model(model_cfg.name, model_cfg, num_classes)
        datamodule = get_classification_dataset(dataset, data_cfg, data_mode, data_type)

    else:  # segmentation
        # Load segmentation dataset config
        from hydra.utils import get_original_cwd
        config_path = Path(get_original_cwd()) / "config" / "segmentation" / "default.yaml"

        if not config_path.exists():
            raise ValueError(f"Segmentation config file not found: {config_path}")

        data_cfg = OmegaConf.load(config_path)
        print(f"Loaded segmentation dataset config")
        print(OmegaConf.to_yaml(data_cfg))

        # Get segmentation model name from config (default: unet3pcgm)
        segmentation_model = cfg.get("segmentation_model", "unet3pcgm")

        # Load model config
        model_config_path = Path(get_original_cwd()) / "config" / "model" / f"{segmentation_model}.yaml"
        if not model_config_path.exists():
            raise ValueError(f"Model config file not found: {model_config_path}")

        model_cfg = OmegaConf.load(model_config_path)
        print(f"Loaded model config: {segmentation_model}")
        print(OmegaConf.to_yaml(model_cfg))

        # Get model and datamodule
        model = get_segmentation_model(segmentation_model, model_cfg)
        datamodule = get_segmentation_dataset(data_cfg.name, data_cfg)

    # Setup callbacks
    if task == "classification":
        output_dir = Path("outputs") / task / f"{model_cfg.name}" / f"{dataset}_{data_mode}" / f"{data_type}"
        logs_dir = Path("logs") / task / f"{model_cfg.name}" / f"{dataset}_{data_mode}" / f"{data_type}"
    else:
        output_dir = Path("outputs") / task / f"{data_cfg.name}"
        logs_dir = Path("logs") / task / f"{data_cfg.name}"

    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir,
        filename=f"{cfg.model.name}-{dataset}-{data_mode}-{data_type}-{{epoch:02d}}-{{val_loss:.4f}}",
        monitor="val_loss",
        mode="min",
        save_top_k=3,
        save_last=True,
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=10,
        mode="min",
    )

    # Setup logger
    if task == "classification":
        logger_name = f"{task}_{data_type}_{model_cfg.name}_{dataset}_{data_mode}_"
        wandb_project = "ecg_task_classification"
        wandb_tags = [task, model_cfg.name, dataset, data_mode, data_type]
    else:  # segmentation
        segmentation_model = cfg.get("segmentation_model", "unet3pcgm")
        logger_name = f"{task}_{segmentation_model}_{data_cfg.name}"
        wandb_project = "ecg_task_segmentation"
        wandb_tags = [task, segmentation_model, data_cfg.name]

    # Use WandB logger
    logger = WandbLogger(
        project=wandb_project,
        name=logger_name,
        tags=wandb_tags,
        save_dir=logs_dir,
        log_model=True,
    )

    # Log config to wandb
    logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))

    # Setup trainer
    trainer = L.Trainer(
        max_epochs=trainer_cfg.max_epochs,
        accelerator=trainer_cfg.accelerator,
        devices=trainer_cfg.devices,
        precision=trainer_cfg.precision,
        log_every_n_steps=trainer_cfg.log_every_n_steps,
        gradient_clip_val=trainer_cfg.gradient_clip_val,
        accumulate_grad_batches=trainer_cfg.accumulate_grad_batches,
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=logger,
    )

    # Train
    ckpt_path = cfg.get("resume_from_checkpoint", None)
    if ckpt_path:
        print(f"Resuming training from checkpoint: {ckpt_path}")
    trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)

    # Test
    trainer.test(model,
                 datamodule=datamodule,
                 ckpt_path=str(Path(output_dir) / "last.ckpt"),
                 weights_only=False)

    print(f"Training completed! Results saved to: {output_dir}")


if __name__ == "__main__":
    train()
