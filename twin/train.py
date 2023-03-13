"""
Main train script
"""
import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, TQDMProgressBar

from twin.data.datamodule import TwinDataModule
from twin.data.images import Images
from twin.data.synthetic import Synthetic
from twin.runner import TwinRunner


def train(max_epochs=500, seed=42):
    """
    Fit textured mesh
    :param max_epochs: number of epochs
    :param seed: random state
    """
    print("Training")
    seed_everything(seed, workers=True)

    datamodule = TwinDataModule(dataset_cl=Images)
    camera_params = datamodule.dataset_cl().camera_params
    runner = TwinRunner(
        mesh_only_epochs=max_epochs // 4, size=128, camera_params=camera_params
    )

    callbacks = [
        LearningRateMonitor(),
        # ModelCheckpoint(),
        TQDMProgressBar(refresh_rate=1),
    ]
    use_cuda = torch.cuda.is_available()
    trainer_params = {
        "accelerator": "gpu" if use_cuda else "cpu",
        "benchmark": False,
        "gpus": -1 if use_cuda else 0,
        "max_epochs": max_epochs,
        "num_sanity_val_steps": 0,
        "callbacks": callbacks,
    }
    trainer = Trainer(**trainer_params)

    trainer.fit(runner, datamodule)

    # Save
    runner.save_textured_mesh("model.obj")


if __name__ == "__main__":
    train()
