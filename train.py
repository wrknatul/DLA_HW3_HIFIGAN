import warnings

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from itertools import chain
from src.datasets.data_utils import get_dataloaders
from src.trainer import Trainer
from src.utils.init_utils import set_random_seed, setup_saving_and_logging

warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(version_base=None, config_path="src/configs", config_name="baseline")
def main(config):
    """
    Main script for training. Instantiates the model, optimizer, scheduler,
    metrics, logger, writer, and dataloaders. Runs Trainer to train and
    evaluate the model.

    Args:
        config (DictConfig): hydra experiment config.
    """
    set_random_seed(config.trainer.seed)

    project_config = OmegaConf.to_container(config)
    logger = setup_saving_and_logging(config)
    writer = instantiate(config.writer, logger, project_config)

    if config.trainer.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.trainer.device

    # setup data_loader instances
    # batch_transforms should be put on device
    dataloaders, batch_transforms = get_dataloaders(config, device)

    # build model architecture, then print to console
    model = instantiate(config.model).to(device)
    logger.info(model)

    # get function handles of loss and metrics
    loss_function = instantiate(config.loss_function).to(device)
    metrics = config.metrics

    # build optimizer, learning rate scheduler
    disc_params = chain(
        filter(lambda p: p.requires_grad, model.mpd.parameters()),
        filter(lambda p: p.requires_grad, model.msd.parameters())
    )
    optimizer_disk = instantiate(config.optimizer_disk, params=disc_params)
    gen_params = filter(lambda p: p.requires_grad, model.generator.parameters())
    optimizer_gen = instantiate(config.optimizer_gen, params=gen_params)

    lr_scheduler_disk = instantiate(config.lr_scheduler_disk, optimizer=optimizer_disk)
    lr_scheduler_gen = instantiate(config.lr_scheduler_gen, optimizer=optimizer_gen)

    # epoch_len = number of iterations for iteration-based training
    # epoch_len = None or len(dataloader) for epoch-based training
    epoch_len = config.trainer.get("epoch_len")

    trainer = Trainer(
        model=model,
        criterion=loss_function,
        metrics=metrics,
        optimizer_disk=optimizer_disk,
        optimizer_gen=optimizer_gen,
        lr_scheduler_disk=lr_scheduler_disk,
        lr_scheduler_gen=lr_scheduler_gen,
        config=config,
        device=device,
        dataloaders=dataloaders,
        epoch_len=epoch_len,
        logger=logger,
        writer=writer,
        batch_transforms=batch_transforms,
        skip_oom=config.trainer.get("skip_oom", True),
    )

    trainer.train()


if __name__ == "__main__":
    main()
