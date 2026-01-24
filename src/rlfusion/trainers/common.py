import logging
from typing import Any, cast

import torch


def configure_logging(logger: logging.Logger, log_level: int) -> None:
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )
    logger.setLevel(log_level)


def is_main_process(accelerator: Any) -> bool:
    return accelerator is None or bool(getattr(accelerator, "is_main_process", True))


def unwrap_model_for_saving(model: torch.nn.Module, accelerator: Any) -> torch.nn.Module:
    if accelerator is None:
        return model
    return cast(torch.nn.Module, accelerator.unwrap_model(model))
