from collections.abc import Iterable
from typing import Any

import torch
from tqdm.auto import tqdm


def is_rank_zero_for_progress() -> bool:
    return (
        not torch.distributed.is_available()
        or not torch.distributed.is_initialized()
        or torch.distributed.get_rank() == 0
    )


def rank_zero_tqdm(
    iterable: Iterable | None = None,
    total: int | None = None,
    *,
    disable: bool = False,
    **kwargs: Any,
) -> tqdm:
    return tqdm(
        iterable=iterable,
        total=total,
        disable=disable or not is_rank_zero_for_progress(),
        **kwargs,
    )
