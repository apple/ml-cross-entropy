from dataclasses import dataclass

import torch.distributed


@dataclass
class VocabParallelOptions:
    start: int
    stop: int
    group: torch.distributed.ProcessGroup | None = None
    total_vocab_size: int | None = None
