from torch.optim.optimizer import Optimizer
import torch
from typing import Optional

class TiAda(Optimizer):
    def __init__(
        self,
        params,
        lr=1e-2,
        alpha=0.5,
        opponent_optim=None,
    ):
        initial_accumulator_value=0
        
        super(TiAda, self).__init__(params, dict(
            lr=lr,
            lr_decay=0,
            eps=1e-10,
            weight_decay=0,
            initial_accumulator_value=initial_accumulator_value,
            foreach=None,
            maximize=False,
        ))

        self.alpha = alpha
        self.opponent_optim = opponent_optim

    def step(self):
        pass
