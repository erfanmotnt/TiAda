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
        defaults = dict(
            lr=lr,
            lr_decay=0,
            eps=1e-10,
            weight_decay=0,
            initial_accumulator_value=0,
            foreach=None,
            maximize=False,
        )

        super(TiAda, self).__init__(params, defaults)

        self.params = params
        self.lr = lr
        self.alpha = alpha
        self.opponent_optim = opponent_optim

        self.v = torch.Tensor([0])

        
    @torch.no_grad()
    def step(self):
        loss = None
        for p in self.params:
            if p.grad is not None:
                grad2 = torch.mul(p.grad, p.grad)
                self.v.add_(grad2.sum())

        ratio = self.v.pow(self.alpha)
        if self.opponent_optim is not None:
            ratio = torch.max(ratio, self.opponent_optim.v.pow(self.alpha))

        for p in self.params:
            if p.grad is not None:
                p.sub_(p.grad.mul_(self.lr).div_(ratio.add_(1e-10)))
        return loss
