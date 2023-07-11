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

        self.total_sum = self.param_groups[0]["params"][0].new_zeros(1)
        with torch.no_grad():
            for p in self.params:
                state = self.state[p]
                init_value = 0
                state["sum"] = torch.full_like(
                    p, init_value, memory_format=torch.preserve_format
                )

                self.total_sum.add_(state["sum"].sum())

    @torch.no_grad()
    def step(self):
        loss = None

        for p in self.params:
            if p.grad is not None:
                grad2 = torch.mul(p.grad, p.grad)
                self.state[p]["sum"].add_(grad2)
                self.total_sum.add_(grad2.sum())

        if self.opponent_optim is not None:
            ratio = self.total_sum.pow(self.alpha)
            ratio.div_(
                    torch.max(
                        ratio,
                        self.opponent_optim.total_sum.pow(self.alpha)
                        )
                    )
        else:
            ratio = 1

        for p in self.params:
            if p.grad is not None:
                state_sum = self.state[p]["sum"]
                diff =p.grad.mul_(ratio).div_(state_sum.pow(self.alpha).add_(1e-10))
                p.sub_(diff.mul_(self.lr))   

        return loss
