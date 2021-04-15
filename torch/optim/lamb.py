import torch
from . import _functional as F
from .optimizer import Optimizer


class Lamb(Optimizer):
    r"""Implements Lamb algorithm.

    It has been proposed in `Large Batch Optimization for Deep Learning:
    Training BERT in 76 minutes`_.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        fused (boolean, optional): whether to use fused kernel to accelerate
            (default: False)

    .. _Large Batch Optimization for Deep Learning: Training BERT in 76 minutes:
        https://arxiv.org/abs/1904.00962
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, fused=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, fused=fused)
        super(Lamb, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Lamb, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('fused', False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            trails = []
            state_steps = []

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    if p.grad.is_sparse:
                        raise RuntimeError('Lamb does not support sparse gradients')
                    if p.grad.device != torch.device('cpu'):
                        raise RuntimeError('Lamb supports only CPU device')
                    grads.append(p.grad)

                    state = self.state[p]
                    # Lazy state initialization
                    if len(state) == 0:
                        use_split_sgd = p.dtype == torch.bfloat16 and group['fused']
                        state['step'] = 0
                        if use_split_sgd:
                            # Exponential moving average of gradient values
                            state['exp_avg'] = torch.zeros_like(p, dtype=torch.float, memory_format=torch.preserve_format)
                            # Exponential moving average of squared gradient values
                            state['exp_avg_sq'] = torch.zeros_like(p, dtype=torch.float, memory_format=torch.preserve_format)
                            # Lower 16 bits of master weight (stored as BFloat16)
                            state['trail'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        else:
                            state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                            state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                            state['trail'] = torch.Tensor()

                    exp_avgs.append(state['exp_avg'])
                    exp_avg_sqs.append(state['exp_avg_sq'])
                    trails.append(state['trail'])

                    # update the steps for each param group update
                    state['step'] += 1
                    # record the step after step update
                    state_steps.append(state['step'])

            beta1, beta2 = group['betas']
            F.lamb(params_with_grad,
                   grads,
                   exp_avgs,
                   exp_avg_sqs,
                   trails,
                   state_steps,
                   beta1,
                   beta2,
                   group['lr'],
                   group['weight_decay'],
                   group['eps'],
                   group['fused'])
        return loss
