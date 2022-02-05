# coding=utf-8
# Copyright (c) Microsoft. All rights reserved.
import math
import torch
from torch.optim import Optimizer
from torch.nn.utils import clip_grad_norm_
from pytorch_pretrained_bert.optimization import warmup_constant, warmup_cosine, warmup_linear
from typing import Callable, Iterable, Tuple

def warmup_linear_xdl(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return (1.0 - x)/(1.0 - warmup)

def schedule_func(sch):
    try:
        f = eval(sch)
    except:
        f = warmup_linear
    return f

class Adamax(Optimizer):
    """Implements BERT version of Adam algorithm with weight decay fix (and no ).
    Params:
        lr: learning rate
        warmup: portion of t_total for the warmup, -1  means no warmup. Default: -1
        t_total: total number of training steps for the learning
            rate schedule, -1  means constant learning rate. Default: -1
        schedule: schedule to use for the warmup (see above). Default: 'warmup_linear'
        b1: Adams b1. Default: 0.9
        b2: Adams b2. Default: 0.999
        e: Adams epsilon. Default: 1e-6
        weight_decay: Weight decay. Default: 0.01
        max_grad_norm: Maximum norm for the gradients (-1 means no clipping). Default: 1.0
    by xiaodl
    """
    def __init__(self, params, lr, warmup=-1, t_total=-1, schedule='warmup_linear',
                 betas=(0.9, 0.999), eps=1e-6, weight_decay=0.01,
                 max_grad_norm=1.0):
        if not lr >= 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= warmup < 1.0 and not warmup == -1:
            raise ValueError("Invalid warmup: {} - should be in [0.0, 1.0[ or -1".format(warmup))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, schedule=schedule, warmup=warmup, t_total=t_total,
                        betas=betas, eps=eps, weight_decay=weight_decay,
                        max_grad_norm=max_grad_norm)
        super(Adamax, self).__init__(params, defaults)

    def get_lr(self):
        lr = []
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                if len(state) == 0:
                    return [0]
                if group['t_total'] != -1:
                    schedule_fct = schedule_func(group['schedule'])
                    lr_scheduled = group['lr'] * schedule_fct(state['step']/group['t_total'], group['warmup'])
                else:
                    lr_scheduled = group['lr']
                lr.append(lr_scheduled)
        return lr

    def to(self, device):
        """ Move the optimizer state to a specified device"""
        for state in self.state.values():
            state['exp_avg'].to(device)
            state['exp_inf'].to(device)

    def initialize_step(self, initial_step):
        """Initialize state with a defined step (but we don't have stored averaged).
        Arguments:
            initial_step (int): Initial step number.
        """
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                # State initialization
                state['step'] = initial_step
                # Exponential moving average of gradient values
                state['exp_avg'] = torch.zeros_like(p.data)
                # Exponential moving average of squared gradient values
                state['exp_inf'] = torch.zeros_like(p.data)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_inf'] = torch.zeros_like(p.data)

                exp_avg, exp_inf = state['exp_avg'], state['exp_inf']
                beta1, beta2 = group['betas']
                eps = group['eps']
                # Add grad clipping
                if group['max_grad_norm'] > 0:
                    clip_grad_norm_(p, group['max_grad_norm'])

                # Update biased first moment estimate.
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                # Update the exponentially weighted infinity norm.
                norm_buf = torch.cat([
                    exp_inf.mul_(beta2).unsqueeze(0),
                    grad.abs().add_(eps).unsqueeze_(0)
                ], 0)
                torch.max(norm_buf, 0, keepdim=False, out=(exp_inf, exp_inf.new().long()))
                update = exp_avg / (exp_inf + eps)

                if group['weight_decay'] > 0.0:
                    update += group['weight_decay'] * p.data

                if group['t_total'] != -1:
                    schedule_fct = schedule_func(group['schedule'])
                    lr_scheduled = group['lr'] * schedule_fct(state['step']/group['t_total'], group['warmup'])
                else:
                    lr_scheduled = group['lr']

                update_with_lr = lr_scheduled * update
                p.data.add_(-update_with_lr)
                state['step'] += 1

        return loss

class RAdam(Optimizer):
    """Modified from: https://github.com/LiyuanLucasLiu/RAdam/blob/master/radam.py
    """
    def __init__(self, params, lr, warmup=-1, t_total=-1, schedule='warmup_linear',
                 betas=(0.9, 0.999), eps=1e-6, weight_decay=0.001,
                 max_grad_norm=1.0):
        if not lr >= 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= warmup < 1.0 and not warmup == -1:
            raise ValueError("Invalid warmup: {} - should be in [0.0, 1.0[ or -1".format(warmup))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, schedule=schedule, warmup=warmup, t_total=t_total,
                        betas=betas, eps=eps, weight_decay=weight_decay,
                        max_grad_norm=max_grad_norm)
        self.buffer = [[None, None, None] for ind in range(10)]
        super(RAdam, self).__init__(params, defaults)

    def get_lr(self):
        lr = []
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                if len(state) == 0:
                    return [0]
                if group['t_total'] != -1:
                    schedule_fct = schedule_func(group['schedule'])
                    lr_scheduled = group['lr'] * schedule_fct(state['step']/group['t_total'], group['warmup'])
                else:
                    lr_scheduled = group['lr']
                lr.append(lr_scheduled)
        return lr

    def to(self, device):
        """ Move the optimizer state to a specified device"""
        for state in self.state.values():
            state['exp_avg'].to(device)
            state['exp_avg_sq'].to(device)

    def initialize_step(self, initial_step):
        """Initialize state with a defined step (but we don't have stored averaged).
        Arguments:
            initial_step (int): Initial step number.
        """
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                # State initialization
                state['step'] = initial_step
                # Exponential moving average of gradient values
                state['exp_avg'] = torch.zeros_like(p.data)
                # Exponential moving average of squared gradient values
                state['exp_avg_sq'] = torch.zeros_like(p.data)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        # set_trace()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('RAdam does not support sparse gradients')

                p_data_fp32 = p.data.float()

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                eps = group['eps']
                # Add grad clipping
                if group['max_grad_norm'] > 0:
                    clip_grad_norm_(p, group['max_grad_norm'])

                # Update biased first moment estimate.
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                state['step'] += 1

                if group['t_total'] != -1:
                    schedule_fct = schedule_func(group['schedule'])
                    lr_scheduled = group['lr'] * schedule_fct(state['step']/group['t_total'], group['warmup'])
                else:
                    lr_scheduled = group['lr']

                buffered = self.buffer[int(state['step'] % 10)]
                if state['step'] == buffered[0]:
                    N_sma, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state['step']
                    beta2_t = beta2 ** state['step']
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
                    buffered[1] = N_sma

                    # more conservative since it's an approximated value
                    if N_sma >= 5:
                        step_size = lr_scheduled * math.sqrt((1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (N_sma_max - 2)) / (1 - beta1 ** state['step'])
                    else:
                        step_size = lr_scheduled / (1 - beta1 ** state['step'])
                    buffered[2] = step_size

                if N_sma >= 5:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    p_data_fp32.addcdiv_(-step_size, exp_avg, denom)
                else:
                    p_data_fp32.add_(-step_size, exp_avg)

                if group['weight_decay'] != 0:
                    p_data_fp32.add_(-group['weight_decay'] * lr_scheduled, p_data_fp32)

                p.data.copy_(p_data_fp32)

        return loss

class AdamW(Optimizer):
    """
    Implements Adam algorithm with weight decay fix as introduced in `Decoupled Weight Decay Regularization
    <https://arxiv.org/abs/1711.05101>`__.
    Parameters:
        params (:obj:`Iterable[torch.nn.parameter.Parameter]`):
            Iterable of parameters to optimize or dictionaries defining parameter groups.
        lr (:obj:`float`, `optional`, defaults to 1e-3):
            The learning rate to use.
        betas (:obj:`Tuple[float,float]`, `optional`, defaults to (0.9, 0.999)):
            Adam's betas parameters (b1, b2).
        eps (:obj:`float`, `optional`, defaults to 1e-6):
            Adam's epsilon for numerical stability.
        weight_decay (:obj:`float`, `optional`, defaults to 0):
            Decoupled weight decay to apply.
        correct_bias (:obj:`bool`, `optional`, defaults to `True`):
            Whether ot not to correct bias in Adam (for instance, in Bert TF repository they use :obj:`False`).
    """

    def __init__(
        self,
        params: Iterable[torch.nn.parameter.Parameter],
        lr: float = 1e-3,
        warmup: float = -1,
        t_total: int = -1,
        schedule: str = 'warmup_linear',
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        correct_bias: bool = True,
        max_grad_norm: float = 1.0,
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= warmup < 1.0 and not warmup == -1:
            raise ValueError("Invalid warmup: {} - should be in [0.0, 1.0[ or -1".format(warmup))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(lr=lr, warmup=warmup, t_total=t_total, schedule=schedule, \
                        betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias, max_grad_norm=max_grad_norm)
        super(AdamW, self).__init__(params, defaults)

    def get_lr(self):
        lr = []
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                if len(state) == 0:
                    return [0]
                if group['t_total'] != -1:
                    schedule_fct = schedule_func(group['schedule'])
                    lr_scheduled = group['lr'] * schedule_fct(state['step']/group['t_total'], group['warmup'])
                else:
                    lr_scheduled = group['lr']
                lr.append(lr_scheduled)
        return lr

    def to(self, device):
        """ Move the optimizer state to a specified device"""
        for state in self.state.values():
            state['exp_avg'].to(device)
            state['exp_avg_sq'].to(device)

    def step(self, closure: Callable = None):
        """
        Performs a single optimization step.
        Arguments:
            closure (:obj:`Callable`, `optional`): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1
                # Add grad clipping
                if group['max_grad_norm'] > 0:
                    clip_grad_norm_(p, group['max_grad_norm'])

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                denom = exp_avg_sq.sqrt().add_(group["eps"])

                step_size = group["lr"]

                if group['t_total'] != -1:
                    schedule_fct = schedule_func(group['schedule'])
                    step_size = group['lr'] * schedule_fct(state['step']/group['t_total'], group['warmup'])
                else:
                    step_size = group['lr']

                if group["correct_bias"]:  # No bias correction for Bert
                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    bias_correction2 = 1.0 - beta2 ** state["step"]
                    step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

                p.data.addcdiv_(exp_avg, denom, value=-step_size)

                # Just adding the square of the weights to the loss function is *not*
                # the correct way of using L2 regularization/weight decay with Adam,
                # since that will interact with the m and v parameters in strange ways.
                #
                # Instead we want to decay the weights in a manner that doesn't interact
                # with the m/v parameters. This is equivalent to adding the square
                # of the weights to the loss with plain (non-momentum) SGD.
                # Add weight decay at the end (fixed version)
                if group["weight_decay"] > 0.0:
                    p.data.add_(p.data, alpha=-group["lr"] * group["weight_decay"])

        return loss

class UnstructAwareAdamW(Optimizer):
    """
    Implements Sensitvity-Guided AdamW (AdamW-SAGE) algorithm with weight decay fix as
    introduced in `No Parameters Left Behind: Sensitivity Guided Adaptive
    Learning Rate for Training Large Transformer Models `__.
    Parameters:
        params (:obj:`Iterable[torch.nn.parameter.Parameter]`):
            Iterable of parameters to optimize or dictionaries defining parameter groups.
        lr (:obj:`float`, `optional`, defaults to 1e-3):
            The learning rate to use.
        betas (:obj:`Tuple[float,float]`, `optional`, defaults to (0.9, 0.999)):
            Adam's betas parameters (b1, b2).
        beta3 (:obj:`float`,`optional`, default to 0.75):
            SAGE's beta parameter (b3).
        eps (:obj:`float`, `optional`, defaults to 1e-6):
            Adam's epsilon for numerical stability.
        weight_decay (:obj:`float`, `optional`, defaults to 0):
            Decoupled weight decay to apply.
        correct_bias (:obj:`bool`, `optional`, defaults to `True`):
            Whether ot not to correct bias in Adam (for instance, in Bert TF repository they use :obj:`False`).
    """

    def __init__(
        self,
        params: Iterable[torch.nn.parameter.Parameter],
        lr: float = 1e-3,
        warmup: float = -1,
        t_total: int = -1,
        schedule: str = 'warmup_linear',
        betas: Tuple[float, float] = (0.9, 0.999),
        beta3: float = 0.75,
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        correct_bias: bool = True,
        max_grad_norm: float = 1.0
    ):
        betas = betas + (beta3,)
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= warmup < 1.0 and not warmup == -1:
            raise ValueError("Invalid warmup: {} - should be in [0.0, 1.0[ or -1".format(warmup))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= betas[2] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[2]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(lr=lr, warmup=warmup, t_total=t_total, schedule=schedule, \
                        betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias, max_grad_norm=max_grad_norm)
        super(UnstructAwareAdamW, self).__init__(params, defaults)

    def get_lr(self):
        lr = []
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                if len(state) == 0:
                    return [0]
                if group['t_total'] != -1:
                    schedule_fct = schedule_func(group['schedule'])
                    lr_scheduled = group['lr'] * schedule_fct(state['step']/group['t_total'], group['warmup'])
                else:
                    lr_scheduled = group['lr']
                lr.append(lr_scheduled)
        return lr

    def to(self, device):
        """ Move the optimizer state to a specified device"""
        for state in self.state.values():
            state['exp_avg'].to(device)
            state['exp_avg_sq'].to(device)
            state['exp_avg_ipt'].to(device)

    def step(self, closure: Callable = None):
        """
        Performs a single optimization step.
        Arguments:
            closure (:obj:`Callable`, `optional`): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            beta1, beta2, beta3 = group["betas"]

            if group['params_type'] == 'common':
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    grad = p.grad.data
                    if grad.is_sparse:
                        raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                    state = self.state[p]

                    # State initialization
                    if len(state) == 0:
                        state["step"] = 0
                        # Exponential moving average of gradient values
                        state["exp_avg"] = torch.zeros_like(p.data)
                        # Exponential moving average of squared gradient values
                        state["exp_avg_sq"] = torch.zeros_like(p.data)

                    exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]

                    state["step"] += 1
                    # Add grad clipping
                    if group['max_grad_norm'] > 0:
                        clip_grad_norm_(p, group['max_grad_norm'])

                    # Decay the first and second moment running average coefficient
                    # In-place operations to update the averages at the same time
                    exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                    denom = exp_avg_sq.sqrt().add_(group["eps"])

                    step_size = group["lr"]

                    if group['t_total'] != -1:
                        schedule_fct = schedule_func(group['schedule'])
                        step_size = group['lr'] * schedule_fct(state['step']/group['t_total'], group['warmup'])
                    else:
                        step_size = group['lr']

                    if group["correct_bias"]:  # No bias correction for Bert
                        bias_correction1 = 1.0 - beta1 ** state["step"]
                        bias_correction2 = 1.0 - beta2 ** state["step"]
                        step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

                    p.data.addcdiv_(exp_avg, denom, value=-step_size)

                    # Just adding the square of the weights to the loss function is *not*
                    # the correct way of using L2 regularization/weight decay with Adam,
                    # since that will interact with the m and v parameters in strange ways.
                    #
                    # Instead we want to decay the weights in a manner that doesn't interact
                    # with the m/v parameters. This is equivalent to adding the square
                    # of the weights to the loss with plain (non-momentum) SGD.
                    # Add weight decay at the end (fixed version)
                    if group["weight_decay"] > 0.0:
                        p.data.add_(p.data, alpha=-group["lr"] * group["weight_decay"])
            else:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    grad = p.grad.data
                    if grad.is_sparse:
                        raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                    state = self.state[p]

                    # State initialization
                    if len(state) == 0:
                        state["step"] = 0
                        # Exponential moving average of gradient values
                        state["exp_avg"] = torch.zeros_like(p.data)
                        # Exponential moving average of squared gradient values
                        state["exp_avg_sq"] = torch.zeros_like(p.data)
                        state['exp_avg_ipt'] = torch.zeros_like(p.data)

                    exp_avg, exp_avg_sq, exp_avg_ipt = state['exp_avg'], state['exp_avg_sq'], state['exp_avg_ipt']

                    state["step"] += 1
                    # Add grad clipping
                    if group['max_grad_norm'] > 0:
                        clip_grad_norm_(p, group['max_grad_norm'])

                    # Decay the first and second moment running average coefficient
                    # In-place operations to update the averages at the same time
                    exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                    denom = exp_avg_sq.sqrt().add_(group["eps"])

                    # Update moving average of importance score
                    ipt = (p.data * grad).abs()
                    exp_avg_ipt.mul_(beta3).add_(ipt, alpha=1.0 - beta3)

                    step_size = group["lr"]

                    if group['t_total'] != -1:
                        schedule_fct = schedule_func(group['schedule'])
                        step_size = group['lr'] * schedule_fct(state['step']/group['t_total'], group['warmup'])
                    else:
                        step_size = group['lr']

                    if group["correct_bias"]:  # No bias correction for Bert
                        bias_correction1 = 1.0 - beta1 ** state["step"]
                        bias_correction2 = 1.0 - beta2 ** state["step"]
                        bias_correction3 = 1.0 - beta3 ** state["step"]
                        exp_avg_ipt_bias_corrected = exp_avg_ipt / bias_correction3
                        step_size *= (ipt - exp_avg_ipt_bias_corrected).abs() / (exp_avg_ipt_bias_corrected + group["eps"]**2)
                        step_size *= math.sqrt(bias_correction2) / bias_correction1
                    else:
                        step_size *= (ipt - exp_avg_ipt).abs() / (exp_avg_ipt + group["eps"]**2)

                    p.data -= exp_avg / denom * step_size

                    # Just adding the square of the weights to the loss function is *not*
                    # the correct way of using L2 regularization/weight decay with Adam,
                    # since that will interact with the m and v parameters in strange ways.
                    #
                    # Instead we want to decay the weights in a manner that doesn't interact
                    # with the m/v parameters. This is equivalent to adding the square
                    # of the weights to the loss with plain (non-momentum) SGD.
                    # Add weight decay at the end (fixed version)
                    if group["weight_decay"] > 0.0:
                        p.data.add_(p.data, alpha=-group["lr"] * group["weight_decay"])
        return loss

class UnstructAwareAdamax(Optimizer):
    """Implements BERT version of Sensitvity-Guided Adamax (Adamax-SAGE) algorithm with weight decay fix (and no ).
    Params:
        lr: learning rate
        warmup: portion of t_total for the warmup, -1  means no warmup. Default: -1
        t_total: total number of training steps for the learning
            rate schedule, -1  means constant learning rate. Default: -1
        schedule: schedule to use for the warmup (see above). Default: 'warmup_linear'
        b1: Adams b1. Default: 0.9
        b2: Adams b2. Default: 0.999
        beta3: SAGE b3. Default: 0.75.
        e: Adams epsilon. Default: 1e-6
        weight_decay: Weight decay. Default: 0.01
        max_grad_norm: Maximum norm for the gradients (-1 means no clipping). Default: 1.0
    by xiaodl
    """
    def __init__(self, params, lr, warmup=-1, t_total=-1, schedule='warmup_linear',
                 betas=(0.9, 0.999), beta3=0.75, eps=1e-6, weight_decay=0.01,
                 max_grad_norm=1.0):
        betas = betas + (beta3,)
        if not lr >= 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= warmup < 1.0 and not warmup == -1:
            raise ValueError("Invalid warmup: {} - should be in [0.0, 1.0[ or -1".format(warmup))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= betas[2] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[2]))
        defaults = dict(lr=lr, schedule=schedule, warmup=warmup, t_total=t_total,
                        betas=betas, eps=eps, weight_decay=weight_decay,
                        max_grad_norm=max_grad_norm)
        super(UnstructAwareAdamax, self).__init__(params, defaults)

    def get_lr(self):
        lr = []
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                if len(state) == 0:
                    return [0]
                if group['t_total'] != -1:
                    schedule_fct = schedule_func(group['schedule'])
                    lr_scheduled = group['lr'] * schedule_fct(state['step']/group['t_total'], group['warmup'])
                else:
                    lr_scheduled = group['lr']
                lr.append(lr_scheduled)
        return lr

    def to(self, device):
        """ Move the optimizer state to a specified device"""
        for state in self.state.values():
            state['exp_avg'].to(device)
            state['exp_inf'].to(device)
            state['exp_avg_ipt'].to(device)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            beta1, beta2, beta3 = group["betas"]

            if group['params_type'] == 'common':
                for p in group['params']:
                    if p.grad is None:
                        continue
                    grad = p.grad.data
                    if grad.is_sparse:
                        raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')

                    state = self.state[p]

                    # State initialization
                    if len(state) == 0:
                        state['step'] = 0
                        # Exponential moving average of gradient values
                        state['exp_avg'] = torch.zeros_like(p.data)
                        state['exp_inf'] = torch.zeros_like(p.data)

                    exp_avg, exp_inf = state['exp_avg'], state['exp_inf']

                    eps = group['eps']
                    # Add grad clipping
                    if group['max_grad_norm'] > 0:
                        clip_grad_norm_(p, group['max_grad_norm'])

                    # Update biased first moment estimate.
                    exp_avg.mul_(beta1).add_(1 - beta1, grad)
                    # Update the exponentially weighted infinity norm.
                    norm_buf = torch.cat([
                        exp_inf.mul_(beta2).unsqueeze(0),
                        grad.abs().add_(eps).unsqueeze_(0)
                    ], 0)
                    torch.max(norm_buf, 0, keepdim=False, out=(exp_inf, exp_inf.new().long()))
                    update = exp_avg / (exp_inf + eps)

                    if group['weight_decay'] > 0.0:
                        update += group['weight_decay'] * p.data

                    if group['t_total'] != -1:
                        schedule_fct = schedule_func(group['schedule'])
                        lr_scheduled = group['lr'] * schedule_fct(state['step']/group['t_total'], group['warmup'])
                    else:
                        lr_scheduled = group['lr']

                    update_with_lr = lr_scheduled * update
                    p.data.add_(-update_with_lr)
                    state['step'] += 1
            else:
                for p in group['params']:
                    if p.grad is None:
                        continue
                    grad = p.grad.data
                    if grad.is_sparse:
                        raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')

                    state = self.state[p]

                    # State initialization
                    if len(state) == 0:
                        state['step'] = 0
                        # Exponential moving average of gradient values
                        state['exp_avg'] = torch.zeros_like(p.data)
                        state['exp_inf'] = torch.zeros_like(p.data)
                        state['exp_avg_ipt'] = torch.zeros_like(p.data)

                    exp_avg, exp_inf, exp_avg_ipt = state['exp_avg'], state['exp_inf'], state['exp_avg_ipt']
                    eps = group['eps']
                    # Add grad clipping
                    if group['max_grad_norm'] > 0:
                        clip_grad_norm_(p, group['max_grad_norm'])

                    # Update biased first moment estimate.
                    exp_avg.mul_(beta1).add_(1 - beta1, grad)
                    # Update the exponentially weighted infinity norm.
                    norm_buf = torch.cat([
                        exp_inf.mul_(beta2).unsqueeze(0),
                        grad.abs().add_(eps).unsqueeze_(0)
                    ], 0)
                    torch.max(norm_buf, 0, keepdim=False, out=(exp_inf, exp_inf.new().long()))
                    update = exp_avg / (exp_inf + eps)
                    # Update moving average of importance score
                    ipt = (p.data * grad).abs()

                    exp_avg_ipt.mul_(beta3).add_(ipt, alpha=1.0 - beta3)

                    if group['t_total'] != -1:
                        schedule_fct = schedule_func(group['schedule'])
                        lr_scheduled = group['lr'] * schedule_fct(state['step']/group['t_total'], group['warmup'])
                    else:
                        lr_scheduled = group['lr']

                    if group['weight_decay'] > 0.0:
                        update += group['weight_decay'] * p.data

                    lr_scheduled *= (ipt - exp_avg_ipt).abs() / (exp_avg_ipt + eps**2)
                    update_with_lr = lr_scheduled * update
                    p.data.add_(-update_with_lr)
                    state['step'] += 1

        return loss
