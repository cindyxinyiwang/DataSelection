import math
import torch
from torch.optim.optimizer import Optimizer


class customAdam(Optimizer):
    r"""Implements Adam algorithm.

    It has been proposed in `Adam: A Method for Stochastic Optimization`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)

    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(self, params, hparams, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False):
        self.hparams = hparams
        self.scale_0, self.scale_1 = self.hparams.scale_0, self.hparams.scale_1
        #self.scale_0, self.scale_1 = torch.FloatTensor([self.hparams.scale_0]), torch.FloatTensor([self.hparams.scale_1])
        #if self.hparams.cuda: 
        #  self.scale_0, self.scale_1 = self.scale_0.cuda(), self.scale_1.cuda()
        self.baseline = 0
        self.cur_step = 0
        
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(customAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(customAdam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)
    #########
    # methods for bucketed version
    ########
    def get_cosine_sim_bucketed(self):
        # return a list of cosine sim of base lan and the lan_id
        cosine_prod = 0
        cosine_norm_train = 0
        cosine_norm_dev = 0
        base_lan_id = self.hparams.base_lan_id
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if not 'dev_grad' in state:
                  state['dev_grad'] = torch.zeros_like(p.data)
                if p.grad is None: continue
                grad = p.grad.data
                
                if not self.hparams.dev_adam_modified:
                  state["dev_grad"].mul_(self.scale_0).add_(grad *self.scale_1)
                else:
                  # clone so that we don't modify the grads
                  exp_avg, exp_avg_sq = state['exp_avg'].clone(), state['exp_avg_sq'].clone()
                  beta1, beta2 = group['betas']

                  if group['weight_decay'] != 0:
                      grad.add_(group['weight_decay'], p.data)
                  # Decay the first and second moment running average coefficient
                  exp_avg.mul_(beta1).add_(1 - beta1, grad)
                  exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                  denom = exp_avg_sq.sqrt().add_(group['eps'])
                  bias_correction1 = 1 - beta1 ** (state['step']+1)
                  bias_correction2 = 1 - beta2 ** (state['step']+1)
                  step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                  grad.mul_(step_size).mul_(self.scale_1).div_(denom)
                  state["dev_grad"].mul_(self.scale_0).add_(grad)

                cosine_prod += (state["train_grad"] * state["dev_grad"]).sum()
                cosine_norm_train += state["train_grad"].norm(2) ** 2
                cosine_norm_dev += state["dev_grad"].norm(2) ** 2
        cosine_dist = cosine_prod / (cosine_norm_dev.sqrt() * cosine_norm_train.sqrt()+1e-10)
        self.cur_step += 1
        return cosine_dist, cosine_prod

    def step_bucketed(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
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
                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    # Maintains max of all exp. moving avg. of sq. grad. values
                    state['max_exp_avg_sq'] = torch.zeros_like(p.data)
                    state['train_grad'] = torch.zeros_like(p.data)

                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad.add_(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                p.data.addcdiv_(-step_size, exp_avg, denom)
                
                if self.hparams.adam_raw_grad:
                  if self.hparams.train_adam_noscale:
                    state["train_grad"] = grad
                  else:
                    state["train_grad"].mul_(self.scale_0).add_(grad*self.scale_1)
                else:
                  if self.hparams.train_adam_modified:
                    if self.hparams.train_adam_noscale:
                      state["train_grad"] = grad*step_size/denom
                    else:
                      state["train_grad"].mul_(self.scale_0).add_(grad*step_size*self.scale_1/denom)
                  else:
                    if self.hparams.train_adam_noscale:
                      state["train_grad"] = exp_avg*step_size/denom
                    else:
                      state["train_grad"].mul_(self.scale_0).add_(exp_avg*step_size*self.scale_1/denom)
        return loss

    #########
    # end of methods for bucketed version
    ########

    def zero_prev_grad(self):
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                state["prev_grad"] = torch.zeros_like(p.data)
                state["exp_avg_grad"] = state["exp_avg"].clone()
                state["exp_avg_sq_grad"] = state["exp_avg_sq"].clone()
 
    def get_cosine_sim_all(self):
        # return a list of cosine sim of base lan and the lan_id
        cosine_prod = 0
        cosine_norm_train = 0
        cosine_norm_dev = 0
        base_lan_id = self.hparams.base_lan_id
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if p.grad is None: continue
                grad = p.grad.data
                
                if self.hparams.adam_raw_grad:
                  grad.mul_(self.scale_0).add_(grad *self.scale_1)
                else:
                  # clone so that we don't modify the grads
                  exp_avg, exp_avg_sq = state['exp_avg'].clone(), state['exp_avg_sq'].clone()
                  beta1, beta2 = group['betas']

                  if group['weight_decay'] != 0:
                      grad.add_(group['weight_decay'], p.data)
                  # Decay the first and second moment running average coefficient
                  exp_avg.mul_(beta1).add_(1 - beta1, grad)
                  exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                  denom = exp_avg_sq.sqrt().add_(group['eps'])
                  bias_correction1 = 1 - beta1 ** (state['step']+1)
                  bias_correction2 = 1 - beta2 ** (state['step']+1)
                  step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                  grad.mul_(step_size).mul_(self.scale_1).div_(denom)
                  state["dev_grad"].mul_(self.scale_0).add_(grad)

                cosine_prod += (grad * state["dev_grad"]).sum()
                cosine_norm_train += grad.norm(2) ** 2
                cosine_norm_dev += state["dev_grad"].norm(2) ** 2
        cosine_dist = cosine_prod / (cosine_norm_dev.sqrt() * cosine_norm_train.sqrt()+1e-10)
        self.cur_step += 1
        return cosine_dist, cosine_prod

    def save_gradients(self, lan_id):
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of gradient values for calculate grad sim
                    state['exp_avg_grad'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values for calculate grad sim
                    state['exp_avg_sq_grad'] = torch.zeros_like(p.data)
                    # Maintains max of all exp. moving avg. of sq. grad. values
                    state['max_exp_avg_sq'] = torch.zeros_like(p.data)
                    state["ave_grad"] = [torch.zeros_like(p.data) for _ in range(self.hparams.lan_size)]
                    state["prev_grad"] = torch.zeros_like(p.data)
                if p.grad is None: continue
                
                d_p = p.grad.data
                cur_grad = d_p - state["prev_grad"]
                
                if self.hparams.adam_raw_grad:
                  #state["ave_grad"][lan_id] = self.scale_0*state["ave_grad"][lan_id] + self.scale_1*cur_grad
                  state["ave_grad"][lan_id].mul_(self.scale_0).add_(self.scale_1*cur_grad)
                else:
                  exp_avg, exp_avg_sq = state['exp_avg_grad'], state['exp_avg_sq_grad']
                  beta1, beta2 = group['betas']

                  if group['weight_decay'] != 0:
                      cur_grad.add_(group['weight_decay'], p.data)
                  # Decay the first and second moment running average coefficient
                  exp_avg.mul_(beta1).add_(1 - beta1, cur_grad)
                  exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, cur_grad, cur_grad)
                  denom = exp_avg_sq.sqrt().add_(group['eps'])

                  bias_correction1 = 1 - beta1 ** (state['step']+1)
                  bias_correction2 = 1 - beta2 ** (state['step']+1)
                  step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                  #p.data.addcdiv_(-step_size, exp_avg, denom)
                  cur_grad.mul_(self.scale_1).mul_(step_size).div(denom)
                  state["ave_grad"][lan_id].mul_(self.scale_0).add_(cur_grad) 
                
                state["prev_grad"] = d_p.clone()
    
    def get_cosine_sim(self):
        # return a list of cosine sim of base lan and the lan_id
        cosine_prod = [0 for _ in range(self.hparams.lan_size)]
        cosine_norm = [0 for _ in range(self.hparams.lan_size)]
        base_lan_id = self.hparams.base_lan_id
        for group in self.param_groups:
            for p in group["params"]:
                param_state = self.state[p]
                if not "ave_grad" in param_state:
                    param_state["ave_grad"] = [torch.zeros_like(p.data) for _ in range(self.hparams.lan_size)]
                if p.grad is None: continue
                for i in range(self.hparams.lan_size):
                  prod = param_state["ave_grad"][i] * param_state["ave_grad"][base_lan_id]
                  prod = prod.sum()
                  cosine_prod[i] = cosine_prod[i] + prod
                  if self.hparams.grad_dist == "cosine":
                    norm = param_state["ave_grad"][i].norm(2) ** 2
                    cosine_norm[i] = cosine_norm[i] + norm  
        if self.hparams.grad_dist == "cosine":
          cosine_dist = [p / (n.sqrt()*cosine_norm[base_lan_id].sqrt() +1e-10) for p, n in zip(cosine_prod, cosine_norm)]
        elif self.hparams.grad_dist == "dot_prod":
          cosine_dist = cosine_prod
        return cosine_dist

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
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
                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of gradient values for calculate grad sim
                    state['exp_avg_grad'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values for calculate grad sim
                    state['exp_avg_sq_grad'] = torch.zeros_like(p.data)
                    # Maintains max of all exp. moving avg. of sq. grad. values
                    state['max_exp_avg_sq'] = torch.zeros_like(p.data)
                    state["ave_grad"] = [torch.zeros_like(p.data) for _ in range(self.hparams.lan_size)]
                    state["prev_grad"] = torch.zeros_like(p.data)

                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad.add_(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                p.data.addcdiv_(-step_size, exp_avg, denom)

        return loss
