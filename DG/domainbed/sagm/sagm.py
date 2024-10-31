import torch
from .util import enable_running_stats, disable_running_stats
import contextlib
from torch.distributed import ReduceOp
import numpy as np


def to_minibatch(x, y):
    minibatches = list(zip(x, y))
    return minibatches


def random_pairs_of_minibatches(minibatches):
    # n_tr_envs = len(minibatches)
    perm = torch.randperm(len(minibatches)).tolist()
    pairs = []

    for i in range(len(minibatches)):
        # j = cyclic(i + 1)
        j = i + 1 if i < (len(minibatches) - 1) else 0

        xi, yi = minibatches[perm[i]][0], minibatches[perm[i]][1]
        xj, yj = minibatches[perm[j]][0], minibatches[perm[j]][1]

        min_n = min(len(xi), len(xj))

        pairs.append(((xi[:min_n], yi[:min_n]), (xj[:min_n], yj[:min_n])))

    return pairs


class SAGM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, model, alpha, rho_scheduler, adaptive=False, perturb_eps=1e-12,
                 grad_reduce='mean', **kwargs):
        defaults = dict(adaptive=adaptive, **kwargs)
        super(SAGM, self).__init__(params, defaults)
        self.model = model
        self.base_optimizer = base_optimizer
        self.param_groups = self.base_optimizer.param_groups
        self.adaptive = adaptive
        self.rho_scheduler = rho_scheduler
        self.perturb_eps = perturb_eps
        self.alpha = alpha

        # initialize self.rho_t
        self.update_rho_t()

        # set up reduction for gradient across workers
        if grad_reduce.lower() == 'mean':
            if hasattr(ReduceOp, 'AVG'):
                self.grad_reduce = ReduceOp.AVG
                self.manual_average = False
            else:  # PyTorch <= 1.11.0 does not have AVG, need to manually average across processes
                self.grad_reduce = ReduceOp.SUM
                self.manual_average = True
        elif grad_reduce.lower() == 'sum':
            self.grad_reduce = ReduceOp.SUM
            self.manual_average = False
        else:
            raise ValueError('"grad_reduce" should be one of ["mean", "sum"].')

    @torch.no_grad()
    def update_rho_t(self):
        self.rho_t = self.rho_scheduler.step()
        return self.rho_t

    @torch.no_grad()
    def perturb_weights(self, rho=0.0):
        grad_norm = self._grad_norm(weight_adaptive=self.adaptive)
        for group in self.param_groups:
            scale = (rho / (grad_norm + self.perturb_eps) - self.alpha)

            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_g"] = p.grad.data.clone()
                e_w = p.grad * scale.to(p)
                if self.adaptive:
                    e_w *= torch.pow(p, 2)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"
                self.state[p]['e_w'] = e_w

    @torch.no_grad()
    def unperturb(self):
        for group in self.param_groups:
            for p in group['params']:
                if 'e_w' in self.state[p].keys():
                    p.data.sub_(self.state[p]['e_w'])

    @torch.no_grad()
    def gradient_decompose(self, alpha=0.0):

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                sam_grad = self.state[p]['old_g'] * 0.5 - p.grad * 0.5
                p.grad.data.add_(sam_grad)

    @torch.no_grad()
    def _sync_grad(self):
        if torch.distributed.is_initialized():  # synchronize final gardients
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None: continue
                    if self.manual_average:
                        torch.distributed.all_reduce(p.grad, op=self.grad_reduce)
                        world_size = torch.distributed.get_world_size()
                        p.grad.div_(float(world_size))
                    else:
                        torch.distributed.all_reduce(p.grad, op=self.grad_reduce)
        return

    @torch.no_grad()
    def _grad_norm(self, by=None, weight_adaptive=False):
        # shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        if not by:

            norm = torch.norm(
                torch.stack([
                    ((torch.abs(p.data) if weight_adaptive else 1.0) * p.grad).norm(p=2)
                    for group in self.param_groups for p in group["params"]
                    if p.grad is not None
                ]),
                p=2
            )

        else:

            norm = torch.norm(
                torch.stack([
                    ((torch.abs(p.data) if weight_adaptive else 1.0) * self.state[p][by]).norm(p=2)
                    for group in self.param_groups for p in group["params"]
                    if p.grad is not None
                ]),
                p=2
            )

        return norm

    # def norm(tensor_list: List[torch.tensor], p=2):
    #     """Compute p-norm for tensor list"""
    #     return torch.cat([x.flatten() for x in tensor_list]).norm(p)

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups

    def maybe_no_sync(self):
        if torch.distributed.is_initialized():
            return self.model.no_sync()
        else:
            return contextlib.ExitStack()

    @torch.no_grad()
    def set_closure(self, loss_fn, inputs, targets, weight, **kwargs):
        # create self.forward_backward_func, which is a function such that
        # self.forward_backward_func() automatically performs forward and backward passes.
        # This function does not take any arguments, and the inputs and targets data
        # should be pre-set in the definition of partial-function

        def get_grad():
            self.base_optimizer.zero_grad()
            with torch.enable_grad():
                outputs = self.model(inputs)
                loss = loss_fn(outputs, targets, weight, **kwargs)
            loss_value = loss.data.clone().detach()
            loss.backward()
            return outputs, loss_value

        self.forward_backward_func = get_grad

    @torch.no_grad()
    def set_closure_mixup(self, loss_fn, inputs, targets, mixup_alpha, **kwargs):
        def get_grad():
            self.base_optimizer.zero_grad()
            with torch.enable_grad():
                minibatches = to_minibatch(inputs, targets)
                objective = 0

                for (xi, yi), (xj, yj) in random_pairs_of_minibatches(minibatches):
                    lam = np.random.beta(mixup_alpha, mixup_alpha)

                    x = lam * xi + (1 - lam) * xj
                    predictions = self.model(x)

                    objective += lam * loss_fn(predictions, yi)
                    objective += (1 - lam) * loss_fn(predictions, yj)
                objective /= len(minibatches)
                # outputs = self.model(inputs)
                # loss = loss_fn(outputs, targets, **kwargs)
                loss = objective
            loss_value = loss.data.clone().detach()
            loss.backward()
            return predictions, loss_value

        self.forward_backward_func = get_grad

    @torch.no_grad()
    def step(self, closure=None):

        if closure:
            get_grad = closure
        else:
            get_grad = self.forward_backward_func

        with self.maybe_no_sync():
            # get gradient
            outputs, loss_value = get_grad()

            # perturb weights
            self.perturb_weights(rho=self.rho_t)

            # disable running stats for second pass
            disable_running_stats(self.model)

            # get gradient at perturbed weights
            get_grad()

            # decompose and get new update direction
            self.gradient_decompose(self.alpha)

            # unperturb
            self.unperturb()

        # synchronize gradients across workers
        self._sync_grad()

        # update with new directions
        self.base_optimizer.step()

        # enable running stats
        enable_running_stats(self.model)

        return outputs, loss_value
