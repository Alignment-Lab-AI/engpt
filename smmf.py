import torch
from typing import Optional

class SMMF(torch.optim.Optimizer):
    def __init__(
        self, params, lr:float=1e-3, beta:Optional[float]=0.9, eps:float=1e-8, weight_decay:float=0.0,
        decay_rate:float=-0.5, growth_rate:Optional[float]=0.999, vector_reshape:bool=True, weight_decay_mode='adamw'
    ):
        if not (0.0 <= lr):
            raise ValueError(f'Learning-rate should be greater than or equal to 0.0. Current: {lr}')
        if beta != None:
            if not (0.0 <= beta and beta <= 1.0):
                raise ValueError(f'Beta should be in [0.0, 1.0]. Current: {beta}')
        if not (0.0 <= eps):
            raise ValueError(f'Epilson should be greater than or equal to 0.0. Current: {eps}')
        if not (0.0 <= weight_decay):
            raise ValueError(f'Weight-decay should be greater than or equal to 0.0. Current: {weight_decay}')
        if not (-1.0 <= decay_rate and decay_rate <= 0.0):
            raise ValueError(f'Decay-rate should be in [-1.0, 0.0]. Current: {decay_rate}')
        if not (0.0 <= growth_rate and growth_rate <= 1.0):
            raise ValueError(f'Growth-rate should be in [0.0, 1.0]. Current: {growth_rate}')
        
        if (beta == None) and (growth_rate != None):
            Warning("Beta is not used, but growth_rate is defined.")
        defaults = dict(
            lr=lr, beta=beta, eps=eps, weight_decay=weight_decay,
            decay_rate=decay_rate, growth_rate=growth_rate, vector_reshape=vector_reshape
        )
        super(SMMF, self).__init__(params, defaults)
        self.weight_decay_mode = weight_decay_mode
    
    @torch.no_grad()
    def step(self, closure=None):
        '''Implements SMMF Algorithm.
        Args:
            params (iterable): Iterable of parameters to optimize or dicts defining parameter groups
            lr: Learning-rate (default: 1e-3)
            beta: Coefficient used for computing running average of gradient (default: 0.9)
            eps: Regularization constant for square gradient (default: 1e-8)
            weight_decay: Weight-decay (default: 0.0)
            decay_rate: Decay-rate coefficient used for computing running average of gradient (default: -0.8)
            growth_rate: Growth-rate coefficient used for computing running average of square gradient (default: 0.999)
            vector_reshape: Vector Square-Matricization (default: True)
        '''
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
            
        for group in self.param_groups:
            for params in group['params']:
                grad = params.grad
                if grad is None:
                    continue
                
                if group['weight_decay'] != 0.0 and self.weight_decay_mode == 'adam':
                    grad = grad.add(params, alpha=group['weight_decay'])
                elif group['weight_decay'] != 0.0 and self.weight_decay_mode == 'adamw':
                    params.mul_(1 - group['lr'] * group['weight_decay'])
                
                dimension = len(grad.squeeze().shape)
                factorization = not (dimension == 1 and (not group['vector_reshape']))
                
                if factorization:
                    update = self._factorized_adam(params, group)
                else:
                    update = self._adam(params, group)
                
                params.add_(update, alpha=-group['lr'])
        return loss
    
    @torch.no_grad()
    def _factorized_adam(self, params, group):
        beta = group['beta']
        eps = group['eps']
        decay_rate = group['decay_rate']
        growth_rate = group['growth_rate']
        
        grad = params.grad
        state = self.state[params]
        original_shape = grad.shape
        device = grad.device
        
        if len(state) == 0:
            state['step'] = 1
            state['effective_shape'] = self._get_effective_shape(params.numel())
            if beta != None:
                state['momentum_m'] = (
                    torch.zeros(state['effective_shape'][0], device=device),
                    torch.zeros(state['effective_shape'][1], device=device),
                )
                state['sign'] = torch.zeros(state['effective_shape'], dtype=torch.bool, device=device)              
            state['momentum_v'] = (
                torch.zeros(state['effective_shape'][0], device=device),
                torch.zeros(state['effective_shape'][1], device=device),
            )
        
        if not grad.is_contiguous():
            grad = grad.contiguous()
        grad = grad.view(state['effective_shape'])
        
        # Decompressing
        if beta != None:
            update_m = self._decompression(state, 'momentum_m')
        update_v = self._decompression(state, 'momentum_v')
        
        # Update
        if beta != None:
            beta_m = beta * growth_rate ** (state['step'] - 1.0)
            update_m.mul_(beta_m).add_(grad, alpha=(1.0 - beta_m))
        beta_v = 1.0 - state['step'] ** decay_rate
        update_v.mul_(beta_v).add_(grad ** 2, alpha=(1.0 - beta_v))
        
        # Compressing
        if beta != None:
            self._compression(update_m, state, 'momentum_m')
        self._compression(update_v, state, 'momentum_v')
        
        # Compute and Reshape
        if beta != None:
            update = update_m / (update_v.sqrt() + eps)
        else:
            update = grad / (update_v.sqrt() + eps)
        update = update.contiguous().view(original_shape)
        
        state['step'] += 1.0
        return update
    
    @torch.no_grad()
    def _adam(self, params, group):
        beta = group['beta']
        eps = group['eps']
        decay_rate = group['decay_rate']
        growth_rate = group['growth_rate']
        
        grad = params.grad
        state = self.state[params]
        
        if len(state) == 0:
            state['step'] = 1
            if beta != None:
                state['momentum_m'] = torch.zeros_like(params)
            state['momentum_v'] = torch.zeros_like(params)
            
        if beta != None:
            update_m = state['momentum_m']
        update_v = state['momentum_v']
        
        if beta != None:
            beta_m = beta * growth_rate ** (state['step'] - 1.0)
            update_m.mul_(beta_m).add_(grad, alpha=(1.0 - beta_m))
        beta_v = 1.0 - state['step'] ** decay_rate
        update_v.mul_(beta_v).add_(grad ** 2, alpha=(1.0 - beta_v))
        
        if beta != None:
            state['momentum_m'] = update_m
        state['momentum_v'] = update_v
        
        if beta != None:
            update = update_m / (update_v.sqrt() + eps)
        else:
            update = grad / (update_v.sqrt() + eps)
        
        state['step'] += 1.0
        return update
        
    @torch.no_grad()
    def _get_effective_shape(self, numel:int)->tuple:
        sqrt_num = int(numel ** 0.5) ** 2
        
        if numel == sqrt_num:
            sqrt_num = int(numel ** 0.5)
            return (sqrt_num, sqrt_num)
        
        reversed_range = reversed(range(1, int(numel **0.5) + 1))
        for i in reversed_range:
            if numel % i == 0:
                return (numel // i, i)
            
        return (numel, 1)
    
    @torch.no_grad()
    def _decompression(self, state, momentum:str)->torch.Tensor:
        update = self._unnmf(state[momentum])
        if momentum == 'momentum_m':
            sign = state['sign']
            if sign.dtype != torch.bool:
                sign = sign.type(torch.bool)
            torch.where(sign, update, -update, out=update)
        return update
        
    @torch.no_grad()
    def _compression(self, matrix:torch.Tensor, state, momentum:str)->tuple:
        if momentum == 'momentum_m':
            state['sign'] = matrix > 0
            self._nnmf(torch.abs(matrix), out=state[momentum] )
        else:
            self._nnmf(matrix, out=state[momentum])
        return state[momentum]
            
    @torch.no_grad()
    def _unnmf(self, row_col:tuple)->torch.Tensor:
        return torch.outer(row_col[0], row_col[1])

    @torch.no_grad()
    def _nnmf(self, matrix:torch.Tensor, out)->tuple:
        shape = matrix.shape
        torch.sum(matrix, dim=1, out=out[0])
        torch.sum(matrix, dim=0, out=out[1])

        if shape[0] < shape[1]:
            scale = out[0].sum()
            if scale != 0:
                torch.div(out[0], scale, out=out[0])
        else:
            scale = out[1].sum()
            if scale != 0:
                torch.div(out[1], scale, out=out[1])

        return out

