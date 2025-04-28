import torch
import torch.optim
from typing import Callable, Optional, Iterable, Dict, Any

class IntegratedOptimizerWrapper(torch.optim.Optimizer):
    """
    Integrates SAM, OrthoGrad, and ScheduleFree logic around a base optimizer.
    (Corrected OrthoGrad calculation)

    Order of operations within step(closure):
    1. SAM First Step: Calculate L(theta), grad(L(theta)), perturb weights -> theta + e_w.
    2. SAM Second Step: Calculate L(theta + e_w), grad(L(theta + e_w)). Restore weights -> theta.
                         p.grad now holds grad(L(theta + e_w)).
    3. OrthoGrad: Orthogonalize the SAM gradient in p.grad w.r.t. theta (with norm rescaling).
    4. ScheduleFree Momentum Prep & Base Step Trigger:
       - Store processed grad for base optimizer.
       - Assign grad to z.grad.
    5. Base Optimizer Step: Execute base_optimizer.step() which reads z.grad and updates z.data.
    6. ScheduleFree Update: Restore z.grad, perform weighted averaging update on p (theta),
                            calculate next momentum point y.
    """
    def __init__(self,
                 params: Iterable[torch.Tensor] | Iterable[Dict[str, Any]],
                 base_optimizer_class: type[torch.optim.Optimizer], # e.g., SMMF
                 # Base Optimizer Args
                 lr: float = 1e-3, # Base LR needed for ScheduleFree WD and base optimizer
                 base_optimizer_args: Optional[dict] = None,
                 # ScheduleFree Args
                 weight_decay_at_y: float = 0.0,
                 momentum: float = 0.9,
                 weight_lr_power: float = 2.0,
                 r: float = 0.0,
                 # SAM Args
                 rho: float = 0.05,
                 adaptive_sam: bool = False,
                 eps_sam: float = 1e-12, # Epsilon specific to SAM grad norm
                 # OrthoGrad Args
                 eps_ortho: float = 1e-30, # Epsilon for OrthoGrad projection stability
                 # General Args
                 **kwargs # Catch-all, potentially passed to base_optimizer
                 ):

        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"Momentum must be in [0, 1). Current: {momentum}")
        if not rho >= 0.0:
             raise ValueError(f"SAM rho must be >= 0. Current: {rho}")

        self.momentum = momentum
        self.weight_decay_at_y = weight_decay_at_y
        self.weight_lr_power = weight_lr_power
        self.r = r
        self.rho = rho
        self.adaptive_sam = adaptive_sam
        self.eps_sam = eps_sam
        self.eps_ortho = eps_ortho # Store OrthoGrad epsilon
        self.train_mode = True # Start in train mode by default

        # Initialize defaults for the wrapper state (ScheduleFree part mostly)
        defaults = dict(
            lr=lr, # Store base LR
            k=0.0,
            d=1.0,
            lr_max=0.0,
            weight_sum=0.0,
            rho=rho,
            adaptive_sam=adaptive_sam,
            eps_sam=eps_sam,
            eps_ortho=eps_ortho
        )
        super().__init__(params, defaults)

        # Instantiate the base optimizer *after* super().__init__ has processed params
        base_args = base_optimizer_args if base_optimizer_args is not None else {}
        base_args['lr'] = lr
        base_args.update(kwargs)
        self.base_optimizer = base_optimizer_class(self.param_groups, **base_args)

        # Initialize base optimizer state if needed (already done by base_optimizer init usually)


    def _orthogonalize_gradients(self, params_with_grad: Iterable[torch.Tensor], eps: float):
        """
        Applies OrthoGrad projection to gradients in-place.
        Corrected version with norm rescaling.
        """

        for p in params_with_grad:
            if p.grad is None or p.dim() < 1: # Skip scalars or params without grad
                continue

            # Use p.data for orthogonalization w.r.t current weights
            w = p.data.view(-1)
            g = p.grad.view(-1)

            # Project gradient to be orthogonal to the parameters
            w_norm_sq = torch.dot(w, w).add_(eps) # Add epsilon for stability
            proj_coeff = torch.dot(w, g) / w_norm_sq

            g_orth = g - proj_coeff * w

            # Rescale g_orth to have the same norm as g
            g_norm = torch.norm(g, p=2)
            g_orth_norm = torch.norm(g_orth, p=2).add_(eps) # Avoid div by zero
            scale_factor = g_norm / g_orth_norm

            # Update gradient in-place with the orthogonalized and rescaled version
            p.grad.copy_(g_orth.mul_(scale_factor).view_as(p.grad))

    @staticmethod
    def _swap_tensors(tensor1: torch.Tensor, tensor2: torch.Tensor):
        """In-place swap of tensor data using bitwise XOR."""
        assert tensor1.shape == tensor2.shape and tensor1.dtype == tensor2.dtype and tensor1.device == tensor2.device
        view1 = tensor1.view(torch.uint8)
        view2 = tensor2.view(torch.uint8)
        view1.bitwise_xor_(view2)
        view2.bitwise_xor_(view1)
        view1.bitwise_xor_(view2)


    def train(self, mode: bool = True):
        """Sets the optimizer to training mode (for ScheduleFree momentum)."""
        # --- Logic exactly as before ---
        if not mode: # Switching to eval
            if self.train_mode: # Was training before
                for group in self.param_groups:
                     momentum = self.momentum # Use wrapper's momentum
                     for p in group['params']:
                         state = self.state[p]
                         if 'z' in state:
                              p.data.lerp_(state['z'], weight=1 - 1/momentum)
            self.train_mode = False
        else: # Switching to train
             if not self.train_mode: # Was eval before
                 for group in self.param_groups:
                     momentum = self.momentum
                     for p in group['params']:
                          state = self.state[p]
                          if 'z' in state:
                              p.data.lerp_(state['z'], weight=1 - momentum)
             self.train_mode = True
        return self

    def eval(self):
        """Sets the optimizer to evaluation mode."""
        return self.train(mode=False)


    def _sam_first_step(self):
        """Calculates perturbation e_w and applies it to weights."""
        # --- Logic exactly as before ---
        global_grad_norm = self._sam_grad_norm()
        for group in self.param_groups:
            rho = group['rho']
            adaptive = group['adaptive_sam']
            eps = group['eps_sam']
            scale = rho / (global_grad_norm + eps)

            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]
                state['sam_old_p'] = p.data.clone()

                if adaptive:
                    # ASAM approx: scale perturbation by |w|
                    e_w = (torch.abs(p.data)* p.grad) * scale # Rough ASAM interpretation needs care
                    # More standard ASAM interpretation uses param_norm in scale calc, handled in _sam_grad_norm
                    # Let's assume _sam_grad_norm handles adaptive logic correctly for scale
                    e_w = torch.pow(p.data, 2) * p.grad * scale # If adaptive=True in _sam_grad_norm used |w|*g norm
                else:
                    e_w = p.grad * scale.to(p.grad)

                p.data.add_(e_w)


    def _sam_restore_weights(self):
        """Restores weights from stored 'sam_old_p'."""
        # --- Logic exactly as before ---
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                if 'sam_old_p' in state:
                    p.data.copy_(state['sam_old_p'])
                    del state['sam_old_p']


    def _sam_grad_norm(self) -> torch.Tensor:
        """Calculates the norm needed for SAM scaling factor."""
        # --- Logic exactly as before ---
        shared_device = self.param_groups[0]['params'][0].device
        norm = torch.norm(
            torch.stack([
                (torch.abs(p.data) * p.grad if group['adaptive_sam'] else p.grad).norm(p=2).to(shared_device)
                for group in self.param_groups for p in group['params']
                if p.grad is not None
            ]),
            p=2
        )
        return norm.clamp(min=self.eps_sam)


    def step(self, closure: Callable[[], torch.Tensor]) -> Optional[torch.Tensor]:
        """Performs the integrated SAM + OrthoGrad + ScheduleFree step."""

        if not self.train_mode:
            raise Exception("Optimizer must be in train mode (.train()) to call step.")
        if closure is None:
             # Re-emphasize the error if called without closure
            raise ValueError("Closure is required for SAM. Please provide a closure "
                             "that calculates loss and calls backward().")


        # --- Stage 1: SAM Gradient Calculation ---
        loss = closure()
        self._sam_first_step()
        self.zero_grad() # Zero before second backward
        closure()
        self._sam_restore_weights()
        # p.grad now holds SAM gradient

        # --- Stage 2: OrthoGrad ---
        params_with_sam_grad = [p for group in self.param_groups for p in group['params'] if p.grad is not None]
        # Use the corrected orthogonalize method with its specific epsilon
        self._orthogonalize_gradients(params_with_sam_grad, self.eps_ortho)
        # p.grad now holds orthogonalized SAM gradient

        # --- Stage 3 & 4: ScheduleFree Prep & Base Optimizer Step Trigger ---
        temp_grad_assignments = {}
        params_to_restore_grad = []

        for group in self.param_groups:
            lr = group['lr']
            momentum = self.momentum
            weight_decay = self.weight_decay_at_y

            for p in group['params']:
                if p.grad is None: continue # Use the orthogonalized SAM grad from p.grad
                state = self.state[p]

                if 'z' not in state:
                    state['z'] = torch.clone(p.data, memory_format=torch.preserve_format)
                z = state['z'] # Average buffer x_k

                # Apply WD and unextrapolate p (y_k -> x_k)
                if weight_decay != 0.0:
                    z.data.sub_(p.data, alpha=lr * weight_decay)
                    p.data.sub_(p.data, alpha=lr * weight_decay * (1 - momentum))
                p.data.lerp_(z.data, weight= 1 - 1 / momentum) # p is now x_k

                # Store grad needed for base optimizer step on z
                state['current_grad'] = p.grad.clone()

                # Assign grad to z for base_optimizer.step()
                if 'z' in state:
                    z = state['z']
                    if z is not p: # Avoid assigning grad to self if z==p somehow
                       temp_grad_assignments[z] = z.grad # Store original grad of z if any
                       z.grad = state['current_grad'] # Assign the processed grad to z
                       params_to_restore_grad.append(z)


        # --- Execute Base Optimizer Step on 'z' buffers ---
        self.base_optimizer.step() # Reads z.grad and updates z.data

        # --- Restore gradients and clean up ---
        for z in params_to_restore_grad:
             z.grad = temp_grad_assignments.get(z) # Restore original grad

        for group in self.param_groups: # Clean up current_grad state
             for p in group['params']:
                 state = self.state[p]
                 if 'current_grad' in state:
                     del state['current_grad']

        # --- Stage 5: ScheduleFree Final Update ---
        for group in self.param_groups:
             weight_lr_power = self.weight_lr_power
             r = self.r
             k = group['k']
             d = group['d']
             lr = group['lr']
             lr_scaled = lr * d
             group['lr_max'] = max(lr_scaled, group.get('lr_max', 0.0)) # Update lr_max here
             weight = ((k + 1) ** r) * (group['lr_max'] ** weight_lr_power)
             group['weight_sum'] = group.get('weight_sum', 0.0) + weight
             ckp1 = weight / group['weight_sum'] if group['weight_sum'] > 0 else 0

             for p in group['params']:
                 state = self.state[p]
                 if 'z' not in state: continue # Skip if z wasn't initialized

                 z = state['z'] # Contains updated x_k value from base optimizer

                 # p.data currently holds x_k from before the base opt step
                 # Update p (x_k) using the weighted average: x_{k+1} = (1-ckp1)*x_k + ckp1*z
                 p.data.lerp_(z.data, weight=ckp1)

                 # Update z to store the new average x_{k+1} for next step
                 z.data.copy_(p.data)

                 # Calculate next momentum point y_{k+1} and store in p.data
                 p.data.lerp_(z.data, weight=1.0/self.momentum) # p = z/m + p*(1-1/m)

             # Increment ScheduleFree step counter for the group
             group['k'] = k + 1.0

        return loss

    # --- State Dict Handling (Same as before, assuming it's correct) ---
    def state_dict(self):
        wrapper_super_state = super().state_dict()
        wrapper_state = wrapper_super_state['state']
        wrapper_param_groups = wrapper_super_state['param_groups']
        base_state_dict = self.base_optimizer.state_dict()
        combined_state = {
            'wrapper_state': wrapper_state,
            'wrapper_param_groups_state': wrapper_param_groups,
            'base_optimizer_state_dict': base_state_dict,
            'momentum': self.momentum, 'weight_decay_at_y': self.weight_decay_at_y,
            'weight_lr_power': self.weight_lr_power, 'r': self.r, 'rho': self.rho,
            'adaptive_sam': self.adaptive_sam, 'eps_sam': self.eps_sam,
            'eps_ortho': self.eps_ortho, 'train_mode': self.train_mode
        }
        return combined_state

    def load_state_dict(self, state_dict):
        self.momentum = state_dict['momentum']
        self.weight_decay_at_y = state_dict['weight_decay_at_y']
        self.weight_lr_power = state_dict['weight_lr_power']
        self.r = state_dict['r']
        self.rho = state_dict['rho']
        self.adaptive_sam = state_dict['adaptive_sam']
        self.eps_sam = state_dict['eps_sam']
        self.eps_ortho = state_dict.get('eps_ortho', 1e-30) # Add default for backward compat
        self.train_mode = state_dict['train_mode']

        base_state_dict = state_dict['base_optimizer_state_dict']
        self.base_optimizer.load_state_dict(base_state_dict)

        wrapper_state = state_dict['wrapper_state']
        wrapper_param_groups_state = state_dict['wrapper_param_groups_state']
        super_state_to_load = {'state': wrapper_state, 'param_groups': wrapper_param_groups_state}
        super().load_state_dict(super_state_to_load)

        loaded_groups_state = super_state_to_load['param_groups']
        if len(self.param_groups) == len(loaded_groups_state):
             for i, group in enumerate(self.param_groups):
                  loaded_group_state = loaded_groups_state[i]
                  group['k'] = loaded_group_state.get('k', 0.0)
                  group['d'] = loaded_group_state.get('d', 1.0)
                  group['lr_max'] = loaded_group_state.get('lr_max', 0.0)
                  group['weight_sum'] = loaded_group_state.get('weight_sum', 0.0)
                  group['rho'] = loaded_group_state.get('rho', self.rho)
                  group['adaptive_sam'] = loaded_group_state.get('adaptive_sam', self.adaptive_sam)
                  group['eps_sam'] = loaded_group_state.get('eps_sam', self.eps_sam)
                  group['eps_ortho'] = loaded_group_state.get('eps_ortho', self.eps_ortho) # Add eps_ortho
        else:
             print("Warning: Mismatch in number of param groups during load_state_dict.")
        print(f"IntegratedOptimizerWrapper loaded state. Rho: {self.rho}, Momentum: {self.momentum}, TrainMode: {self.train_mode}")


    def zero_grad(self, set_to_none: bool = True) -> None:
        """Zeros the gradients of all parameters managed by this optimizer."""
        # --- Logic exactly as before ---
        super().zero_grad(set_to_none=set_to_none)
        for group in self.param_groups:
            for p in group['params']:
                 if p in self.state and 'z' in self.state[p]:
                      z = self.state[p]['z']
                      if z.grad is not None:
                          if set_to_none:
                              z.grad = None
                          else:
                              z.grad.zero_()