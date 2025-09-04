from torch.optim import Optimizer


class CombinedOptimizer(Optimizer):
    """Combines multiple optimizers for different parameter groups."""
    
    def __init__(self, optimizers):
        """
        Initialize combined optimizer.
        
        Args:
            optimizers: List of optimizer instances
        """
        assert all(len(opt.param_groups) == 1 for opt in optimizers)
        self.optimizers = optimizers
        self.param_groups = [pg for opt in self.optimizers for pg in opt.param_groups]
        self.base_lrs = [opt.param_groups[0]['lr'] for opt in self.optimizers]
    
    def step(self):
        """Perform optimization step for all optimizers."""
        for optimizer in self.optimizers:
            optimizer.step()
    
    def zero_grad(self, **kwargs):
        """Zero gradients for all optimizers."""
        for optimizer in self.optimizers:
            optimizer.zero_grad(**kwargs)
    
    def scale_lrs(self, lr_scale):
        """Scale learning rates for all optimizers."""
        for base_lr, opt in zip(self.base_lrs, self.optimizers):
            opt.param_groups[0]['lr'] = base_lr * lr_scale
    
    def state_dict(self):
        """Get state dict from all optimizers."""
        return [optimizer.state_dict() for optimizer in self.optimizers]
    
    def load_state_dict(self, state_dicts):
        """Load state dict for all optimizers."""
        for optimizer, state_dict in zip(self.optimizers, state_dicts):
            optimizer.load_state_dict(state_dict)
    
 