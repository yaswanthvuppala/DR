import sys

# Lazy exports for the models package (PEP 562)

def __getattr__(name):
    if name == 'DREarlyXAI':
        from .proposed_model import DREarlyXAI
        return DREarlyXAI
    elif name == 'load_proposed_model':
        from .proposed_model import load_proposed_model
        return load_proposed_model
    elif name == 'CORALLoss':
        from .ordinal_loss import CORALLoss
        return CORALLoss
    elif name == 'EMDLoss':
        from .ordinal_loss import EMDLoss
        return EMDLoss
    elif name == 'CombinedLoss':
        from .ordinal_loss import CombinedLoss
        return CombinedLoss
    
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = [
    'DREarlyXAI',
    'load_proposed_model',
    'CORALLoss',
    'EMDLoss',
    'CombinedLoss',
]
