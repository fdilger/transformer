from ..utils.utils import sigmoid

def silu(x):
    return x * sigmoid(x)
