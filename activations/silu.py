from ..utils.utils import sigmoid
from .mtypes import Tensor, Shape, Any, Callable, Params, PRNGKey



def silu(x : Tensor) -> Tensor:
    return x * sigmoid(x)
