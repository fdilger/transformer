import jax import Array as Tensor
from jax._src.typing import Shape
from typing import Any, Callable, TypeVar, Union, Sequence
from jax._src.prng import PRNGKey

Params = Dict[str, Union[Tensor, Dict[str, Any]]]
