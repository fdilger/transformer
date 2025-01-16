import jax
import jax.numpy as jnp
from .mtypes import Tensor, Shape, Any, Callable, Params, PRNGKey


class LayerNorm:
    def __init__(self, embed_dim : int, eps=1e-05 : float) -> None:
        self.eps = eps
        self.embed_dim

        
    def init(self, key : PRNGKey) -> Params:
        gamma = jnp.ones((self.embed_dim,)) 
        beta  = jnp.zeros((self.embed_dim,))
        return {'beta': beta, 'gamma': gamma}
    
    def __call__(self, params : Params, x : Tensor) -> Tensor:
        mean       = x.mean(axis=-1,keepdims=True)
        centered   = x - mean
        variance   = jnp.square(centered).mean(axis=-1,keepdims=True)
        normalised = centered / jnp.sqrt(variance+self.eps)
        return normalised * params['gamma'] + params['beta']

    
