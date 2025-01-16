import jax
import jax.numpy as jnp
from ..activations.silu import silu
from .mtypes import Tensor, Shape, Any, Callable, Params, PRNGKey



class Embedding:
    def __init__(self, in_dim : int, out_dim : int) -> None:
        self.in_dim = in_dim
        self.out_dim = out_dim
        
    def init(self, key : PRNGKey) -> Params:
        scale = 1/self.in_dim
        return {'w_embed': jax.random.normal(key,(self.in_dim,self.out_dim))}
    
    def __call__(self,params : Params,x : Tensor) -> Tensor:
        return jnp.einsum('btc,cj->btj', x, params['w_embed'])
    

    
class FFN:
    def __init__(self,in_dim : int, ff_dim : int ,out_dim : int) -> None:
        self.in_dim = in_dim
        self.ff_dim = ff_dim
        self.out_dim = out_dim
    
    def init(self, key : PRNGKey) -> Params:
        k1, k2 = jax.random.split(key, 2)
        scale  = 1 / jnp.sqrt(self.ff_dim)
        
        ffn_w1 = jax.random.normal(k1, (self.in_dim, self.ff_dim))  * scale 
        ffn_w2 = jax.random.normal(k2, (self.ff_dim, self.out_dim)) * scale
        return {'ffn_w1': ffn_w1, 'ffn_w2': ffn_w2}
        
    def __call__(self, params : Params, x : Tensor):
        y = jnp.einsum('btc,cj->btj', x, params['ffn_w1'])
        y = silu(y)
        y = jnp.einsum('btc,cj->btj', y, params['ffn_w2'])
        return silu(y)+x


class LatentCompression:
    def __init__(self,d_model,ratio):
        self.d_model=d_model
        self.d_latent=d_model//ratio

    def init(self,key):
        keys  = jax.random.split(key, 2)
        scale = jnp.sqrt(1/self.d_model)
        down = jax.random.normal(keys[0],(self.d_model,self.d_latent)) * scale
        up = jax.random.normal(keys[1],(self.d_model,self.d_latent)) * scale
        return {'up': up, 'down': down}
    
    def __call__(self,params,x):
        latent = jnp.einsum('bts,sj->btj',x,params['down'])
        latent = silu(latent)
        return silu(jnp.einsum('btj,js->bts',x,params['up']))

    
