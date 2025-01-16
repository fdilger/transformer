import jax
import jax.numpy as jnp
from ..activations.silu import silu

class PredictionHead:
    def __init__(self,d_model,v_size):
        self.d_model = d_model
        self.v_size = v_size
        
    def init(self,key):
        keys = jax.random.split(key,2)
        w = jax.random.normal(keys[0],(self.d_model,self.v_size))*1/(jnp.sqrt(self.d_model))
        return {'w':w}
    def __call__(self,params,x):
        return jnp.einsum('btc,cj->btj',x,params['w'])

    
class FFN:
    def __init__(self,in_dim,hidden_dim,out_dim):
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
    
    def init(self, key):
        keys = jax.random.split(key, 2)
        weights1 = jax.random.normal(keys[0], (self.in_dim, self.hidden_dim))*1/(jnp.sqrt(self.hidden_dim))
        weights2 = jax.random.normal(keys[1], (self.hidden_dim, self.out_dim))*1/(jnp.sqrt(self.hidden_dim))
        return {'ffnw1': weights1, 'ffnw2': weights2}
        
    def __call__(self, params,x):
        y = jnp.einsum('btc,cj->btj',x, params['ffnw1'])
        y = silu(y)
        y = jnp.einsum('btc,cj->btj',y, params['ffnw2'])
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

    
y
