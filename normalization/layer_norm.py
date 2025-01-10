import jax
import jax.numpy as jnp

class LayerNorm:
    def __init__(self,d_model,eps=1e-05):
        self.eps = eps
        self.d_model = d_model
        self.n_params = 2*d_model
        
    def init(self,key):
        gamma = jnp.ones((self.d_model,)) 
        beta = jnp.zeros((self.d_model,))
        return {'beta': beta, 'gamma': gamma}
    
    def __call__(self,params,x):
        m = x.mean(axis=-1,keepdims=True)
        d = x - m
        v = jnp.square(d).mean(axis=-1,keepdims=True)
        return d / jnp.sqrt(v+self.eps)*params['gamma']+params['beta']

    
