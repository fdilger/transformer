import jax
import jax.numpy as jnp

class Embeddings:
    def __init__(self,d_model,v_size):
        self.d_model=d_model
        self.v_size = v_size

    def init(self,key):
        return {'w_embed' : jax.random.normal(key,(self.v_size,self.d_model))*1/jnp.sqrt(self.d_model)}

    def __call__(self,params,token_ids):
        return params['w_embed'][token_ids]
