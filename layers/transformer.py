import jax
import jax.numpy as jnp

from ..normalization.layer_norm import LayerNorm
from ..layers.attention import LatentAttention,CrossAttention
from ..layers.linear import FFN
from ..utils.utils import softmax

class TransformerDecoderBlock:
    def __init__(self,d_model,hidden_dim,n_heads,mask,d_latent = 0):
        att = LatentAttention(n_heads,d_model,d_latent,mask) if d_latent else Attention(n_heads,d_model,mask)
        self.layers = {'norm1': LayerNorm(d_model),
                       'att'  : att,
                       'norm2': LayerNorm(d_model),
                       'ffn'  : FFN(d_model,hidden_dim,d_model)}

    def init(self,key):
        params = {}
        for name,layer in self.layers.items():
            params[name] = layer.init(key)
        return params
            
    def __call__(self,params,x):
        for name,layer in self.layers.items():
            x = layer(params[name],x)
        return x


class TransformerEncoderBlock:
    def __init__(self,d_model,hidden_dim,n_heads,mask,d_k,d_v):
        att = CrossAttention(n_heads,d_model,d_k,d_v,mask)
        self.layers = {'norm1': LayerNorm(d_model),
                       'att'  : att,
                       'norm2': LayerNorm(d_model),
                       'ffn'  : FFN(d_model,hidden_dim,d_model)}

    def init(self,key):
        params = {}
        for name,layer in self.layers.items():
            params[name] = layer.init(key)
        return params
            
    def __call__(self,params,x,k,v):
        for name,layer in self.layers.items():
            if name == 'att':
                x = layer(params[name],x,k,v)
            else:
                x = layer(params[name],x)
        return x
