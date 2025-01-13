import jax
import jax.numpy as jnp
from jax import grad
import optax

from .layers.embeddings import Embeddings
from .layers.attention import LatentAttention,CrossAttention
from .normalization.layer_norm import LayerNorm
from .layers.linear import FFN,PredictionHead
from .utils.utils import softmax, log_softmax,crossentropy,count_params
from .layers.rope import Rope


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

    

# mask = (segments[:, :, None] == segments[:, None, :]) patch-mask
# first compute the byte representations meant for the entropy model
# pass these to entropy model for patches
# max pool patches
# project patches to cross attention model with cross attention to original byte representations
# -> remove embedding from the transformer class

class TransformerDecoder:
    def __init__(self,n_layers,d_hidden,d_model,n_heads,v_size,mask,d_latent = 0):
        self.n_layers = n_layers
        self.d_hidden = d_hidden
        self.layers = {'embed' : Embeddings(d_model,v_size)}
        for i in range(n_layers):
            self.layers['block'+str(i)] = TransformerDecoderBlock(d_model,d_hidden,n_heads,mask,d_latent=d_latent)
        self.layers['pred_head'] = PredictionHead(d_model,v_size)
        
    def init(self,key):
        keys = jax.random.split(key,self.n_layers+1)
        params = {}
        for i,(name,layer) in enumerate(self.layers.items()):
            params[name] = layer.init(keys[i])
        return params
    
    def __call__(self,params,x):
        for name,layer in self.layers.items():
            x = layer(params[name],x)
        return x


class TransformerEncoder:
    def __init__(self,n_layers,d_hidden,d_model,d_k,d_v,n_heads,v_size,mask):
        self.n_layers = n_layers
        self.d_hidden = d_hidden
        self.layers = {'embed' : Embeddings(d_model,v_size)}
        for i in range(n_layers):
            self.layers['block'+str(i)] = TransformerEncoderBlock(d_model,d_hidden,n_heads,mask,d_k,d_v)
        self.layers['pred_head'] = PredictionHead(d_model,v_size)
        
    def init(self,key):
        keys = jax.random.split(key,self.n_layers+1)
        params = {}
        for i,(name,layer) in enumerate(self.layers.items()):
            params[name] = layer.init(keys[i])
        return params
    
    def __call__(self,params,x,k,v):
        for name,layer in self.layers.items():
            # this is terrrible
            if name[0:5] == 'block':
                x = layer(params[name],x,k,v)
            else:
                x = layer(params[name],x)
        return x



# deepseek-v2
# d_latent = 4*(d_model // n_heads) 
