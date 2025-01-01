import jax
import jax.numpy as jnp
from jax import random
from jax import grad


def sigmoid(x):
    return 1 / (1+jnp.exp(-x))


def silu(x):
    return x * sigmoid(x)


def softmax(x):
    exps = jnp.exp(x)
    return exps / exps.sum(axis=-1,keepdims=True)


class FFN:
    def __init__(self,in_dim,hidden_dim,out_dim):
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
    
    def init(self, key):
        keys = jax.random.split(key, 2)
        weights1 = jax.random.normal(keys[0], (self.in_dim, self.hidden_dim))
        weights2 = jax.random.normal(keys[1], (self.hidden_dim, self.out_dim))
        return {'ffnw1': weights1, 'ffnw2': weights2}
        
    def __call__(self, params, x):
        x = jnp.dot(x, params['ffnw1'])
        x = silu(x)
        x = jnp.dot(x,params['ffnw2'])
        return silu(x)+x


class LayerNorm:
    def __init__(self,d_model,eps=1e-05):
        self.eps = eps
        self.d_model = d_model
        
    def init(self,key):
        gamma = jnp.ones((self.d_model,)) 
        beta = jnp.zeros((self.d_model,))
        return {'beta': beta, 'gamma': gamma}
    
    def __call__(self,params,x):
        m = x.mean(axis=-1,keepdims=True)
        d = x - m
        v = jnp.square(d).mean(axis=-1,keepdims=True)
        return d / jnp.sqrt(v+self.eps)*params['gamma']+params['beta']
    
        
class Attention:
    def __init__(self,n_heads,d_model):
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model//n_heads
        
    def init(self,key):
        d = self.d_head
        keys = jax.random.split(key, 4)
        scale = jnp.sqrt(1/d_model)
        wq = jax.random.normal(keys[0],(self.n_heads,d_model,d)) * scale
        wk = jax.random.normal(keys[1],(self.n_heads,d_model,d)) * scale
        wv = jax.random.normal(keys[2],(self.n_heads,d_model,d)) * scale
        wo = jax.random.normal(keys[3],(self.n_heads,d,d_model)) * scale
        return {'query_proj': wq, 'key_proj': wk, 'value_proj': wv, 'out_proj': wo}
    
    def __call__(self,params,x):
        queries = jnp.einsum('btc,ncj-> bntj',x,params['query_proj'])
        keys = jnp.einsum('btc,ncj->bntj',x,params['key_proj'])
        values = jnp.einsum('btc,ncj->bntj',x,params['value_proj'])
        att = jnp.einsum('bntj,bnsj->bnts',queries,keys)
        att_scaled = att / jnp.sqrt(self.d_model)
        att_scores = softmax(att_scaled)
        att_values = jnp.einsum('bnts,bnsv-> bntv',att_scores,values)
        att_output = jnp.einsum('bntv,nvk->btk', att_values,params['out_proj'])
        return att_output+x
    
        
        
class TransformerBlock:
    def __init__(self,d_model,hidden_dim,n_heads):
        self.layers = {'norm1': LayerNorm(d_model),
                       'att'  : Attention(n_heads,d_model),
                       'norm2': LayerNorm(d_model),
                       'ffn'  : FFN(d_model,hidden_dim,d_model)}

    def init(self):
        params = {}
        key = random.PRNGKey(0)
        for name,layer in self.layers.items():
            params[name] = layer.init(key)
        return params
            
    def __call__(self,params,x):
        for name,layer in self.layers.items():
            x = layer(params[name],x)
        return x

# small test         
d_model = 5
hidden_dim = 5
n_heads = 3
b_size = 2
s_len = 3

input_shape = (2,3,5)
key = jax.random.PRNGKey(42)
data = jax.random.normal(key,input_shape)
block = TransformerBlock(d_model,hidden_dim,n_heads)
params = block.init()
print(block(params,data))


        
    
