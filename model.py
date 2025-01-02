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
        y = jnp.dot(x, params['ffnw1'])
        y = silu(y)
        y = jnp.dot(y,params['ffnw2'])
        return silu(y)+x
    
    def __len__(self):
        return self.in_dim*self.hidden_dim+self.hidden_dim*self.out_dim


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
    
    def __len__(self):
        return 2*d_model
        
    
        
class Attention:
    def __init__(self,n_heads,d_model):
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model//n_heads
        
    def init(self,key):
        d = self.d_head
        keys = jax.random.split(key, 4)
        scale = jnp.sqrt(1/d_model)
        wq = jax.random.normal(keys[0],(self.n_heads,self.d_model,d)) * scale
        wk = jax.random.normal(keys[1],(self.n_heads,self.d_model,d)) * scale
        wv = jax.random.normal(keys[2],(self.n_heads,self.d_model,d)) * scale
        wo = jax.random.normal(keys[3],(self.n_heads,d,self.d_model)) * scale
        return {
                'query_proj': wq,
                'key_proj': wk,
                'value_proj': wv,
                'out_proj': wo
               }
    
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
    
    def __len__(self):
        return 4*self.d_model*self.d_model


class LatentAttention:
    # causal masking
    # TODO: implement decoupled rope
    def __init__(self,n_heads,d_model,d_latent):
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model//n_heads
        self.d_latent = d_latent // n_heads
        
    def init(self,key):
        keys  = jax.random.split(key, 6)
        scale = jnp.sqrt(1/d_model)
        
        wdkv =  jax.random.normal(keys[0],(self.n_heads,self.d_model, self.d_latent)) * scale
        wdq  =  jax.random.normal(keys[1],(self.n_heads,self.d_model, self.d_latent)) * scale
        wuk  =  jax.random.normal(keys[2],(self.n_heads,self.d_latent,self.d_head))   * scale
        wuv  =  jax.random.normal(keys[3],(self.n_heads,self.d_latent,self.d_head))   * scale
        wuq  =  jax.random.normal(keys[4],(self.n_heads,self.d_latent,self.d_head))   * scale
        wo   =  jax.random.normal(keys[5],(self.n_heads,self.d_head,  self.d_model))  * scale
        return {
                'kv_down_proj' : wdkv,
                'q_down_proj'  : wdq,
                'query_up_proj': wuq,
                'key_up_proj'  : wuk,
                'value_up_proj': wuv,
                'out_proj'     : wo
               }
    
    def __call__(self,params,x):
        
        latent_q = jnp.einsum('btc,ncj->bntj',x,params['q_down_proj'])
        latent_kv = jnp.einsum('btc,ncj->bntj',x,params['kv_down_proj'])

        queries = jnp.einsum('bntj,njs->bnts',latent_q ,params['query_up_proj'])
        keys    = jnp.einsum('bntj,njs->bnts',latent_kv,params['key_up_proj'])
        values  = jnp.einsum('bntj,njs->bnts',latent_kv,params['value_up_proj'])
        
        att = jnp.einsum('bntj,bnsj->bnts',queries,keys)
        att_scaled = att / jnp.sqrt(self.d_model)
        att_scores = softmax(att_scaled)
        att_values = jnp.einsum('bnts,bnsv-> bntv',att_scores,values)
        att_output = jnp.einsum('bntv,nvk->btk', att_values,params['out_proj'])
        return att_output+x
    
    def __len__(self):
        return 4*self.d_model*self.d_model

class LatentCompression:
    def __init__(self,d_model,ratio):
        self.d_model=d_model
        self.d_latent=d_model//ratio

    def init(self,key):
        keys  = jax.random.split(key, 2)
        scale = jnp.sqrt(1/d_model)
        down = jax.random.normal(keys[0],(self.dmodel,self.d_latent)) * scale
        up = jax.random.normal(keys[1],(self.dmodel,self.d_latent)) * scale
        return {'up': up, 'down': down}
    
    def __call__(self,params,x):
        latent = jnp.einsum('bts,sj->btj',x,params['down'])
        latent = silu(latent)
        return silu(jnp.einsum('btj,js->bts',x,params['up']))

    
class TransformerBlock:
    def __init__(self,d_model,hidden_dim,n_heads,d_latent = 0):
        att = LatentAttention(n_heads,d_model,d_latent) if d_latent else Attention(n_heads,d_model)
        self.layers = {'norm1': LayerNorm(d_model),
                       'att'  : att,
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
    
    def __len__(self):
        return sum(len(layer) for layer in self.layers.values())


class Patching:
    def __init__(self,model,vocab):
        self.model = model
        self.vocab = vocab
    def init(self):
        return model.init()
        
    def byte_entropy(x,params):
        probs = model(x,params)
        return sum(probs[vocab[x]]*log(probs[v]) for v in vocab.items())
    
    def local_entropy_patching(x,omega,params):
        ent = [byte_entropy(xi,params) for xi in x]
        return [1 if xi-xprev>omega else 0 for xi,xprev in zip(ent,[0]+ent[:-1])]
    
    def global_entropy_patching(x,omega,params):
        ent = [byte_entropy(xi,params) for xi in x]
        return [1 if xi>omega else 0 for xi in ent]
    











    

# deepseek-v2
# d_latent = 4*(d_model // n_heads) 
    
# small test         
d_model = 6
hidden_dim = 5
n_heads = 3
b_size = 2
s_len = 3
d_latent = 3
input_shape = (2,3,6)
key = jax.random.PRNGKey(42)
data = jax.random.normal(key,input_shape)
block = TransformerBlock(d_model,hidden_dim,n_heads,d_latent=d_latent)
print(len(block))
params = block.init()
print(block(params,data).shape)


        
    
