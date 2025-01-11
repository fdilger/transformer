import jax
import jax.numpy as jnp
from jax import random
from ..utils.utils import softmax

class Attention:
    def __init__(self,n_heads,d_model,mask):
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model//n_heads
        self.mask = mask
        
    def init(self,key):
        d = self.d_head
        keys = jax.random.split(key, 4)
        scale = jnp.sqrt(1/self.d_model)
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
        att_masked = jnp.where(self.mask, -jnp.inf, att_scaled)
        att_scores = softmax(att_masked)
        att_values = jnp.einsum('bnts,bnsv-> bntv',att_scores,values)
        att_output = jnp.einsum('bntv,nvk->btk', att_values,params['out_proj'])
        return att_output+x
    

    
class LatentAttention:
    # causal masking
    # TODO: implement decoupled rope
    def __init__(self,n_heads,d_model,d_latent,mask):
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model//n_heads
        self.d_latent = d_latent // n_heads
        self.mask = mask
        
    def init(self,key):
        keys  = jax.random.split(key, 6)
        scale = jnp.sqrt(1/self.d_model)
        
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
        att_masked = jnp.where(self.mask, -jnp.inf, att_scaled)
        att_scores = softmax(att_masked)
        att_values = jnp.einsum('bnts,bnsv-> bntv',att_scores,values)
        att_output = jnp.einsum('bntv,nvk->btk', att_values,params['out_proj'])
        return att_output+x
    

class CrossAttention:
    
     def __init__(self,n_heads,d_model,d_k,d_v,mask):
         self.d_model = d_model
         self.n_heads = n_heads
         self.d_head = d_model//n_heads
         self.d_k = d_k
         self.d_v = d_v
         self.mask = mask
         
     def init(self,key):
         d = self.d_head
         keys = jax.random.split(key, 4)
         scale = jnp.sqrt(1/self.d_model)
         wq = jax.random.normal(keys[0],(self.n_heads,self.d_model,self.d_head))    * scale
         wk = jax.random.normal(keys[1],(self.n_heads,self.d_k,    self.d_head))    * scale
         wv = jax.random.normal(keys[2],(self.n_heads,self.d_v,    self.d_head))    * scale
         wo = jax.random.normal(keys[3],(self.n_heads,d,           self.d_model))   * scale
         return {
             'query_proj': wq,
             'key_proj': wk,
             'value_proj': wv,
             'out_proj': wo
         }
     
     def __call__(self,params,q,k,v):
         queries = jnp.einsum('btc,ncj-> bntj',q,params['query_proj'])
         keys = jnp.einsum('btc,ncj->bntj',k,params['key_proj'])
         values = jnp.einsum('btc,ncj->bntj',v,params['value_proj'])
         att = jnp.einsum('bntj,bnsj->bnts',queries,keys)
         # scale by head dim
         att_scaled = att / jnp.sqrt(self.d_head)
         att_masked = jnp.where(self.mask, -jnp.inf, att_scaled)
         att_scores = softmax(att_masked)
         att_values = jnp.einsum('bnts,bnsv-> bntv',att_scores,values)
         att_output = jnp.einsum('bntv,nvk->btk', att_values,params['out_proj'])
         return att_output+q
    
