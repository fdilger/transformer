import jax
import jax.numpy as jnp
from jax import random
from jax import grad
import optax

def sigmoid(x):
    return 1 / (1+jnp.exp(-x))


def silu(x):
    return x * sigmoid(x)


def softmax(x):
    exps = jnp.exp(x)
    return exps / exps.sum(axis=-1,keepdims=True)

def log_softmax(x):
    exps = jnp.exp(x)
    return jnp.log(exps / exps.sum(axis=-1,keepdims=True))


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
        

class Rope:

    def __init__(self,d_head,seq_len):
        self.d_head = d_head
        self.seq_len = seq_len
        
    def init(self):
        theta = 10000 ** (-2*(jnp.arange(self.d_head//2))/self.d_head) # (self.dhead//2,)
        positions = jnp.arange(self.seq_len).reshape(-1, 1) # (seq_len,1)
        cos = jnp.cos(positions * theta)
        sin = jnp.sin(positions*theta)
        return cos,sin
    
    def __call__(self,cos,sin,x):
        x1, x2 = jnp.split(x, 2, axis=-1)  # (batch_size, seq_len, dim // 2)
        cos = cos[None, :, :]
        sin = sin[None, :, :]
        x_rotated = jnp.concatenate([
            x1 * cos - x2 * sin,  
            x1 * sin + x2 * cos   
        ], axis=-1)
        return x_rotated
        
        
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
    
    def __len__(self):
        return 4*self.d_model*self.d_model


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
    
    def __len__(self):
        return 4*self.d_model*self.d_model

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

    
class TransformerBlock:
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
    
    def __len__(self):
        return sum(len(layer) for layer in self.layers.values())


class PredictionHead:
    def __init__(self,d_model,v_size):
        self.d_model = d_model
        self.v_size = v_size
        
    def init(self,key):
        keys = jax.random.split(key,2)
        w = jax.random.normal(keys[0],(self.d_model,self.v_size))*1/jnp.sqrt(self.d_model)
        b = jax.random.normal(keys[1],(self.v_size,))*1/jnp.sqrt(self.d_model)
        return {'w':w,'b':b}
    def __call__(self,params,x):
        return jnp.einsum('btc,cj->btj',x,params['w']) + params['b']
         
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
    

class ByteTokenizer:
    def __init__(self,data,d_model):
        unique_bytes = sorted(set(data))
        v_size = len(unique_bytes)
        self.byte_to_id = {b: i for i,b in enumerate(unique_bytes)}
        self.id_to_byte = {i: b for i,b in enumerate(unique_bytes)}
    
    def encode(self,params,byte):
        return byte_to_id[byte]
    
    def decode(self,params,token):
        return id_to_byte[token]

class Embeddings:
    def __init__(self,d_model,v_size):
        self.d_model=d_model
        self.v_size = v_size

    def init(self,key):
        return {'w_embed' : jax.random.normal(key,(self.v_size,self.d_model))*1/jnp.sqrt(self.d_model)}

    def __call__(self,params,token_ids):
        return params['w_embed'][token_ids]

    
class Transformer:
    def __init__(self,n_layers,d_hidden,d_model,n_heads,v_size,mask,d_latent = 0):
        self.n_layers = n_layers
        self.d_hidden = d_hidden
        self.layers = {'embed' : Embeddings(d_model,v_size)}
        for i in range(n_layers):
            self.layers['block'+str(i)] = TransformerBlock(d_model,d_hidden,n_heads,mask,d_latent=d_latent)
        self.layers['pred_head'] = PredictionHead(d_model,v_size)
        
    def init(self,key):
        keys = jax.random.split(key,self.n_layers+2)
        params = {}
        for i,(name,layer) in enumerate(self.layers.items()):
            params[name] = layer.init(keys[i])
        return params
    
    def __call__(self,params,x):
        for name,layer in self.layers.items():
            x = layer(params[name],x)
        return x


class DataSet:
    
    def __init__(self,tokens,batch_size,seq_len):
        self.seq_len = seq_len
        self.batch_size = batch_size
        n_seqs = len(tokens) // seq_len
        self.data = tokens[:n_seqs*seq_len].reshape(-1,seq_len)
        
    def shuffle(self,key):
        self.data = jax.random.permutation(key,self.data)
        n_batches = len(self.data) // batch_size
        self.data = self.data[:n_batches*batch_size].reshape(-1,self.batch_size,self.seq_len)
       
    def __iter__(self):
        for batch in self.data:
            yield batch
        

            
def crossentropy(targets,logits):
    probs = softmax(logits)
    logprobs = jnp.log(probs)
    loss = -jnp.mean(logprobs[...,targets])
    return loss

# deepseek-v2
# d_latent = 4*(d_model // n_heads) 

# small test
seqlen = 3
causal_mask = jnp.triu(jnp.ones((seqlen, seqlen)), k=1).astype(bool)
key = jax.random.PRNGKey(42)
token_ids = jnp.ones((2, seqlen), dtype=jnp.int32)  # [batch=2, seq_len=3]
token_ids = jnp.array([
    [0, 1, 2],  # Sequence 1
    [3, 4, 5]   # Sequence 2
])
targets = jnp.array([
    [1, 2, 3],
    [4, 5, 6]
])
x = jnp.array([
    [0, 1,2],
    [3, 4,5]
])
transformer = Transformer(
    n_layers=2, 
    d_hidden=8, 
    d_model=16, 
    n_heads=4,
    v_size=256,  
    mask=causal_mask,
    d_latent=4
)
print('__________ model parameters: ')
params = transformer.init(key)
print(params)
print('_____________________________')

def forward_loss(params,model,batch,targets):
    logits = transformer(params,x)
    logits = logits - jnp.max(logits, axis=-1, keepdims=True)
    loss = crossentropy(targets,logits)
    return loss


forward_loss(params,transformer,x,targets)

gradient = jax.value_and_grad(forward_loss)

def step(params,opt_state,batch,targets,model):
    loss,grads = gradient(params,model,batch,targets)
    updates, new_opt_state = optimizer.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state,loss


optimizer = optax.adam(
    learning_rate=1e-4, 
    b1=0.9,             
    b2=0.999,           
    eps=1e-8            
)
opt_state = optimizer.init(params)

optimizer = optax.sgd(
    learning_rate=0.1  # Tune this to your liking!
)

# Initialize it with your params!
opt_state = optimizer.init(params)

for epoch in range(1):
    params,opt_state,loss = step(params,opt_state,x,targets,transformer)
    print(loss)
print(softmax(transformer(params,x)))

r = Rope(2,3)
cos,sin = r.init()
print(r(cos,sin,jnp.array([[[1,2],[2,3],[4,5]]])))
## TODO: ##
# - implement different FFN
# - add jax.lax.scan optimisations
# - 
#
#
#
#
#
