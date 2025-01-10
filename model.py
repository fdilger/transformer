import jax
import jax.numpy as jnp
from jax import grad
import optax


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

    
class ByteEncoder:
    # operates on byte embeddings
    def __init__(self,c_model,e_model,omega):
        self.embed = Embeddings(c_model.d_model,tokenizer.v_size)
        self.c_model = model
        self.e_model = model
        self.omega = omega
        
    def init(self,key):
        patch_embed = jax.random.normal(key,(self.c_model.d_model,self.d_embed)
        return {'patch_model' : self.e_model.init(key), 'rep_model': self.c_model.init(key), 'patch_embed' : patch_embed}

    def byte_entropy(x,params):
        probs = model(params,x)
        return -jnp.sum((probs * jnp.log(probs+1e-10)),axis=-1)

    def local_entropy_patching(x,omega,params):
        ent = [byte_entropy(xi,params) for xi in x]
        res = []
        k = -1
        for (xi,xprev) in zip(ent,[0]+ent[:-1]):
            if xi-xprev>omega:
                k+=1
                res.append(k)
            else:
                res.append(k)
        return k,res
    
    def global_entropy_patching(x,omega,params):
        ent = [byte_entropy(xi,params) for xi in x]
        return [1 if xi>omega else 0 for xi in ent]
    
    def max_pool(x):
        # segment_max doesn't work with batches so we'll do this terribleness until i bother to write a good segment_max
        num_segments,segments = patching.local_entropy_patching(x,self.omega,params) # segments with model
        results = []
        for i,seq in enumerate(x):
            results.append(jax.ops.segment_max(seq,segments[i],num_segments = num_segments[i]))
        return jnp.stack(results)
        
        
    def __call__(self,x):
        # apply transformer with cross attention
        patches = max_pool(x)
        for layer in layers.items():
            patches = layer(patches)
        return x
        # for patch cross attention q = patches, kv= original byte embeddings
    
# mask = (segments[:, :, None] == segments[:, None, :]) patch-mask
# first compute the byte representations meant for the entropy model
# pass these to entropy model for patches
# max pool patches
# project patches to cross attention model with cross attention to original byte representations
# -> remove embedding from the transformer class

class Transformer:
    def __init__(self,n_layers,d_hidden,d_model,n_heads,v_size,mask,d_latent = 0):
        self.n_layers = n_layers
        self.d_hidden = d_hidden
        self.layers = {'embed' : Embeddings(d_model,v_size)}
        for i in range(n_layers):
            self.layers['block'+str(i)] = TransformerBlock(d_model,d_hidden,n_heads,mask,d_latent=d_latent)
        self.layers['pred_head'] = PredictionHead(d_model,v_size)
        
    def init(self,key):
        keys = jax.random.split(key,self.n_layers+1))
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
