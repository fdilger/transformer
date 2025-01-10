import jax
import jax.numpy as jnp

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
