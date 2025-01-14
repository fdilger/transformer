import jax
import jax.numpy as jnp
from ..layers.embeddings import Embeddings
from ..model import TransformerDecoderBlock,TransformerEncoderBlock
from ..layers.linear import PredictionHead
from ..utils.utils import softmax

class ByteEncoder:

    def __init__(self,d_model,v_size,mask,n_heads,mask,d_latent):
        self.d_model = d_model
        self.v_size = v_size
        self.hidden_dim = 4*d_model
        
        self.init_embed = Embeddings(d_model,v_size)}
        
        # transformer for entropy estimation
        # maybe this one should live outside this class
        self.entropy_layers = {
            't_block'+str(i) : TransformerDecoderBlock(d_model,hidden_dim,n_heads,mask,d_latent)
            for i in range(n_entropy_layers)
        }

        self.head = PredictionHead(d_model,v_size)

        # transformer for computing patch representations
        # d_k and d_v have the same embedding dimensoin as patches since patch embeddings result from pooled byte embeddings
        # however due to pooling seq_lens will be different
        # does there need to be a separate embedding AFTER pooling, why?
        # implement patch mask
        patch_mask = 
        
        self.patch_layers = {
            't_block'+str(i) : TransformerEncoderBlock(d_model,hidden_dim,n_heads,mask,d_model,d_model)
            for i in range(n_patch_layers)
        }
        
        
    def init(self,key):
        embed = self.init_embed.init(key)
        ent = {name: block.init(key) for name,block in self.entropy_layers.items()}
        pat = {name: block.init(key) for name,block in self.patch_layers.items()}
        h = self.head.init(key)
        return {
            'init_embed' : embed,
            'entropy_transformer' : ent,
            'pred_head' : h
            'patch_transformer' : pat
        }

    def get_probs(self,params,x):
        # operates on the embeddings
        
    
    def byte_entropy(self,params,x):
        # operates on last hidden rep
        logits = self.head(params['pred_head'],x)
        probs = softmax(logits)
        return -jnp.sum((probs * jnp.log(probs+1e-10)),axis=-1)

    def local_entropy_patching(self,x,omega,params):
        # there must be a better way 
        entropy = byte_entropy(self,params,x)
        res = []
        ks = []
        for ent in entropy:
            r = []
            k = -1
            for (xi,xprev) in zip(ent,[0]+ent[:-1]):
                if xi-xprev>omega:
                    k+=1
                    r.append(k)
                else:
                    r.append(k)
            res.append(r)
            ks.append(k)
        return ks,res
    
    def global_entropy_patching(self,params,omega,x):
        ent = [byte_entropy(params,xi) for xi in x]
        return [1 if xi>omega else 0 for xi in ent]
    
    def max_pool(self,params,x):
        # segment_max doesn't work with batches so we'll do this terribleness until i bother to write a good segment_max
        num_segments,segments = self.local_entropy_patching(params,self.omega,x) # segments with model
        results = []
        for i,seq in enumerate(x):
            results.append(jax.ops.segment_max(seq,segments[i],num_segments = num_segments[i]))
        return jnp.stack(results)
    
        
    def __call__(self,params,x):
        # apply transformer with cross attention
        x = self.init_embed(params['init_embed'],x)
        
        for name,layer in self.entropy_layers.items:
            x = layer(params['entropy_transformer'][name],x)

        patches = self.max_pool(params['patch_model'],x)
        
        patches = self.embed(params['patch_embed'],patches)
        rs = c_model(params['rep_model'],patches,x,x) # cross attention to original last byte representation, needs mask 
        return rs
        # for patch cross attention q = patches, kv= original byte embeddings
