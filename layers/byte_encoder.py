import jax
import jax.numpy as jnp

from ..layers.embeddings import Embeddings
from ..layers.transformer import TransformerDecoderBlock,TransformerEncoderBlock
from ..layers.linear import PredictionHead
from ..utils.utils import softmax

class ByteEncoder:

    def __init__(self,d_model,v_size,n_heads,mask,d_latent,omega,seq_len):
        self.d_model = d_model
        self.v_size = v_size
        self.hidden_dim = 4*d_model
        self.omega = omega
        self.init_embed = Embeddings(d_model,v_size)
        self.seq_len = seq_len
        n_entropy_layers = 2
        n_patch_layers = 2
        hidden_dim = 4*d_model
        
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
        #patch_mask = 
        
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
            'pred_head' : h,
            'patch_transformer' : pat
        }

   
    def byte_entropy(self,params,x):
        # operates on last hidden rep
        logits = self.head(params['pred_head'],x)
        probs = softmax(logits)
        return -jnp.sum((probs * jnp.log(probs+1e-10)),axis=-1)

    def local_entropy_patching(self,params,omega,x):
        # there must be a better way 
        entropy = self.byte_entropy(params,x)
        res = []
        ks = []
        for ent in entropy:
            r = []
            k = -1
            if ent[0] > omega:
                k+=1
                r.append(k)
            else:
                r.append(k)
            for (xi,xprev) in zip(ent[1:],ent[:-1]):
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
            
            pooled = jax.ops.segment_max(seq,jnp.array(segments[i]),num_segments = num_segments[i])
            print(pooled.shape)
            # doing this padding means we need to construct proper attention masks
            padded = jnp.pad(
                pooled,
                pad_width=((self.seq_len - len(pooled),0),(0,0)), 
                mode='constant',
                constant_values=0
            )
            results.append(padded)
        return jnp.stack(results,axis=0)
    
        
    def __call__(self,params,x):
        # apply transformer with cross attention
        x = self.init_embed(params['init_embed'],x)
        
        for name,layer in self.entropy_layers.items():
            x = layer(params['entropy_transformer'][name],x)

        patches = self.max_pool(params,x)
        print(patches.shape)
        #patches = self.embed(params['patch_embed'],patches)
        for name,layer in self.patch_layers.items():
            patches = layer(params['patch_transformer'][name],patches,x,x) # cross attention to original last byte representation, needs mask 
        return patches
        # for patch cross attention q = patches, kv= original byte embeddings
