import jax
import jax.numpy as jnp

from .layers.embeddings import Embeddings
from .layers.transformer import TransformerDecoderBlock,TransformerEncoderBlock
from .layers.linear import Embedding
from .utils.utils import softmax, log_softmax,crossentropy,count_params
from .mtypes import Tensor, Shape, Any, Callable, Params, PRNGKey

# mask = (segments[:, :, None] == segments[:, None, :]) patch-mask
# first compute the byte representations meant for the entropy model
# pass these to entropy model for patches
# max pool patches
# project patches to cross attention model with cross attention to original byte representations
# -> remove embedding from the transformer class

class TransformerDecoder:
    def __init__(
            self,
            num_layers : int,
            num_heads  : int,
            embed_dim  : int,
            ff_dim     : int,
            vocab_size : int,
            seq_len    : int,
            mask       : Tensor
            latent_dim : int = None

    ) -> None:
        self.att_mask   = mask 
        self.seq_len    = seq_len
        self.num_layers = num_layers
        self.ff_dim     = ff_dim
        self.embed_dim  = embed_dim
        
        # model expects one-hot encodings
        self.layers     = {'embed' : Embedding(vocab_size,embed_dim)}
        
        for idx in range(n_layers):
            block_name    = f'block{idx}'
            decoder_block = TransformerDecoderBlock(
                num_heads,
                embed_dim,
                ff_dim,
                vocab_size,
                maskt,
                latent_dim
            )
            self.layers[block_name] = decoder_block
            
        self.layers['out_proj'] = Embedding(embed_dim,vocab_size)
    
    def init(self, key : PRNGKey) -> Params:
        keys = jax.random.split(key,self.num_layers+1)
        return {
            name : layer.init(key)
            for key,(name,layer) in zip(keys,self.layers.items())
        }
    
    def __call__(self, params : Params, x : Tensor) -> Tensor:
        for name,layer in self.layers.items():
            x = layer(params[name],x)
        return x


class TransformerEncoder:
    def __init__(
            self,
            num_layers : int,
            num_heads  : int,
            embed_dim  : int,
            ff_dim     : int,
            key_dim    : int,
            value_dim  : int,
            vocab_size : int,
            seq_len    : int,
            mask       : Tensor
    ) -> None:
        
        self.n_layers = n_layers
        self.d_hidden = d_hidden
        self.d_model = d_model
        
        self.layers = {'embed' : Embeddings(d_model,v_size)}
        for idx in range(n_layers):
            block_name = f'block{idx}'
            encoder_block = TransformerEncoderBlock(
                d_model,
                d_hidden,
                n_heads,
                mask,
                d_k,
                d_v
            )
            self.layers[block_name] = encoder_block
        self.layers['pred_head'] = PredictionHead(d_model,v_size)

        
    def init(self, key : PRNGKey) -> Params:
        keys = jax.random.split(key,self.num_layers+1)
        return {
            name : layer.init(key)
            for key,(name,layer) in zip(keys,self.layers.items())
        }
    
    def __call__(
            self,
            params : Params,
            x : Tensor,
            k : Tensor,
            v : Tensor
    ) -> Tensor:
        
        for name,layer in self.layers.items():
            # this is terrrible
            if name[0:5] == 'block':
                x = layer(params[name],x,k,v)
            else:
                x = layer(params[name],x)
        return x



# deepseek-v2
# d_latent = 4*(d_model // n_heads) 
