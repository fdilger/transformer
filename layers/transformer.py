import jax
import jax.numpy as jnp

from .mtypes import Tensor, Shape, Any, Callable, Params, PRNGKey
from ..normalization.layer_norm import LayerNorm
from ..layers.attention import LatentAttention,CrossAttention,Attention
from ..layers.linear import FFN
from ..utils.utils import softmax

class TransformerDecoderBlock:
    def __init__(
            self,
            num_heads  : int,
            embed_dim  : int,
            ff_dim     : int,
            vocab_size : int,
            seq_len    : int,
            mask       : Tensor
            latent_dim : int = None
            ) -> None:


        attention = Attention(num_heads, embed_dim, mask)
        if d_latent:
            attention = LatentAttention(
                num_heads,
                embed_dim,
                latent_dim,
                mask
            )
        
        self.layers = {
            'norm1': LayerNorm(embed_dim),
            'att'  : attention,
            'norm2': LayerNorm(embed_dim),
            'ffn'  : FFN(embed_dim, ff_dim, embed_dim)
        }

    def init(self, key : PRNGKey) -> Params:
        keys = jax.random.split(key,4)
        return {
            name : layer.init(key)
            for key,(name,layer) in zip(keys,self.layers.items())
        }
            
    def __call__(self, params : Params, x : Tensor) -> Tensor:
        for name,layer in self.layers.items():
            x = layer(params[name],x)
        return x


class TransformerEncoderBlock:
    def __init__(
            self,
            num_heads  : int,
            embed_dim  : int,
            ff_dim     : int,
            key_dim    : int,
            value_dim  : int
            vocab_size : int,
            seq_len    : int,
            mask       : Tensor
            ) -> None
    
        attention = CrossAttention(
            num_heads,
            embed_dim,
            key_dim,
            value_dim,
            mask
        )
        self.layers = {
            'norm1': LayerNorm(embed_dim),
            'att'  : attention,
            'norm2': LayerNorm(embed_dim),
            'ffn'  : FFN(embed_dim,ff_dim,embed_dim)
        }

    def init(self, key : PRNGKey) -> Params:
        keys = jax.random.split(key,4)
        return {
            name : layer.init(key)
            for key,(name,layer) in zip(keys,self.layers.items())
        }
            
    def __call__(
            self,
            params : Params,
            x      : Tensor,
            k      : Tensor,
            v      : Tensor
    ) -> Tensor:
        
        for name,layer in self.layers.items():
            if name == 'att':
                x = layer(params[name],x,k,v)
            else:
                x = layer(params[name],x)
        return x
