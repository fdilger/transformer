import jax
import jax.numpy as jnp

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

    
