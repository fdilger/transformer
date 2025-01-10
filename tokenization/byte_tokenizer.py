import jax
import jax.numpy as jnp

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

    
