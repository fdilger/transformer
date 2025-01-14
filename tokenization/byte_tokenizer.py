import jax
import jax.numpy as jnp

class ByteTokenizer:
    def __init__(self,data):
        unique_bytes = sorted(set(data))
        self.v_size = len(unique_bytes)
        self.byte_to_id = {b: i for i,b in enumerate(unique_bytes)}
        self.id_to_byte = {i: b for i,b in enumerate(unique_bytes)}

        self.lookup = jnp.zeros(256, dtype=jnp.int32)
        for b, i in self.byte_to_id.items():
            self.lookup = self.lookup.at[b].set(i)
            
    def encode(self, bytes_data):
        bytes_data = jnp.frombuffer(bytes_data,dtype=jnp.uint8)
        return self.lookup[bytes_data]
    
    def decode(self,token):
        return self.id_to_byte[token]

    
