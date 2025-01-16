import jax
import jax.numpy as jnp
from ..utils.utils import softmax
from ..model import TransformerDecoder
from ..tokenization.byte_tokenizer import ByteTokenizer
import numpy as np

class Sampler:
    def __init__(self,temp=1,top_k=None,top_p=None,min_p=None):
        self.temp = temp
        self.top_k = top_k
        self.top_p = top_p
        self.min_p = min_p
        self.key = jax.random.key(42)
    def __call__(self,logits):
        
        logits = logits - jnp.max(logits, axis=-1, keepdims=True)
        # Convert to probabilities using stable softmax
        probs = jax.nn.softmax(logits)
        # Sample with fresh key
        self.key, subkey = jax.random.split(self.key)
        sampled = jax.random.categorical(subkey, logits)
        return sampled

def to_numpy(tree):
    """Recursively convert all jax.DeviceArray leaves to np.ndarray."""
    if isinstance(tree, dict):
        return {k: to_numpy(v) for k, v in tree.items()}
    elif hasattr(tree, 'device_buffer'):  
        # This checks if 'tree' is a jax.DeviceArray-like object
        return np.array(tree)
    else:
        return tree

def to_jax(tree):
    """Recursively convert all np.ndarray leaves to jnp.array."""
    if isinstance(tree, dict):
        return {k: to_jax(v) for k, v in tree.items()}
    elif isinstance(tree, np.ndarray):
        return jnp.array(tree)
    else:
        return tree

def save_checkpoint(filename, dictionary):
    """Save nested parameter dictionary (JAX -> NumPy -> disk)."""
    numpy_dict = to_numpy(dictionary)
    np.save(filename, numpy_dict, allow_pickle=True)

def load_checkpoint(filename):
    """Load nested parameter dictionary (disk -> NumPy -> JAX)."""
    loaded = np.load(filename + '.npy', allow_pickle=True).item()
    return to_jax(loaded)
            
def load_tiny_shakespeare_bytes():
     with open('transformer/datasets/tinyshakespeare/input.txt', 'r', encoding='ascii') as f:
         text = f.read()
         return text.encode('ascii')





data = load_tiny_shakespeare_bytes()
tokenizer = ByteTokenizer(data)
tokenized = tokenizer.encode(data)

for k in ['a','b','c','d','e','f','h']:
    b = k.encode('ascii')
    print(b)
    print(tokenizer.encode(b))
    c = np.array(tokenizer.encode(b))
    print(c)
    print(tokenizer.decode(c[0]))




model = TransformerDecoder(
    n_layers = 6,
    d_model = 256,
    d_hidden = 4*256,
    n_heads = 8,
    v_size = tokenizer.v_size,
    mask = jnp.tril(jnp.ones((127, 127)), k=1).astype(bool)
)



print(tokenized[0:4])
s = Sampler(temp=2)
params = load_checkpoint('tinyshakes')
x = jnp.zeros(shape = (1,127), dtype=jnp.int32)
x = x.at[0,0].set(18)
x = x.at[0,1].set(47)
x = x.at[0,2].set(56)
x = x.at[0,3].set(57)
print(x)
for k in range(50):
    c = model(params,x)
    print(c.shape)
    logits = c[0][k+4]
    a  = s(logits)
    print(a)
    x= x.at[0,k+4].set(a)


print(x)
bs = [tokenizer.decode(a) for a in np.array(x)[0]]
string1 = bytes(bs).decode('ascii')
print(string1)


