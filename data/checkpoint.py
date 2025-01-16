import jax
import jax.numpy as np
import numpy as np


# barebones utility for storing param dicts

def to_numpy(tree):
    if isinstance(tree, dict):
        return {k: to_numpy(v) for k, v in tree.items()}
    elif hasattr(tree, 'device_buffer'):  
        return np.array(tree)
    else:
        return tree

def to_jax(tree):
    if isinstance(tree, dict):
        return {k: to_jax(v) for k, v in tree.items()}
    elif isinstance(tree, np.ndarray):
        return jnp.array(tree)
    else:
        return tree

def save_checkpoint(filename, dictionary):
    numpy_dict = to_numpy(dictionary)
    np.save(filename, numpy_dict, allow_pickle=True)

def load_checkpoint(filename):
    loaded = np.load(filename + '.npy', allow_pickle=True).item()
    return to_jax(loaded)
