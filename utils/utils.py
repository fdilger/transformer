import jax
import jax.numpy as jnp
from math import prod

def sigmoid(x):
    return 1 / (1+jnp.exp(-x))


def softmax(x):
    logits = x - jnp.max(x, axis=-1, keepdims=True)
    exps = jnp.exp(logits)
    return exps / exps.sum(axis=-1,keepdims=True)

def log_softmax(x):
    exps = jnp.exp(x)
    return jnp.log(exps / exps.sum(axis=-1,keepdims=True))

def crossentropy(targets,logits):
    probs = softmax(logits)
    logprobs = jnp.log(probs)
    loss = -jnp.mean(logprobs[...,targets])
    return loss

def count_params(params):
    if isinstance(params,jax.Array):
        return prod(params.shape)
    if isinstance(params,dict):
        return sum(count_params(item) for item in params.values())
    return 1
        

