import jax
import jax.numpy as jnp
import optax

from jax import grad
from .utils.utils import *
from .model import TransformerDecoder,TransformerEncoder
from .layers.rope import Rope

def forward_loss(params,model,batch,targets):
    logits = model(params,batch)
    logits = logits - jnp.max(logits, axis=-1, keepdims=True)
    loss = crossentropy(targets,logits)
    return loss

def forward_loss_cross(params,model,batch,targets,k,v):
    logits = model(params,batch,k,v)
    logits = logits - jnp.max(logits, axis=-1, keepdims=True)
    loss = crossentropy(targets,logits)
    return loss

def step(params,opt_state,batch,targets,model,gradient,optimizer):
    loss,grads = gradient(params,model,batch,targets)
    updates, new_opt_state = optimizer.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state,loss

def step_cross(params,opt_state,batch,targets,model,gradient,optimizer,k,v):
    loss,grads = gradient(params,model,batch,targets,k,v)
    updates, new_opt_state = optimizer.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state,loss

def decoder_test():
    seqlen = 3
    causal_mask = jnp.triu(jnp.ones((seqlen, seqlen)), k=1).astype(bool)
    key = jax.random.PRNGKey(42)
    token_ids = jnp.ones((2, seqlen), dtype=jnp.int32)  # [batch=2, seq_len=3]
    token_ids = jnp.array([
        [0, 1, 2],  # Sequence 1
        [3, 4, 5]   # Sequence 2
    ])
    targets = jnp.array([
        [1, 2, 3],
        [4, 5, 6]
    ])
    x = jnp.array([
        [0, 1,2],
        [3, 4,5]
    ])
    transformer = TransformerDecoder(
        n_layers=2, 
        d_hidden=8, 
        d_model=16, 
        n_heads=4,
        v_size=256,  
        mask=causal_mask,
        d_latent=4
    )
    print('__________ model parameters: ')
    params = transformer.init(key)
    print(count_params(params['block1']))
    print('_____________________________')
    forward_loss(params,transformer,x,targets)

    gradient = jax.value_and_grad(forward_loss)




    optimizer = optax.adam(
        learning_rate=1e-4, 
        b1=0.9,             
        b2=0.999,           
        eps=1e-8            
    )
    opt_state = optimizer.init(params)

    optimizer = optax.sgd(
        learning_rate=0.1  # Tune this to your liking!
    )

    
    opt_state = optimizer.init(params)

    for epoch in range(1):
        params,opt_state,loss = step(params,opt_state,x,targets,transformer,gradient,optimizer)
        print(loss)
        print(softmax(transformer(params,x)))

def encoder_test():
    # this should work with kinda any key and value dim
    seqlen = 3
    key = jax.random.PRNGKey(42)
    token_ids = jnp.ones((2, seqlen), dtype=jnp.int32)  # [batch=2, seq_len=3]
    token_ids = jnp.array([
        [0, 1, 2],  # Sequence 1
        [3, 4, 5]   # Sequence 2
    ])
    targets = jnp.array([
        [1, 2, 3],
        [4, 5, 6]
    ])
    x = jnp.array([
        [0, 1,2],
        [3, 4,5]
    ])

    v = jax.random.normal(key,(2,2,16))
    k = jax.random.normal(key,(2,2,16))
    causal_mask = jnp.triu(jnp.ones((3,2)),k=1).astype(bool)
    transformer = TransformerEncoder(
        n_layers=2, 
        d_hidden=8, 
        d_model=16, 
        n_heads=4,
        v_size=256,  
        mask=causal_mask,
        d_k = 16,
        d_v = 16
    )
    print('__________ model parameters __________')
    params = transformer.init(key)
    print(count_params(params['block1']))
    print('_____________________________')
    forward_loss_cross(params,transformer,x,targets,k,v)

    gradient = jax.value_and_grad(forward_loss_cross)




    optimizer = optax.adam(
        learning_rate=1e-4, 
        b1=0.9,             
        b2=0.999,           
        eps=1e-8            
    )
    opt_state = optimizer.init(params)

    optimizer = optax.sgd(
        learning_rate=0.1  # Tune this to your liking!
    )

    
    opt_state = optimizer.init(params)
    print('______ shape __________')
    print(softmax(transformer(params,x,k,v)).shape)
    print('______ initial loss ______')

    print('_____________________________')
    for epoch in range(100):
        params,opt_state,loss = step_cross(params,opt_state,x,targets,transformer,gradient,optimizer,k,v)
        print(loss)
    
encoder_test()
















## TODO: ##
# - implement different FFN
# - add jax.lax.scan optimisations
# - implement rope
#
#
#
#
# -----
