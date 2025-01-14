import jax
import jax.numpy as jnp
from jax import grad
import optax



from .tokenization.byte_tokenizer import ByteTokenizer
from .model import TransformerDecoder
from .utils.utils import count_params
from .utils.utils import *

import asyncio
import websockets
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrainingMonitor:
    def __init__(self, websocket_url="ws://localhost:8000/ws"):
        self.websocket_url = websocket_url
        self.websocket = None
        
    async def connect(self):
        if self.websocket is None:
            self.websocket = await websockets.connect(self.websocket_url)
            logger.info("Connected to visualization server")
            
    async def send_update(self, loss, accuracy, lr, epoch, total_epochs, progress):
        try:
            if self.websocket is None:
                await self.connect()
                
            metrics = {
                "loss": float(loss),
                "accuracy": float(accuracy),
                "learning_rate": float(lr),
                "epoch_progress": float(progress),
                "current_epoch": epoch,
                "total_epochs": total_epochs,
                "time_remaining": "00:00:00",
                "warnings": []
            }
            
            await self.websocket.send(json.dumps(metrics))
            logger.debug("Sent training update")
            
        except Exception as e:
            logger.error(f"Error sending update: {e}")
            self.websocket = None  # Reset connection on error
            
    async def close(self):
        if self.websocket:
            await self.websocket.close()
            self.websocket = None

# Global monitor instance
monitor = TrainingMonitor()

# For use in training loop
async def update_training(loss, accuracy, lr, epoch, total_epochs, progress):
    await monitor.send_update(loss, accuracy, lr, epoch, total_epochs, progress)

# For non-async code
def send_training_update(loss, accuracy, lr, epoch, total_epochs, progress):
    asyncio.run(update_training(loss, accuracy, lr, epoch, total_epochs, progress))

def load_tiny_shakespeare_bytes():
     with open('transformer/datasets/tinyshakespeare/input.txt', 'r', encoding='ascii') as f:
         text = f.read()
         return text.encode('ascii')



epochs = range(10)


# prepare data
     
data = load_tiny_shakespeare_bytes()
tokenizer = ByteTokenizer(data)
tokenized = tokenizer.encode(data)
data = tokenized
num_bytes = len(data)
print(data.shape)


batch_size = 32
seq_len = 16

def batch(batch_size,seq_len,data):
    num_seqs = len(data) // seq_len
    data = data[:num_seqs*seq_len]
    num_bs = num_seqs // batch_size
    data = data[:num_bs*batch_size*seq_len]
    print(len(data))
    print(num_bs*batch_size*seq_len)
    data = jnp.reshape(data,(num_bs,batch_size,seq_len))
    return data

batched = batch(batch_size,seq_len,data)
# First create all input-target pairs
xs = batched[:,:seq_len-1,:]
ys = batched[:,1:,:]

# Then split into train/test
train_xs = xs[:len(xs)-100,:,:]
train_ys = ys[:len(ys)-100,:,:]
test_xs = xs[len(xs)-100:,:,:]
test_ys = ys[len(ys)-100:,:,:]
# init model

model = TransformerDecoder(
    n_layers = 10,
    d_model = 16,
    d_hidden = 32,
    n_heads = 4,
    v_size = tokenizer.v_size,
    d_latent = 4,
    mask = jnp.triu(jnp.ones((seq_len, seq_len)), k=1).astype(bool)
)


key = jax.random.key(42)
params = model.init(key)
print('#params : ' + str(count_params(params)))

# init opt
schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0,
    peak_value=1e-3,  # More typical learning rate
    warmup_steps=100,  # More warmup steps
    decay_steps=1000,
    end_value=1e-5,   # Small but non-zero end value
)

opt=optax.adamw(learning_rate=schedule)


state = opt.init(params)


def loss(params,targets,batch):
    logits = model(params,batch)
    loss = crossentropy(targets,logits)
    return loss

@jax.jit
def step(params,state,batch,targets):
    val,grads = jax.value_and_grad(loss)(params, batch,targets)
    updates, state = opt.update(grads, state,params)
    params = optax.apply_updates(params=params, updates=updates)
    return params,state,val
print(type(schedule))
b=1
p = 18
for epoch in epochs:
    for batch,targets in zip(train_xs,train_ys):
        params,state,val = step(params,state,batch,targets)
        if b % 2 == 0:
            if b % 30 == 0:
                v_loss = 0
                n = 0
                for batch1,targets1 in zip(test_xs,test_ys):
                    n+=1
                    v_loss += loss(params,targets1,batch1)
                    p = v_loss/n
            send_training_update(val,p,schedule(state[2][0]),epoch+1,10,round(b/train_xs.shape[0]*len(epochs)))
        b+=1
    
