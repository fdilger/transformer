import jax
import jax.numpy as jnp
from jax import grad
import optax
import numpy as np

from .tokenization.byte_tokenizer import ByteTokenizer
from .model import TransformerDecoder
from .utils.utils import count_params
from .utils.utils import *

import asyncio
import websockets
import json
import logging
from functools import partial

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrainingMonitor:
    def __init__(self, websocket_url="ws://localhost:8000/ws"):
        self.websocket_url = websocket_url
        self.websocket = None
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.reconnect_delay = 1
        self.max_reconnect_delay = 30
        self.connected = False
        self.heartbeat_task = None
        
    async def heartbeat(self):
        """Keep connection alive with periodic pings"""
        try:
            while self.connected:
                if self.websocket and not self.websocket.closed:
                    try:
                        await self.websocket.ping()
                        await asyncio.sleep(5)  # Send ping every 5 seconds
                    except:
                        self.connected = False
                        self.websocket = None
                        await self.connect()
                else:
                    await asyncio.sleep(1)
        except Exception as e:
            logger.error(f"Heartbeat error: {e}")
            
    async def connect(self):
        while not self.connected:
            try:
                if self.websocket is None:
                    self.websocket = await websockets.connect(
                        self.websocket_url,
                        ping_interval=None,  # Disable default ping
                        ping_timeout=None,   # Disable default ping timeout
                        close_timeout=1,     # Fast closure on error
                    )
                    self.connected = True
                    logger.info("Connected to visualization server")
                    self.reconnect_delay = 1
                    
                    # Start heartbeat in the background
                    if self.heartbeat_task is None or self.heartbeat_task.done():
                        self.heartbeat_task = asyncio.create_task(self.heartbeat())
                    
                return
            except Exception as e:
                logger.error(f"Connection failed: {e}")
                self.connected = False
                self.websocket = None
                await asyncio.sleep(self.reconnect_delay)
                self.reconnect_delay = min(self.reconnect_delay * 2, self.max_reconnect_delay)
                
    async def send_update(self, loss, accuracy, lr, epoch, total_epochs, progress):
        max_retries = 5  # Increase max retries
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                if not self.connected or self.websocket is None:
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
                return  # Success
                
            except Exception as e:
                logger.error(f"Error sending update (attempt {retry_count + 1}/{max_retries}): {e}")
                self.connected = False
                self.websocket = None
                retry_count += 1
                await asyncio.sleep(min(1 * retry_count, 5))  # Progressive delay
                
    def send_training_update(self, loss, accuracy, lr, epoch, total_epochs, progress):
        """Non-async wrapper for send_update"""
        if self.loop.is_closed():
            return
            
        future = asyncio.run_coroutine_threadsafe(
            self.send_update(loss, accuracy, lr, epoch, total_epochs, progress),
            self.loop
        )
        try:
            future.result(timeout=10)  # Increased timeout further
        except Exception as e:
            logger.error(f"Failed to send update: {e}")
            
    async def close(self):
        self.connected = False
        if self.heartbeat_task:
            self.heartbeat_task.cancel()
            try:
                await self.heartbeat_task
            except asyncio.CancelledError:
                pass
        if self.websocket:
            await self.websocket.close()
            self.websocket = None
# Global monitor instance
monitor = TrainingMonitor()

# For use in training loop - now just use monitor.send_training_update directly
def send_training_update(loss, accuracy, lr, epoch, total_epochs, progress):
    monitor.send_training_update(loss, accuracy, lr, epoch, total_epochs, progress)

# Start the event loop in a separate thread
import threading
def run_event_loop(loop):
    asyncio.set_event_loop(loop)
    loop.run_forever()

threading.Thread(target=run_event_loop, args=(monitor.loop,), daemon=True).start()

    
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



epochs = range(10)


# prepare data
     
data = load_tiny_shakespeare_bytes()
tokenizer = ByteTokenizer(data)
tokenized = tokenizer.encode(data)
data = tokenized
num_bytes = len(data)
print(data.shape)
print(tokenizer.v_size)

batch_size = 32
seq_len = 128

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

xs = batched[:,:,:seq_len-1]
ys = batched[:,:,1:]
ys = jax.nn.one_hot(ys, tokenizer.v_size)
print(len(xs))

train_xs = xs[:len(xs)-5,:,:]
train_ys = ys[:len(ys)-5,:,:]
test_xs = xs[len(xs)-5:,:,:]
test_ys = ys[len(ys)-5:,:,:]
print(train_xs[0])
# init model
print(train_xs.shape)
print(train_ys.shape)
model = TransformerDecoder(
    n_layers = 6,
    d_model = 256,
    d_hidden = 4*256,
    n_heads = 8,
    v_size = tokenizer.v_size,
    mask = jnp.triu(jnp.ones((seq_len-1, seq_len-1)), k=1).astype(bool)
)


key = jax.random.key(42)
params = model.init(key)
print('#params : ' + str(count_params(params)))

# init opt
schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0,
    peak_value=2e-3,  
    warmup_steps=80, 
    decay_steps=400,
    end_value=1e-4,   
)

opt=optax.adamw(learning_rate=schedule)

state = opt.init(params)


def loss(params,targets,batch):
    logits = model(params,batch)
    #    loss = crossentropy(targets,logits)
    loss = optax.softmax_cross_entropy(logits,targets)
    return jnp.mean(loss)
@jax.jit
def step(params,state,batch,targets):
    val,grads = jax.value_and_grad(loss)(params, targets,batch)
    updates, state = opt.update(grads, state,params)
    params = optax.apply_updates(params=params, updates=updates)
    return params,state,val
print(type(schedule))
b=0
p = 0

for epoch in [0]:
    for batch,targets in zip(train_xs,train_ys):
        b+=1
        params,state,val = step(params,state,batch,targets)
        if b % 5 == 0:
 
            print('loss '+ str(val))
            if b % 50 == 0:
                v_loss = 0
                n = 0
                for batch1,targets1 in zip(test_xs,test_ys):
                    n+=1
                    v_loss += loss(params,targets1,batch1)
                p = v_loss/n
            print('val loss '+ str(p))
            monitor.send_training_update(val,p,schedule(state[2][0]),epoch+1,10,round(b/train_xs.shape[0]*len(epochs)))
    save_checkpoint('tinyshakes', params)

v_loss = 0
n = 0
for batch1,targets1 in zip(train_xs[0:10],train_ys[0:10]):
    n+=1
    v_loss += loss(params,targets1,batch1)
p = v_loss/n
print('train loss first 10 batches'+ str(p))



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







for k in ['a','b','c','d','e','f','h']:
    b = k.encode('ascii')
    print(b)
    print(tokenizer.encode(b))
    c = np.array(tokenizer.encode(b))
    print(c)
    print(tokenizer.decode(c[0]))







print(tokenized[0:4])
s = Sampler(temp=2)
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







