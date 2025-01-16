class DataSet:
    
    def __init__(self,data):
        self.data = data
        self.batched = False
        
    def shuffle(self,key):
        if not batched:
            return
        def batch(self,batch_size,seq_len)
        self.data = jax.random.permutation(key,self.data)
        n_batches = len(self.data) // batch_size
        self.data = self.data[:n_batches*batch_size].reshape(-1,self.batch_size,self.seq_len)
       
    def batch(self,batch_size,seq_len,data):
        num_seqs = len(data) // seq_len
        data = data[:num_seqs*seq_len]
        num_bs = num_seqs // batch_size
        data = data[:num_bs*batch_size*seq_len]
        print(len(data))
        print(num_bs*batch_size*seq_len)
        data = jnp.reshape(data,(num_bs,batch_size,seq_len))
        return data
    
    def batch_and_shift(self,batch_size,seq_len,data):
        xs = None
        ys = None
        return xs,ys

    def __iter__(self):
        for batch in self.data:
            yield sbatch


        
