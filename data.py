class DataSet:
    
    def __init__(self,tokens,batch_size,seq_len):
        self.seq_len = seq_len
        self.batch_size = batch_size
        n_seqs = len(tokens) // seq_len
        self.data = tokens[:n_seqs*seq_len].reshape(-1,seq_len)
        
    def shuffle(self,key):
        self.data = jax.random.permutation(key,self.data)
        n_batches = len(self.data) // batch_size
        self.data = self.data[:n_batches*batch_size].reshape(-1,self.batch_size,self.seq_len)
       
    def __iter__(self):
        for batch in self.data:
            yield batch


        
