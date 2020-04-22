import numpy as np

def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t

def sample_function(itemnum, batch_size, maxlen, SEED):
    def sample():
        seq = np.zeros([maxlen], dtype = np.int32)
        pos = np.zeros([maxlen], dtype = np.int32)
        neg = np.zeros([maxlen], dtype = np.int32)

        tmp = [np.random.randint(1, itemnum + 1) for i in range(5)]

        seq = [0] * (maxlen - len(tmp)) + tmp 
        pos = [0] * (maxlen - 3) + tmp[-3:]
        neg = [0] * (maxlen - 3) + [random_neq(1, itemnum + 1, set(tmp)) for i in range(3)]

        return (0, seq, pos, neg)

    np.random.seed(SEED)
    one_batch = []
    for i in range(batch_size):
        one_batch.append(sample())

    return zip(*one_batch)

class WarpSampler(object):
    def __init__(self, User, usernum, itemnum, batch_size=64, maxlen=10, n_workers=1, num_batch=0):
        self.User = User
        self.usernum = usernum
        self.itemnum = itemnum
        self.batch_size = batch_size
        self.maxlen = maxlen
        self.num_batch = num_batch

    def next_batch(self):
        return sample_function(self.itemnum, self.batch_size, self.maxlen, np.random.randint(2e9))
