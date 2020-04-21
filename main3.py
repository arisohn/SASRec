import os
import time
import argparse
import tensorflow as tf
from sampler2 import WarpSampler
from model2 import Model
from tqdm import tqdm
from util import *


def str2bool(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True)
parser.add_argument('--train_dir', required=True)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--maxlen', default=50, type=int)
parser.add_argument('--hidden_units', default=50, type=int)
parser.add_argument('--num_blocks', default=2, type=int)
parser.add_argument('--num_epochs', default=201, type=int)
parser.add_argument('--num_heads', default=1, type=int)
parser.add_argument('--dropout_rate', default=0.5, type=float)
parser.add_argument('--l2_emb', default=0.0, type=float)

args = parser.parse_args()

dataset = data_partition(args.dataset)
[user_train, user_valid, user_test, usernum, itemnum] = dataset
num_batch = len(user_train) / args.batch_size
cc = 0.0
for u in user_train:
    cc += len(user_train[u])
print 'average sequence length: %.2f' % (cc / len(user_train))

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
sess = tf.Session(config=config)

sampler = WarpSampler(user_train, usernum, itemnum, batch_size=args.batch_size, maxlen=args.maxlen, n_workers=3, num_batch = num_batch)
model   = Model(usernum, itemnum, args)
sess.run(tf.initialize_all_variables())

try:
    for epoch in range(1, args.num_epochs + 1):
        print epoch, args.num_epochs

        for step in tqdm(range(num_batch), total=num_batch, ncols=70, leave=False, unit='b'):
            u, seq, pos, neg = sampler.next_batch()

            auc, loss, _ = sess.run([model.auc, model.loss, model.train_op], {model.u: u, model.input_seq: seq, model.pos: pos, model.neg: neg, model.is_training: True})

            if epoch % 20 == 0:
                seq_logits = sess.run(model.seq_logits, {model.u: u, model.input_seq: seq, model.is_training: False})

                print np.shape(seq)
                print seq_logits.shape
                print seq[0]
                print pos[0]
                print np.argmax(seq_logits[0], axis =-1)
                print auc, loss 

                """
                    t1 = time.time() - t0
                    T += t1
                    print 'Evaluating',
                    t_test = evaluate(model, dataset, args, sess)
                    t_valid = evaluate_valid(model, dataset, args, sess)
                    print ''
                    print 'epoch:%d, time: %f(s), valid (NDCG@10: %.4f, HR@10: %.4f), test (NDCG@10: %.4f, HR@10: %.4f)' % (
                    epoch, T, t_valid[0], t_valid[1], t_test[0], t_test[1])

                    f.write(str(t_valid) + ' ' + str(t_test) + '\n')
                    f.flush()
                    t0 = time.time()
                 """

except Exception as e:
    print(e)
    sampler.close()
    exit(1)

sampler.close()
print("Done")
