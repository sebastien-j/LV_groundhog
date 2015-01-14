import cPickle
import numpy as np

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--old-state", type=str,
                    help="Needed only for vocabulary paths")
parser.add_argument("--new-state", type=str,
                    help="Needed only for vocabulary paths")
parser.add_argument("--old-model", type=str)
parser.add_argument("--new-model", type=str)
args = parser.parse_args()

with open(args.old_state,'rb') as f:
    old_state = cPickle.load(f)
with open(args.new_state,'rb') as f:
    new_state = cPickle.load(f)

with open(old_state['word_indx'], 'rb') as f:
    old_src_w2i = cPickle.load(f)
with open(old_state['word_indx_trgt'], 'rb') as f:
    old_trg_w2i = cPickle.load(f)
with open(new_state['indx_word'], 'rb') as f:
    new_src_i2w = cPickle.load(f)
with open(new_state['indx_word_target'], 'rb') as f:
    new_trg_i2w = cPickle.load(f)

restricted_list = ['W_0_enc_approx_embdr', 'W_0_dec_approx_embdr', 'W2_dec_deep_softmax', 'b_dec_deep_softmax'] 

old_model = np.load(args.old_model)

d = {}

old_tmp = old_model['W_0_enc_approx_embdr']
new_tmp = np.zeros((new_state['n_sym_source'], new_state['rank_n_approx']), dtype=np.float32)
new_tmp[:2] = old_tmp[:2]
for i in xrange(2, new_state['n_sym_source']):
    new_tmp[i] = old_tmp[old_src_w2i[new_src_i2w[i]]]
d['W_0_enc_approx_embdr'] = new_tmp.copy()

old_tmp = old_model['W_0_dec_approx_embdr']
new_tmp = np.zeros((new_state['n_sym_target'], new_state['rank_n_approx']), dtype=np.float32)
new_tmp[:2] = old_tmp[:2]
for i in xrange(2, new_state['n_sym_target']):
    new_tmp[i] = old_tmp[old_trg_w2i[new_trg_i2w[i]]]
d['W_0_dec_approx_embdr'] = new_tmp.copy()

old_tmp = old_model['W2_dec_deep_softmax']
new_tmp = np.zeros((new_state['rank_n_approx'], new_state['n_sym_target']), dtype=np.float32)
new_tmp[:,:2] = old_tmp[:,:2]
for i in xrange(2, new_state['n_sym_target']):
    new_tmp[:,i] = old_tmp[:,old_trg_w2i[new_trg_i2w[i]]]
d['W2_dec_deep_softmax'] = new_tmp.copy()

old_tmp = old_model['b_dec_deep_softmax']
new_tmp = np.zeros((new_state['n_sym_target']), dtype=np.float32)
new_tmp[:2] = old_tmp[:2]
for i in xrange(2, new_state['n_sym_target']):
    new_tmp[i] = old_tmp[old_trg_w2i[new_trg_i2w[i]]]
d['b_dec_deep_softmax'] = new_tmp.copy()

for elt in old_model:
    if elt not in restricted_list:
        d[elt] = old_model[elt]

np.savez(args.new_model, **d)