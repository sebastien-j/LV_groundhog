# Before a code change on October 25th, roll_vocab_small2large() was not called
# when saving the model.
# This script modifies the "large" parameters when building the final model.

import numpy
import shelve
import cPickle
import argparse

from groundhog.utils import invert_dict

parser = argparse.ArgumentParser()
parser.add_argument("cur_model", type=str)
parser.add_argument("cur_large", type=str)
parser.add_argument("final_model", type=str)
parser.add_argument("timing", type=str)
parser.add_argument("state", type=str)
args = parser.parse_args()

cur_model = numpy.load(args.cur_model)
cur_large = numpy.load(args.cur_large)
timing = numpy.load(args.timing)
with open(args.state, 'rb') as f:
    state = cPickle.load(f)

step = timing['step']
if step == 0:
    raise ValueError
step -= 1 # step gives where the model should restart. -1 to make sure we have the dictionary when the model was saved (if we would roll at the beginning)

with open(state['rolling_vocab_dict'], 'rb') as f:
    rolling_vocab_dict = cPickle.load(f)
total_num_batches = max(rolling_vocab_dict)
Dx_shelve = shelve.open(state['Dx_file'])
Dy_shelve = shelve.open(state['Dy_file'])

step_modulo = step % total_num_batches
if step_modulo in rolling_vocab_dict: # 0 always in.
    cur_key = step_modulo
else:
    cur_key = 0
    for key in rolling_vocab_dict:
        if (key < step_modulo) and (key > cur_key): # Find largest key smaller than step_modulo
            cur_key = key
large2small_src = Dx_shelve[str(cur_key)]
large2small_trgt = Dy_shelve[str(cur_key)]
small2large_src = invert_dict(large2small_src)
small2large_trgt = invert_dict(large2small_trgt)

restricted_list = ['W_0_enc_approx_embdr', 'W_0_dec_approx_embdr', 'W2_dec_deep_softmax', 'b_dec_deep_softmax'] 

d = {}

for elt in restricted_list:
    temp = cur_model[elt]
    arr = cur_large['large_' + elt]
    if '_enc_' in elt:
        for large in large2small_src:
            small = large2small_src[large]
            arr[large] = temp[small]
    else:
        for large in large2small_trgt:
            small = large2small_trgt[large]
            if elt != 'W2_dec_deep_softmax':
                arr[large] = temp[small]
            else:
                arr[:,large] = temp[:,small]
    d[elt] = arr

for elt in cur_model:
    if elt not in restricted_list:
        d[elt] = cur_model[elt]

numpy.savez(args.final_model, **d)