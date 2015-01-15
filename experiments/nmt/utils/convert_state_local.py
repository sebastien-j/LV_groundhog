import cPickle

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("state", type=str)
parser.add_argument("final_state", type=str)
args = parser.parse_args()

with open(args.state,'rb') as f:
    d = cPickle.load(f)
d['n_sym_source'] = d['large_vocab_source']
d['n_sym_target'] = d['large_vocab_target']
del d['large_vocab_source']
del d['large_vocab_target']
del d['rolling_vocab']
del d['save_algo']
del d['save_gs']

with open(args.final_state,'w') as f:
    cPickle.dump(d, f, 0)