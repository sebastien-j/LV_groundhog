# Assumes null_sym_source/target = 0
# unk_sym_source/target = 1

# Verify what is missing from each dict
# How was it saved (0,-1)

##For full vocab

# iv : iv[0] = '<\s>'
# iv : iv[1] : KeyError
## v : v['<\s>'] = 0
## v : v['<s>'] = 0
### len(v) = len(iv) + 1
### Use *_sym_* in state to specify <UNK> and <EOS>

import cPickle
import os

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--state", type=str,
                    help="Needed only for vocabulary paths")
parser.add_argument("--source-file", type=str,
                    help="Text to translate")
parser.add_argument("--num-common", type=int)
parser.add_argument("--num-ttables", type=int)
parser.add_argument("--topn-file", type=str,
                    help="With old indices")
parser.add_argument("--ext", type=str,
                    help="*.pkl -> *.ext.pkl")
parser.add_argument("--save-vocab-dir", type=str)
parser.add_argument("--new-state", action="store_true", default=False)
parser.add_argument("--new-topn", action="store_true", default=False)
args = parser.parse_args()

with open(args.state, 'rb') as f:
    d = cPickle.load(f)
with open(d['indx_word'], 'rb') as f:
    old_src_i2w = cPickle.load(f)
with open(d['word_indx'], 'rb') as f:
    old_src_w2i = cPickle.load(f)
with open(d['indx_word_target'], 'rb') as f:
    old_trg_i2w = cPickle.load(f)
with open(d['word_indx_trgt'], 'rb') as f:
    old_trg_w2i = cPickle.load(f)
with open(args.topn_file, 'rb') as f:
    topn = cPickle.load(f)

for elt in topn:
    topn[elt] = topn[elt][:args.num_ttables]

src_i2w = {}
src_w2i = {}
trg_i2w = {}
trg_w2i = {}

src_i2w[0] = '<\s>'
src_w2i['<s>'] = 0
src_w2i['</s>'] = 0

trg_i2w[0] = '<\s>'
trg_w2i['<s>'] = 0
trg_w2i['</s>'] = 0

# Fill common target words
for i in xrange(2, args.num_common):
    trg_i2w[i] = old_trg_i2w[i]
    trg_w2i[trg_i2w[i]] = i

cur_src_index = 2
cur_trg_index = args.num_common
with open(args.source_file) as f:
    for line in f:
        line = line.strip().split()
        for word in line:
            if (old_src_w2i.get(word, d['n_sym_source']) < d['n_sym_source']) and (word not in src_w2i):
                src_w2i[word] = cur_src_index
                src_i2w[cur_src_index] = word
                cur_src_index += 1
                target_old_indices = topn[old_src_w2i[word]]
                for index in target_old_indices: # Should always be < d['n_sym_target'] 
                    trg_word = old_trg_i2w[index]
                    if trg_word not in trg_w2i:
                        trg_w2i[trg_word] = cur_trg_index
                        trg_i2w[cur_trg_index] = trg_word
                        cur_trg_index += 1

# w2i was saved with highest pickle protocol, but not i2w
# Do the same here?
if not args.save_vocab_dir:
    save_vocab_dir = os.path.dirname(d['indx_word'])
else:
    save_vocab_dir = args.save_vocab_dir

with open(os.path.join(save_vocab_dir, os.path.basename(d['indx_word'])[:-3] + args.ext + '.pkl'), 'w') as f:
    cPickle.dump(src_i2w, f, 0)
with open(os.path.join(save_vocab_dir, os.path.basename(d['word_indx'])[:-3] + args.ext + '.pkl'), 'wb') as f:
    cPickle.dump(src_w2i, f, -1)
with open(os.path.join(save_vocab_dir, os.path.basename(d['indx_word_target'])[:-3] + args.ext + '.pkl'), 'w') as f:
    cPickle.dump(trg_i2w, f, 0)
with open(os.path.join(save_vocab_dir, os.path.basename(d['word_indx_trgt'])[:-3] + args.ext + '.pkl'), 'wb') as f:
    cPickle.dump(trg_w2i, f, -1)

if args.new_state:
    d['indx_word'] = os.path.join(save_vocab_dir, os.path.basename(d['indx_word'])[:-3] + args.ext + '.pkl')
    d['word_indx'] = os.path.join(save_vocab_dir, os.path.basename(d['word_indx'])[:-3] + args.ext + '.pkl')
    d['indx_word_target'] = os.path.join(save_vocab_dir, os.path.basename(d['indx_word_target'])[:-3] + args.ext + '.pkl')
    d['word_indx_trgt'] = os.path.join(save_vocab_dir, os.path.basename(d['word_indx_trgt'])[:-3] + args.ext + '.pkl')
    
    d['n_sym_source'] = len(src_w2i)
    d['n_sym_target'] = len(trg_w2i)

    with open(args.state[:-3] + args.ext + '.pkl' , 'wb') as f:
        cPickle.dump(d, f, -1)

if args.new_topn:
    new_topn = {}
    for i in xrange(2, len(src_w2i)):
        new_topn[i] = [trg_w2i[old_trg_i2w[index]] for index in topn[old_src_w2i[src_i2w[i]]]]
    with open(args.topn_file[:-3] + args.ext + '.pkl', 'wb') as f:
        cPickle.dump(new_topn, f, -1)