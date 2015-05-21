import numpy
import cPickle

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--top-unigram", type=str)
parser.add_argument("--src-w2i", type=str)
parser.add_argument("--trg-w2i", type=str)
parser.add_argument("--vocab-size", type=int)
parser.add_argument("--output", type=str)
args = parser.parse_args()

with open(args.src_w2i,'rb') as f:
    src_w2i = cPickle.load(f)
with open(args.trg_w2i,'rb') as f:
    trg_w2i = cPickle.load(f)
with open(args.top_unigram,'rb') as f:
    top_unigram = cPickle.load(f)

new_dict = {}

for old_key in top_unigram:
    if old_key == '<eps>': # Don't consider the empty string
        continue
    new_key = src_w2i[old_key] # Convert source word to its index
    if new_key >= args.vocab_size:
        continue
    old_value = top_unigram[old_key] # This is a list of words (with the most probable one first)
    new_value = [trg_w2i[elt] for elt in old_value if (trg_w2i[elt] < args.vocab_size)]
    if len(new_value) >= 1:
        new_dict[new_key] = new_value

with open(args.output,'wb') as f:
    cPickle.dump(new_dict, f, -1)
