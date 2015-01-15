import cPickle

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("state", type=str)
parser.add_argument("final_state", type=str)
parser.add_argument("lang", type=str)
args = parser.parse_args()

with open(args.state,'rb') as f:
    d = cPickle.load(f)

if args.lang == 'fr':
    d['indx_word'] = "/data/lisatmp3/chokyun/mt/vocab.unlimited/bitexts.selected/ivocab.en.pkl"
    d['indx_word_target'] = "/data/lisatmp3/chokyun/mt/vocab.unlimited/bitexts.selected/ivocab.fr.pkl"
    d['word_indx'] = "/data/lisatmp3/chokyun/mt/vocab.unlimited/bitexts.selected/vocab.en.pkl"
    d['word_indx_trgt'] = "/data/lisatmp3/chokyun/mt/vocab.unlimited/bitexts.selected/vocab.fr.pkl"
elif args.lang == 'de':
    d['indx_word'] = "/data/lisatmp3/jeasebas/nmt/data/deutsch/final/de-en.clean.max50.ivocab.en.pkl"
    d['indx_word_target'] = "/data/lisatmp3/jeasebas/nmt/data/deutsch/final/de-en.clean.max50.ivocab.de.pkl"
    d['word_indx'] = "/data/lisatmp3/jeasebas/nmt/data/deutsch/final/de-en.clean.max50.vocab.en.pkl"
    d['word_indx_trgt'] = "/data/lisatmp3/jeasebas/nmt/data/deutsch/final/de-en.clean.max50.vocab.de.pkl"
else:
    raise NotImplementedError


with open(args.final_state,'w') as f:
    cPickle.dump(d, f, 0)