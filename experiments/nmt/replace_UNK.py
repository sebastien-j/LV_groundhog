#!/usr/bin/env python

import argparse
import cPickle
import traceback
import logging
import time
import sys
import numpy
import theano

from collections import OrderedDict

import experiments.nmt
from experiments.nmt import\
    RNNEncoderDecoder,\
    prototype_phrase_state,\
    parse_input

from experiments.nmt.numpy_compat import argpartition

logger = logging.getLogger(__name__)

def parse_output(word2idx, line, eos_id, unk_id, raise_unk=False):
    seqin = line.split()
    seqlen = len(seqin)
    seq = numpy.zeros(seqlen+1, dtype='int64')
    for idx,sx in enumerate(seqin):
        seq[idx] = word2idx.get(sx, unk_id)
        # Assumes that there are no words with
        # a proper index, but no vector representation.
        # It may crash otherwise.
        if seq[idx] == unk_id and raise_unk:
            raise Exception("Unknown word {}".format(sx))

    seq[-1] = eos_id

    return seq, seqin

# From score.py
def pack(seqs, return_lengths=False):
    num = len(seqs)
    lengths = map(len, seqs)
    max_len = max(lengths)
    x = numpy.zeros((num, max_len), dtype="int64")
    x_mask = numpy.zeros((num, max_len), dtype="float32")
    for i, seq in enumerate(seqs):
        x[i, :len(seq)] = seq
        x_mask[i, :len(seq)] = 1.0
    if not return_lengths:
        return x.T, x_mask.T
    else:
        return x.T, x_mask.T, numpy.asarray(lengths)

def update_dicts(indices, d, D, C, full):
    for word in indices:
        if word not in d:
            if len(d) == full:
                raise RuntimeError("The dictionary is full")
            if word not in D: # Also not in C
                key, value = C.popitem()
                del D[key]
                d[word] = 0
                D[word] = 0
            else: # Also in C as (d UNION C) is D. (d INTERSECTION C) is the empty set.
                d[word] = 0
                del C[word]

def compute_alignment(src_seqs, trg_seqs, alignment_fns, batchsize):
    full_x, full_x_mask, full_x_lengths = pack(src_seqs, return_lengths=True)
    full_y, full_y_mask = pack(trg_seqs)
    assert full_x.shape[1] == full_y.shape[1]

    num_models = len(alignment_fns)

    full_alignments = numpy.zeros((full_y.shape[0], full_x.shape[0], 0), dtype=numpy.float32)

    for batch_start in xrange(0, full_x.shape[1], batchsize):
        alignments = 0.
        x = full_x[:,batch_start:batch_start+batchsize]
        x_mask = full_x_mask[:,batch_start:batch_start+batchsize]
        x_lengths = full_x_lengths[batch_start:batch_start+batchsize]
        y = full_y[:,batch_start:batch_start+batchsize]
        y_mask = full_y_mask[:,batch_start:batch_start+batchsize]
        for j in xrange(num_models):
            # target_len x source_len x num_examples
            alignments += numpy.asarray(alignment_fns[j](x, y, x_mask, y_mask)[0])
        alignments[:,x_lengths-1,range(x.shape[1])] = 0. # Put source <eos> score to 0.
        full_alignments = numpy.concatenate((full_alignments, alignments), axis=2)
        hard_alignments = numpy.argmax(full_alignments, axis=1) # trg_len x num_examples

    return hard_alignments

def replace_unknown_words(src_word_seqs, trg_seqs, trg_word_seqs, hard_alignments,
                          heuristic, mapping, unk_id, new_trans_file, n_best, full_trans_lines=None):
    for i, src_words in enumerate(src_word_seqs):
        trans_words = trg_word_seqs[i]
        trans_seq = trg_seqs[i]
        hard_alignment = hard_alignments[:,i]
        if n_best:
            full_trans_line = full_trans_lines[i]

        new_trans_words = []
        for j in xrange(len(trans_words) - 1): # -1 : Don't write <eos>
            if trans_seq[j] == unk_id:
                UNK_src = src_words[hard_alignment[j]]
                if heuristic == 0: # Copy (ok when training with large vocabularies on en->fr, en->de)
                    new_trans_words.append(UNK_src)
                elif heuristic == 1:
                    # Use the most likely translation (with t-table). If not found, copy the source word.
                    # Ok for small vocabulary (~30k) models
                    if UNK_src in mapping:
                        new_trans_words.append(mapping[UNK_src])
                    else:
                        new_trans_words.append(UNK_src)
                elif heuristic == 2:
                    # Use t-table if the source word starts with a lowercase letter. Otherwise copy
                    # Sometimes works better than other heuristics
                    if UNK_src in mapping and UNK_src.decode('utf-8')[0].islower():
                        new_trans_words.append(mapping[UNK_src])
                    else:
                        new_trans_words.append(UNK_src)
            else:
                new_trans_words.append(trans_words[j])

        to_write = ''
        for j, word in enumerate(new_trans_words):
            to_write = to_write + word
            if j < len(new_trans_words) - 1:
                to_write += ' '
        if n_best:
            print >>new_trans_file, full_trans_line[0].strip() + ' ||| ' + to_write + ' ||| ' + full_trans_line[2].strip()
        else:
            print >>new_trans_file, to_write

def parse_args():
    parser = argparse.ArgumentParser(
            "Replace UNK by original word")
    parser.add_argument("--state",
            required=True, help="State to use")
    parser.add_argument("--mapping",
            help="Top1 unigram mapping (Source to target)")
    parser.add_argument("--source",
            help="File of source sentences")
    parser.add_argument("--trans",
            help="File of translated sentences")
    parser.add_argument("--new-trans",
            help="File to save new translations in")
    parser.add_argument("--verbose",
            action="store_true", default=False,
            help="Be verbose")
    parser.add_argument("--heuristic", type=int, default=0,
            help="0: copy, 1: Use dict, 2: Use dict only if lowercase \
            Used only if a mapping is given. Default is 0.")
    parser.add_argument("--topn-file",
         type=str,
         help="Binarized topn list for each source word (Vocabularies must correspond)")
    parser.add_argument("--num-common",
         type=int,
         help="Number of always used common words (inc. <eos>, UNK) \
         (With --less-transfer, total number of words)")
    parser.add_argument("--num-ttables",
         type=int,
         help="Number of target words taken from the T-tables for each input word")
    parser.add_argument("--change-every", type=int, default=100,
            help="Change the dicts at each multiple of this number. \
            Use -1 to change only if full")
    parser.add_argument("--no-reset", action="store_true", default=False,
            help="Do not reset the dicts when changing vocabularies")
    parser.add_argument("--batchsize", type=int, default=32,
            help="(Maximum) batchsize")
    parser.add_argument("--n-best", action="store_true", default=False,
            help="Trans file is a n-best list, where lines look like \
                  `20 ||| A sentence . ||| 0.353`")
    parser.add_argument("--models", nargs = '+', required=True,
            help="path to the models")
    parser.add_argument("changes",
            nargs="?", default="",
            help="Changes to state")
    return parser.parse_args()

def main():
    args = parse_args()

    state = prototype_phrase_state()
    with open(args.state) as src:
        state.update(cPickle.load(src))
    state.update(eval("dict({})".format(args.changes)))

    logging.basicConfig(level=getattr(logging, state['level']), format="%(asctime)s: %(name)s: %(levelname)s: %(message)s")

    if 'rolling_vocab' not in state:
        state['rolling_vocab'] = 0
    if 'save_algo' not in state:
        state['save_algo'] = 0
    if 'save_gs' not in state:
        state['save_gs'] = 0
    if 'save_iter' not in state:
        state['save_iter'] = -1
    if 'var_src_len' not in state:
        state['var_src_len'] = False

    if args.num_common and args.num_ttables and args.topn_file:
        with open(args.topn_file, 'rb') as f:
            topn = cPickle.load(f) # Load dictionary (source word index : list of target word indices)
            for elt in topn:
                topn[elt] = topn[elt][:args.num_ttables] # Take the first args.num_ttables only

    num_models = len(args.models)
    rng = numpy.random.RandomState(state['seed'])
    enc_decs = []
    lm_models = []
    alignment_fns = []
    if args.num_common and args.num_ttables and args.topn_file:
        original_W_0_dec_approx_embdr = []
        original_W2_dec_deep_softmax = []
        original_b_dec_deep_softmax = []

    for i in xrange(num_models):
        enc_decs.append(RNNEncoderDecoder(state, rng, skip_init=True, compute_alignment=True))
        enc_decs[i].build()
        lm_models.append(enc_decs[i].create_lm_model())
        lm_models[i].load(args.models[i])

        alignment_fns.append(theano.function(inputs=enc_decs[i].inputs, outputs=[enc_decs[i].alignment], name="alignment_fn"))

        if args.num_common and args.num_ttables and args.topn_file:
            original_W_0_dec_approx_embdr.append(lm_models[i].params[lm_models[i].name2pos['W_0_dec_approx_embdr']].get_value())
            original_W2_dec_deep_softmax.append(lm_models[i].params[lm_models[i].name2pos['W2_dec_deep_softmax']].get_value())
            original_b_dec_deep_softmax.append(lm_models[i].params[lm_models[i].name2pos['b_dec_deep_softmax']].get_value())

            lm_models[i].params[lm_models[i].name2pos['W_0_dec_approx_embdr']].set_value(numpy.zeros((1,1), dtype=numpy.float32))
            lm_models[i].params[lm_models[i].name2pos['W2_dec_deep_softmax']].set_value(numpy.zeros((1,1), dtype=numpy.float32))
            lm_models[i].params[lm_models[i].name2pos['b_dec_deep_softmax']].set_value(numpy.zeros((1), dtype=numpy.float32))

    if args.mapping:
        with open(args.mapping, 'rb') as f:
            mapping = cPickle.load(f)
        heuristic = args.heuristic
    else:
        heuristic = 0
        mapping = None


    word2idx_src = cPickle.load(open(state['word_indx'], 'rb'))
    idict_src = cPickle.load(open(state['indx_word'], 'r'))

    word2idx_trg = cPickle.load(open(state['word_indx_trgt'], 'rb'))
    idict_trg = cPickle.load(open(state['indx_word_target'], 'r'))

    word2idx_trg['<eos>'] = state['null_sym_target']
    word2idx_trg[state['oov']] = state['unk_sym_target'] # 'UNK' may be in the vocabulary. Now points to the right index.
    idict_trg[state['null_sym_target']] = '<eos>'
    idict_trg[state['unk_sym_target']] = state['oov']

    if args.num_common and args.num_ttables and args.topn_file:

        # Use OrderedDict instead of set for reproducibility
        d = OrderedDict() # Up to now
        D = OrderedDict() # Full
        C = OrderedDict() # Allowed to reject
        prev_line = 0
        logger.info("%d" % prev_line)
        D_dict = OrderedDict()
        output = False

        for i in xrange(args.num_common):
            D[i] = 0
            C[i] = 0
        null_unk_indices = [state['null_sym_target'],state['unk_sym_target']]
        update_dicts(null_unk_indices, d, D, C, args.num_common)
        with open(args.source, 'r') as f:
            for i, line in enumerate(f):
                seqin = line.strip()
                seq, _ = parse_input(state, word2idx_src, seqin) # seq is the ndarray of indices
                indices = []
                for elt in seq[:-1]: # Exclude the EOL token
                    if elt != 1: # Exclude OOV (1 will not be a key of topn)
                        indices.extend(topn[elt]) # Add topn best unigram translations for each source word
                update_dicts(indices, d, D, C, args.num_common)
                if (i % args.change_every) == 0 and args.change_every > 0 and i > 0:
                    D_dict[prev_line] = D.copy() # Save dictionary for the lines preceding this one
                    prev_line = i
                    logger.info("%d" % i)
                    output = False
                    d = OrderedDict()
                    if args.no_reset:
                        C = D.copy()
                    else:
                        D = OrderedDict() # Full
                        C = OrderedDict() # Allowed to reject
                        for i in xrange(args.num_common):
                            D[i] = 0
                            C[i] = 0
                    null_unk_indices = [state['null_sym_target'], state['unk_sym_target']]
                    update_dicts(null_unk_indices, d, D, C, args.num_common)
                    update_dicts(indices, d, D, C, args.num_common) # Assumes you cannot fill d with only 1 line
            D_dict[prev_line] = D.copy()

    start_time = time.time()

    if args.source and args.trans and args.new_trans:
        with open(args.source, 'r') as src_file:
            with open(args.trans, 'r') as trans_file:
                with open(args.new_trans, 'w') as new_trans_file:
                    if not (args.num_common and args.num_ttables and args.topn_file):
                        eos_id = state['null_sym_target']
                        unk_id = state['unk_sym_target']
                        new_word2idx_trg = word2idx_trg

                    prev_i = -1
                    if args.n_best:
                        full_trans_line = trans_file.readline()
                        if full_trans_line == '':
                            raise IOError("File is empty")
                        full_trans_line = full_trans_line.split('|||')
                        n_best_start = int(full_trans_line[0].strip())
                        trans_file.seek(0)
                    while True:
                        if args.n_best:
                            full_trans_line = trans_file.readline()
                            if full_trans_line == '':
                                break
                            full_trans_line = full_trans_line.split('|||')
                            i = int(full_trans_line[0].strip()) - n_best_start
                            trans_line = full_trans_line[1].strip()
                        else:
                            trans_line = trans_file.readline()
                            if trans_line == '':
                                break
                            i = prev_i + 1

                        if i == (prev_i + 1):
                            prev_i = i

                            if (i % args.change_every) == 0 and i > 0:
                                hard_alignments = compute_alignment(src_seqs, trg_seqs, alignment_fns, args.batchsize)
                                replace_unknown_words(
                                    src_word_seqs, trg_seqs, trg_word_seqs,
                                    hard_alignments, heuristic, mapping, unk_id,
                                    new_trans_file, args.n_best, full_trans_lines)

                            if (i % 100 == 0) and i > 0:
                                new_trans_file.flush()
                                logger.debug("Current speed is {} per sentence".
                                        format((time.time() - start_time) / i))

                            src_line = src_file.readline()
                            src_seq, src_words = parse_input(state, word2idx_src, src_line.strip())
                            src_words.append('<eos>')

                            if (i % args.change_every) == 0:
                                src_seqs = []
                                src_word_seqs = []
                                trg_seqs = []
                                trg_word_seqs = []
                                full_trans_lines = [] # Only used with n-best lists
                                if args.num_common and args.num_ttables and args.topn_file:
                                    indices = D_dict[i].keys()
                                    eos_id = indices.index(state['null_sym_target']) # Find new eos and unk positions
                                    unk_id = indices.index(state['unk_sym_target'])
                                    for j in xrange(num_models):
                                        lm_models[j].params[lm_models[j].name2pos['W_0_dec_approx_embdr']].set_value(original_W_0_dec_approx_embdr[j][indices])
                                        lm_models[j].params[lm_models[j].name2pos['W2_dec_deep_softmax']].set_value(original_W2_dec_deep_softmax[j][:, indices])
                                        lm_models[j].params[lm_models[j].name2pos['b_dec_deep_softmax']].set_value(original_b_dec_deep_softmax[j][indices])
                                    new_word2idx_trg = dict([(idict_trg[index], k) for k, index in enumerate(indices)])
                        elif i != prev_i:
                            raise ValueError("prev_i: %d, i: %d" % (prev_i, i))

                        trans_seq, trans_words = parse_output(new_word2idx_trg, trans_line.strip(), eos_id=eos_id, unk_id=unk_id)
                        trans_words.append('<eos>')

                        src_seqs.append(src_seq)
                        src_word_seqs.append(src_words)
                        trg_seqs.append(trans_seq)
                        trg_word_seqs.append(trans_words)
                        if args.n_best:
                            full_trans_lines.append(full_trans_line)

                    # Out of loop
                    hard_alignments = compute_alignment(src_seqs, trg_seqs, alignment_fns, args.batchsize)
                    replace_unknown_words(src_word_seqs, trg_seqs, trg_word_seqs,
                                          hard_alignments, heuristic, mapping, unk_id,
                                          new_trans_file, args.n_best, full_trans_lines)
    else:
        raise NotImplementedError

if __name__ == "__main__":
    main()
