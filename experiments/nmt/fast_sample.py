#!/usr/bin/env python

import argparse
import cPickle
import traceback
import logging
import time
import sys

import numpy

import experiments.nmt
from experiments.nmt import\
    RNNEncoderDecoder,\
    prototype_phrase_state,\
    parse_input

from experiments.nmt.numpy_compat import argpartition

logger = logging.getLogger(__name__)

class Timer(object):

    def __init__(self):
        self.total = 0

    def start(self):
        self.start_time = time.time()

    def finish(self):
        self.total += time.time() - self.start_time

class BeamSearch(object):

    def __init__(self, enc_decs):
        self.enc_decs = enc_decs

    def compile(self):
        num_models = len(self.enc_decs)
        self.comp_repr = []
        self.comp_init_states = []
        self.comp_next_probs = []
        self.comp_next_states = []
        for i in xrange(num_models):
            self.comp_repr.append(self.enc_decs[i].create_representation_computer())
            self.comp_init_states.append(self.enc_decs[i].create_initializers())
            self.comp_next_probs.append(self.enc_decs[i].create_next_probs_computer())
            self.comp_next_states.append(self.enc_decs[i].create_next_states_computer())

    def search(self, seq, n_samples, eos_id, unk_id, ignore_unk=False, minlen=1, final=False):
        num_models = len(self.enc_decs)
        c = []
        for i in xrange(num_models):
            c.append(self.comp_repr[i](seq)[0])
        states = []
        for i in xrange(num_models):
            states.append(map(lambda x : x[None, :], self.comp_init_states[i](c[i])))
        dim = states[0][0].shape[1]

        num_levels = len(states[0])

        fin_trans = []
        fin_costs = []

        trans = [[]]
        costs = [0.0]

        for k in range(3 * len(seq)):
            if n_samples == 0:
                break

            # Compute probabilities of the next words for
            # all the elements of the beam.
            beam_size = len(trans)
            last_words = (numpy.array(map(lambda t : t[-1], trans))
                    if k > 0
                    else numpy.zeros(beam_size, dtype="int64"))
            #log_probs = (numpy.log(self.comp_next_probs_0(c, k, last_words, *states)[0]) +  numpy.log(self.comp_next_probs_1(c, k, last_words, *states)[0]))/2.
            log_probs = sum(numpy.log(self.comp_next_probs[i](c[i], k, last_words, *states[i])[0]) for i in xrange(num_models))/num_models

            # Adjust log probs according to search restrictions
            if ignore_unk:
                log_probs[:,unk_id] = -numpy.inf
            # TODO: report me in the paper!!!
            if k < minlen:
                log_probs[:,eos_id] = -numpy.inf

            # Find the best options by calling argpartition of flatten array
            next_costs = numpy.array(costs)[:, None] - log_probs
            flat_next_costs = next_costs.flatten()
            best_costs_indices = argpartition(
                    flat_next_costs.flatten(),
                    n_samples)[:n_samples]

            # Decypher flatten indices
            voc_size = log_probs.shape[1]
            trans_indices = best_costs_indices / voc_size
            word_indices = best_costs_indices % voc_size
            costs = flat_next_costs[best_costs_indices]

            # Form a beam for the next iteration
            new_trans = [[]] * n_samples
            new_costs = numpy.zeros(n_samples)
            new_states = []
            for i in xrange(num_models):
                new_states.append([numpy.zeros((n_samples, dim), dtype="float32") for level
                    in range(num_levels)])
            inputs = numpy.zeros(n_samples, dtype="int64")
            for i, (orig_idx, next_word, next_cost) in enumerate(
                    zip(trans_indices, word_indices, costs)):
                new_trans[i] = trans[orig_idx] + [next_word]
                new_costs[i] = next_cost
                for level in range(num_levels):
                    for j in xrange(num_models):
                        new_states[j][level][i] = states[j][level][orig_idx]
                inputs[i] = next_word
            for i in xrange(num_models):
                new_states[i]=self.comp_next_states[i](c[i], k, inputs, *new_states[i])

            # Filter the sequences that end with end-of-sequence character
            trans = []
            costs = []
            indices = []
            for i in range(n_samples):
                if new_trans[i][-1] != eos_id:
                    trans.append(new_trans[i])
                    costs.append(new_costs[i])
                    indices.append(i)
                else:
                    n_samples -= 1
                    fin_trans.append(new_trans[i])
                    fin_costs.append(new_costs[i])
            for i in xrange(num_models):
                states[i]=map(lambda x : x[indices], new_states[i])

        # Dirty tricks to obtain any translation
        if not len(fin_trans):
            if ignore_unk:
                logger.warning("Did not manage without UNK")
                return self.search(seq, n_samples, eos_id=eos_id, unk_id=unk_id, ignore_unk=False, minlen=minlen, final=final)
            elif not final:
                logger.warning("No appropriate translations: using larger vocabulary")
                raise RuntimeError
            else:
                logger.warning("No appropriate translation: return empty translation")
                fin_trans=[[]]
                fin_costs = [0.0]
                

        fin_trans = numpy.array(fin_trans)[numpy.argsort(fin_costs)]
        fin_costs = numpy.array(sorted(fin_costs))
        return fin_trans, fin_costs

def indices_to_words(i2w, seq):
    sen = []
    for k in xrange(len(seq)):
        if i2w[seq[k]] == '<eol>':
            break
        sen.append(i2w[seq[k]])
    return sen

def sample(lm_model, seq, n_samples, eos_id, unk_id,
        sampler=None, beam_search=None,
        ignore_unk=False, normalize=False,
        normalize_p = 1.0,
        alpha=1, verbose=False, final=False):
    if beam_search:
        sentences = []
        trans, costs = beam_search.search(seq, n_samples, eos_id=eos_id, unk_id=unk_id,
                ignore_unk=ignore_unk, minlen=len(seq) / 2, final=final)
        if normalize:
            counts = [len(s) for s in trans]
            costs = [co / ((max(cn,1))**normalize_p) for co, cn in zip(costs, counts)]
        for i in range(len(trans)):
            sen = indices_to_words(lm_model.word_indxs, trans[i]) # Make sure that indices_to_words has been changed
            sentences.append(" ".join(sen))
        for i in range(len(costs)):
            if verbose:
                print "{}: {}".format(costs[i], sentences[i])
        return sentences, costs, trans
    elif sampler:
        raise NotImplementedError
    else:
        raise Exception("I don't know what to do")


def parse_args():
    parser = argparse.ArgumentParser(
            "Sample (of find with beam-serch) translations from a translation model")
    parser.add_argument("--state",
            required=True, help="State to use")
    parser.add_argument("--beam-search",
            action="store_true", help="Beam size, turns on beam-search")
    parser.add_argument("--beam-size",
            type=int, help="Beam size")
    parser.add_argument("--ignore-unk",
            default=False, action="store_true",
            help="Ignore unknown words")
    parser.add_argument("--source",
            help="File of source sentences")
    parser.add_argument("--trans",
            help="File to save translations in")
    parser.add_argument("--normalize",
            action="store_true", default=False,
            help="Normalize log-prob with the word count")
    parser.add_argument("--normalize-p",
            type=float, default=1.0,
            help="Controls preference to longer output. Only used if `normalize` is true.")
    parser.add_argument("--verbose",
            action="store_true", default=False,
            help="Be verbose")
    parser.add_argument("--topn-file",
         type=str,
         help="Binarized topn list for each source word (Vocabularies must correspond)")
    parser.add_argument("--num-common",
         type=int,
         help="Number of always used common words (inc. <eos>, UNK)")
    parser.add_argument("--num-ttables",
         type=int,
         help="Number of target words taken from the T-tables for each input word")
    parser.add_argument("--final",
            action="store_true", default=False,
            help="Do not try to expand the vocabulary if a translation fails")
    parser.add_argument("--max-src-vocab", type=int,
            help="Maximum number of tokens in source vocab")
    parser.add_argument("--models", nargs = '+', required=True,
            help="path to the models")
    parser.add_argument("--changes",
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

    with open(args.topn_file, 'rb') as f:
        topn = cPickle.load(f) # Load dictionary (source word index : list of target word indices)
    for elt in topn:
        topn[elt] = set(topn[elt][:args.num_ttables]) # Take the first args.num_ttables only and convert list to set
 
    num_models = len(args.models)
    rng = numpy.random.RandomState(state['seed'])
    enc_decs = []
    lm_models = []
    for i in xrange(num_models):
        enc_decs.append(RNNEncoderDecoder(state, rng, skip_init=True))
        enc_decs[i].build()
        lm_models.append(enc_decs[i].create_lm_model())
        lm_models[i].load(args.models[i])

    #enc_dec_0 = RNNEncoderDecoder(state, rng, skip_init=True)
    #enc_dec_0.build()
    #lm_model_0 = enc_dec_0.create_lm_model()
    #lm_model_0.load(args.model_0)

    indx_word = cPickle.load(open(state['word_indx'],'rb')) #Source w2i
    if args.max_src_vocab:
        for elt in indx_word:
            if indx_word[elt] >= max_source_vocab:
                del indx_word[elt]

    sampler = None
    beam_search = None
    if args.beam_search:
        beam_search = BeamSearch(enc_decs)
        beam_search.compile()
    else:
        raise NotImplementedError
        #sampler = enc_dec.create_sampler(many_samples=True)

    idict_src = cPickle.load(open(state['indx_word'],'r')) #Source i2w
    if args.max_src_vocab:
        for elt in idict_src:
            if elt >= max_source_vocab:
                del idict_src[elt]
        lm_models[0].params[lm_models[0].name2pos['W_0_enc_approx_embdr']].set_value(lm_models[0].params[lm_models[0].name2pos['W_0_enc_approx_embdr']].get_value()[:max_source_vocab])
    
    original_target_i2w = lm_models[0].word_indxs.copy()
    # I don't think that we need target_word2index
    
    original_W_0_dec_approx_embdr = [lm_models[i].params[lm_models[i].name2pos['W_0_dec_approx_embdr']].get_value() for i in xrange(num_models)]
    original_W2_dec_deep_softmax = [lm_models[i].params[lm_models[i].name2pos['W2_dec_deep_softmax']].get_value() for i in xrange(num_models)]
    original_b_dec_deep_softmax = [lm_models[i].params[lm_models[i].name2pos['b_dec_deep_softmax']].get_value() for i in xrange(num_models)]
    #original_W_0_dec_approx_embdr_0 = lm_model_0.params[lm_model_0.name2pos['W_0_dec_approx_embdr']].get_value()
    #original_W2_dec_deep_softmax_0 = lm_model_0.params[lm_model_0.name2pos['W2_dec_deep_softmax']].get_value()
    #original_b_dec_deep_softmax_0 = lm_model_0.params[lm_model_0.name2pos['b_dec_deep_softmax']].get_value()

    max_words = len(original_b_dec_deep_softmax[0])
 
    if args.source and args.trans:
        # Actually only beam search is currently supported here
        assert beam_search
        assert args.beam_size

        fsrc = open(args.source, 'r')
        ftrans = open(args.trans, 'w')

        start_time = time.time()

        n_samples = args.beam_size
        total_cost = 0.0
        logging.debug("Beam size: {}".format(n_samples))
        for i, line in enumerate(fsrc):
            seqin = line.strip()
            seq, parsed_in = parse_input(state, indx_word, seqin, idx2word=idict_src) # seq is the ndarray of indices
            # For now, keep all input words in the model.
            # In the future, we may want to filter them to save on memory, but this isn't really much of an issue now
            if args.verbose:
                print "Parsed Input:", parsed_in
            # Extract the indices you need
            indices = set()
            for elt in seq[:-1]: # Exclude the EOL token
                if elt != 1: # Exclude OOV (1 will not be a key of topn)
                    indices = indices.union(topn[elt]) # Add topn best unigram translations for each source word
            num_common_words = args.num_common
            while True:
                if num_common_words >= max_words:
                    final = True
                    num_common_words = max_words
                else:
                    final = False

                if args.final: # No matter the number of words
                    final = True
                indices = indices.union(set(xrange(num_common_words))) # Add common words
                indices = list(indices) # Convert back to list for advanced indexing
                eos_id = indices.index(state['null_sym_target']) # Find new eos and unk positions
                unk_id = indices.index(state['unk_sym_target'])
                # Set the target word matrices and biases
                for j in xrange(num_models):
                    lm_models[j].params[lm_models[j].name2pos['W_0_dec_approx_embdr']].set_value(original_W_0_dec_approx_embdr[j][indices])
                    lm_models[j].params[lm_models[j].name2pos['W2_dec_deep_softmax']].set_value(original_W2_dec_deep_softmax[j][:, indices])
                    lm_models[j].params[lm_models[j].name2pos['b_dec_deep_softmax']].set_value(original_b_dec_deep_softmax[j][indices])
                lm_models[0].word_indxs = dict([(k, original_target_i2w[index]) for k, index in enumerate(indices)]) # target index2word

                try:
                    trans, costs, _ = sample(lm_models[0], seq, n_samples, sampler=sampler,
                            beam_search=beam_search, ignore_unk=args.ignore_unk, normalize=args.normalize,
                            normalize_p=args.normalize_p, eos_id=eos_id, unk_id=unk_id, final=final)
                    break # Breaks only if it succeeded (If final=True, wil always succeed)
                except RuntimeError:
                    indices = set(indices)
                    num_common_words *= 2
            best = numpy.argmin(costs)
            print >>ftrans, trans[best]
            if args.verbose:
                print "Translation:", trans[best]
            total_cost += costs[best]
            if (i + 1)  % 100 == 0:
                ftrans.flush()
                logger.debug("Current speed is {} per sentence".
                        format((time.time() - start_time) / (i + 1)))
        print "Total cost of the translations: {}".format(total_cost)

        fsrc.close()
        ftrans.close()
    else:
        raise NotImplementedError

if __name__ == "__main__":
    main()
