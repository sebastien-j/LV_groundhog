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

    def __init__(self, enc_dec):
        self.enc_dec = enc_dec
        state = self.enc_dec.state
        # Do this elsewhere
        #self.eos_id = state['null_sym_target']
        #self.unk_id = state['unk_sym_target']

    def compile(self):
        self.comp_repr = self.enc_dec.create_representation_computer()
        self.comp_init_states = self.enc_dec.create_initializers()
        self.comp_next_probs = self.enc_dec.create_next_probs_computer()
        self.comp_next_states = self.enc_dec.create_next_states_computer()

    def search(self, seq, n_samples, eos_id, unk_id, ignore_unk=False, minlen=1):
        c = self.comp_repr(seq)[0]
        states = map(lambda x : x[None, :], self.comp_init_states(c))
        dim = states[0].shape[1]

        num_levels = len(states)

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
            log_probs = numpy.log(self.comp_next_probs(c, k, last_words, *states)[0])

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
            new_states = [numpy.zeros((n_samples, dim), dtype="float32") for level
                    in range(num_levels)]
            inputs = numpy.zeros(n_samples, dtype="int64")
            for i, (orig_idx, next_word, next_cost) in enumerate(
                    zip(trans_indices, word_indices, costs)):
                new_trans[i] = trans[orig_idx] + [next_word]
                new_costs[i] = next_cost
                for level in range(num_levels):
                    new_states[level][i] = states[level][orig_idx]
                inputs[i] = next_word
            new_states = self.comp_next_states(c, k, inputs, *new_states)

            # Filter the sequences that end with end-of-sequence character
            trans = []
            costs = []
            indices = []
            for i in range(n_samples):
                if new_trans[i][-1] != self.enc_dec.state['null_sym_target']:
                    trans.append(new_trans[i])
                    costs.append(new_costs[i])
                    indices.append(i)
                else:
                    n_samples -= 1
                    fin_trans.append(new_trans[i])
                    fin_costs.append(new_costs[i])
            states = map(lambda x : x[indices], new_states)

        # Dirty tricks to obtain any translation
        if not len(fin_trans):
            if ignore_unk:
                logger.warning("Did not manage without UNK")
                return self.search(seq, n_samples, False, minlen)
            elif n_samples < 500:
                #logger.warning("Still no translations: try beam size {}".format(n_samples * 2))
                #return self.search(seq, n_samples * 2, False, minlen)
                logger.warning("No appropriate translation")
                fin_trans=[[]]
                fin_costs = [0.0]
            else:
                logger.error("Translation failed")

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
        alpha=1, verbose=False):
    if beam_search:
        sentences = []
        trans, costs = beam_search.search(seq, n_samples, eos_id=eos_id, unk_id=unk_id,
                ignore_unk=ignore_unk, minlen=len(seq) / 2)
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
    parser.add_argument("model_path",
            help="Path to the model")
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

    with open(args.topn_file, 'rb') as f:
        topn = cPickle.load(f) # Load dictionary (source word index : list of target word indices)
    for elt in topn:
        topn[elt] = set(topn[elt][:args.num_ttables]) # Take the first args.num_ttables only and convert list to set
 
    rng = numpy.random.RandomState(state['seed'])
    enc_dec = RNNEncoderDecoder(state, rng, skip_init=True)
    enc_dec.build()
    lm_model = enc_dec.create_lm_model()
    lm_model.load(args.model_path)
    indx_word = cPickle.load(open(state['word_indx'],'rb')) #Source w2i

    sampler = None
    beam_search = None
    if args.beam_search:
        beam_search = BeamSearch(enc_dec)
        beam_search.compile()
    else:
        raise NotImplementedError
        #sampler = enc_dec.create_sampler(many_samples=True)

    idict_src = cPickle.load(open(state['indx_word'],'r')) #Source i2w
    
    original_target_i2w = lm_model.word_indxs.copy()
    # I don't think that we need target_word2index
    
    original_W_0_dec_approx_embdr = lm_model.params[lm_model.name2pos['W_0_dec_approx_embdr']].get_value()
    original_W2_dec_deep_softmax = lm_model.params[lm_model.name2pos['W2_dec_deep_softmax']].get_value()
    original_b_dec_deep_softmax = lm_model.params[lm_model.name2pos['b_dec_deep_softmax']].get_value()
 
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
                indices = indices.union(topn[elt]) # Add topn best unigram translations for each source word
            indices = indices.union(set(xrange(args.num_common))) # Add common words
            indices = list(indices) # Convert back to list for advanced indexing
            eos_id = indices.index(state['null_sym_target']) # Find new eos and unk positions
            unk_id = indices.index(state['unk_sym_target'])
            # Set the target word matrices and biases
            lm_model.params[lm_model.name2pos['W_0_dec_approx_embdr']].set_value(original_W_0_dec_approx_embdr[indices])
            lm_model.params[lm_model.name2pos['W2_dec_deep_softmax']].set_value(original_W2_dec_deep_softmax[:, indices])
            lm_model.params[lm_model.name2pos['b_dec_deep_softmax']].set_value(original_b_dec_deep_softmax[indices])
            lm_model.word_indxs = dict([(i, original_target_i2w[index]) for i, index in enumerate(indices)]) # target index2word
            trans, costs, _ = sample(lm_model, seq, n_samples, sampler=sampler,
                    beam_search=beam_search, ignore_unk=args.ignore_unk, normalize=args.normalize,
                    normalize_p=args.normalize_p, eos_id=eos_id, unk_id=unk_id)
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
