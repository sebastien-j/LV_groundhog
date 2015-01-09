#!/usr/bin/env python

import argparse
import cPickle
import logging
import pprint
import shelve

import numpy

from groundhog.utils import sample_zeros, sample_weights_orth, init_bias, sample_weights_classic
from groundhog.utils import replace_array
from groundhog.trainer.SGD_adadelta import SGD as SGD_adadelta
from groundhog.trainer.SGD import SGD as SGD
from groundhog.trainer.SGD_momentum import SGD as SGD_momentum
from groundhog.mainLoop import MainLoop
from experiments.nmt import\
        RNNEncoderDecoder, prototype_search_state, get_batch_iterator
import experiments.nmt

logger = logging.getLogger(__name__)

class RandomSamplePrinter(object):

    def __init__(self, state, model, train_iter):
        args = dict(locals())
        args.pop('self')
        self.__dict__.update(**args)

    def __call__(self):
        def cut_eol(words):
            for i, word in enumerate(words):
                if words[i] == '<eol>':
                    return words[:i + 1]
            raise Exception("No end-of-line found")

        sample_idx = 0
        while sample_idx < self.state['n_examples']:
            batch = self.train_iter.next(peek=True)
            xs, ys = batch['x'], batch['y']
            if self.state['rolling_vocab']:
                small_xs = replace_array(xs, self.model.large2small_src)
                small_ys = replace_array(ys, self.model.large2small_trgt)
            for seq_idx in range(xs.shape[1]):
                if sample_idx == self.state['n_examples']:
                    break

                x, y = xs[:, seq_idx], ys[:, seq_idx]
                if self.state['rolling_vocab']:
                    small_x = small_xs[:, seq_idx]
                    small_y = small_ys[:, seq_idx]
                    x_words = cut_eol(map(lambda w_idx : self.model.large2word_src[w_idx], x))
                    y_words = cut_eol(map(lambda w_idx : self.model.large2word_trgt[w_idx], y))
                    #Alternatively
                    x_words_alt = cut_eol(map(lambda w_idx : self.model.word_indxs_src[w_idx], small_x))
                    y_words_alt = cut_eol(map(lambda w_idx : self.model.word_indxs[w_idx], small_y))
                    if (x_words == x_words_alt) and (y_words == y_words_alt):
                        logger.debug("OK. Small and large index2word match.")
                    else:
                        logger.error("Small and large index2word DO NOT MATCH.")
                else:
                    x_words = cut_eol(map(lambda w_idx : self.model.word_indxs_src[w_idx], x))
                    y_words = cut_eol(map(lambda w_idx : self.model.word_indxs[w_idx], y))
                if len(x_words) == 0:
                    continue

                print "Input: {}".format(" ".join(x_words))
                print "Target: {}".format(" ".join(y_words))
                if self.state['rolling_vocab']:
                    self.model.get_samples(self.state['seqlen'] + 1, self.state['n_samples'], small_x[:len(x_words)])
                else:
                    self.model.get_samples(self.state['seqlen'] + 1, self.state['n_samples'], x[:len(x_words)])
                sample_idx += 1

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", help="State to use")
    parser.add_argument("--proto",  default="prototype_search_state",
        help="Prototype state to use for state")
    parser.add_argument("--skip-init", action="store_true",
        help="Skip parameter initilization")
    parser.add_argument("changes",  nargs="*", help="Changes to state", default="")
    return parser.parse_args()

def init_extra_parameters(model, state): # May want to add skip_init later
    model.large_W_0_enc_approx_embdr = eval(state['weight_init_fn'])(state['large_vocab_source'], state['rank_n_approx'], -1, state['weight_scale'], model.rng)
    model.large_W_0_dec_approx_embdr = eval(state['weight_init_fn'])(state['large_vocab_target'], state['rank_n_approx'], -1, state['weight_scale'], model.rng)
    model.large_W2_dec_deep_softmax = eval(state['weight_init_fn'])(state['rank_n_approx'], state['large_vocab_target'], -1, state['weight_scale'], model.rng)
    model.large_b_dec_deep_softmax = init_bias(state['large_vocab_target'], 0., model.rng)

def init_adadelta_extra_parameters(algo, state):
    algo.large_W_0_enc_approx_embdr_g2 = sample_zeros(algo.state['large_vocab_source'], algo.state['rank_n_approx'], -1, algo.state['weight_scale'], algo.rng)
    algo.large_W_0_enc_approx_embdr_d2 = sample_zeros(algo.state['large_vocab_source'], algo.state['rank_n_approx'], -1, algo.state['weight_scale'], algo.rng)
    algo.large_W_0_dec_approx_embdr_g2 = sample_zeros(algo.state['large_vocab_target'], algo.state['rank_n_approx'], -1, algo.state['weight_scale'], algo.rng)
    algo.large_W_0_dec_approx_embdr_d2 = sample_zeros(algo.state['large_vocab_target'], algo.state['rank_n_approx'], -1, algo.state['weight_scale'], algo.rng)
    algo.large_W2_dec_deep_softmax_g2 = sample_zeros(algo.state['rank_n_approx'], algo.state['large_vocab_target'], -1, algo.state['weight_scale'], algo.rng)
    algo.large_W2_dec_deep_softmax_d2 = sample_zeros(algo.state['rank_n_approx'], algo.state['large_vocab_target'], -1, algo.state['weight_scale'], algo.rng)
    algo.large_b_dec_deep_softmax_g2 = init_bias(algo.state['large_vocab_target'], 0., algo.rng)
    algo.large_b_dec_deep_softmax_d2 = init_bias(algo.state['large_vocab_target'], 0., algo.rng)
    if state['save_gs']:
        algo.large_W_0_enc_approx_embdr_gs = sample_zeros(algo.state['large_vocab_source'], algo.state['rank_n_approx'], -1, algo.state['weight_scale'], algo.rng)
        algo.large_W_0_dec_approx_embdr_gs = sample_zeros(algo.state['large_vocab_target'], algo.state['rank_n_approx'], -1, algo.state['weight_scale'], algo.rng)
        algo.large_W2_dec_deep_softmax_gs = sample_zeros(algo.state['rank_n_approx'], algo.state['large_vocab_target'], -1, algo.state['weight_scale'], algo.rng)
        algo.large_b_dec_deep_softmax_gs = init_bias(algo.state['large_vocab_target'], 0., algo.rng)


def main():
    args = parse_args()

    state = getattr(experiments.nmt, args.proto)()
    if args.state:
        if args.state.endswith(".py"):
            state.update(eval(open(args.state).read()))
        else:
            with open(args.state) as src:
                state.update(cPickle.load(src))
    for change in args.changes:
        state.update(eval("dict({})".format(change)))

    logging.basicConfig(level=getattr(logging, state['level']), format="%(asctime)s: %(name)s: %(levelname)s: %(message)s")
    logger.debug("State:\n{}".format(pprint.pformat(state)))

    if 'rolling_vocab' not in state:
        state['rolling_vocab'] = 0
    if 'save_algo' not in state:
        state['save_algo'] = 0
    if 'save_gs' not in state:
        state['save_gs'] = 0
    if 'fixed_embeddings' not in state:
        state['fixed_embeddings'] = False
    if 'save_iter' not in state:
        state['save_iter'] = -1
    if 'var_src_len' not in state:
        state['var_src_len'] = False

    rng = numpy.random.RandomState(state['seed'])
    enc_dec = RNNEncoderDecoder(state, rng, args.skip_init)
    enc_dec.build()
    lm_model = enc_dec.create_lm_model()

    logger.debug("Load data")
    train_data = get_batch_iterator(state, rng)
    logger.debug("Compile trainer")
    algo = eval(state['algo'])(lm_model, state, train_data)

    if state['rolling_vocab']:
        logger.debug("Initializing extra parameters")
        init_extra_parameters(lm_model, state)
        if not state['fixed_embeddings']:
            init_adadelta_extra_parameters(algo, state)
        with open(state['rolling_vocab_dict'], 'rb') as f:
            lm_model.rolling_vocab_dict = cPickle.load(f)
        lm_model.total_num_batches = max(lm_model.rolling_vocab_dict)
        lm_model.Dx_shelve = shelve.open(state['Dx_file'])
        lm_model.Dy_shelve = shelve.open(state['Dy_file'])

    logger.debug("Run training")
    main = MainLoop(train_data, None, None, lm_model, algo, state, None,
            reset=state['reset'],
            hooks=[RandomSamplePrinter(state, lm_model, train_data)]
                if state['hookFreq'] >= 0
                else None)
    if state['reload']:
        main.load()
    if state['loopIters'] > 0:
        main.main()
    if state['rolling_vocab']:
        lm_model.Dx_shelve.close()
        lm_model.Dy_shelve.close()

if __name__ == "__main__":
    main()
