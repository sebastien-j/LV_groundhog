"""
Main loop (early stopping).


TODO: write more documentation
"""
__docformat__ = 'restructedtext en'
__authors__ = ("Razvan Pascanu "
               "KyungHyun Cho "
               "Caglar Gulcehre ")
__contact__ = "Razvan Pascanu <r.pascanu@gmail>"


class Unbuffered:
    def __init__(self, stream):
        self.stream = stream

    def write(self, data):
        self.stream.write(data)
        self.stream.flush()

    def __getattr__(self, attr):
        return getattr(self.stream, attr)

import sys
import traceback
sys.stdout = Unbuffered(sys.stdout)

# Generic imports
import numpy
import cPickle
import gzip
import time
import signal
import logging

from groundhog.utils import print_mem, print_time
from groundhog.utils import invert_dict

logger = logging.getLogger(__name__)

class MainLoop(object):
    def __init__(self,
                 train_data,
                 valid_data,
                 test_data,
                 model,
                 algo,
                 state,
                 channel,
                 hooks=None,
                 reset=-1,
                 train_cost=False,
                 validate_postprocess=None,
                 l2_params=False):
        """
        :type train_data: groundhog dataset object
        :param train_data: data iterator used for training

        :type valid_data: groundhog dataset object
        :param valid_data: data iterator used for validation

        :type test_data: groundhog dataset object
        :param test_data: data iterator used for testing

        :type model: groundhog model object
        :param model: the model that is supposed to be trained

        :type algo: groundhog trainer object
        :param algo: optimization algorithm used to optimized the model

        :type state: dictionary (or jobman dictionary)
        :param state: dictionary containing various hyper-param choices,
            but also the current state of the job (the dictionary is used by
            jobman to fill in a psql table)

        :type channel: jobman handler
        :param channel: jobman handler used to communicate with a psql
            server

        :type hooks: function or list of functions
        :param hooks: list of functions that are called every `hookFreq`
            steps to carry on various diagnostics

        :type reset: int
        :param reset: if larger than 0, the train_data iterator position is
            reseted to 0 every `reset` number of updates

        :type train_cost: bool
        :param train_cost: flag saying if the training error (over the
            entire training set) should be computed every time the validation
            error is computed

        :type validate_postprocess: None or function
        :param validate_postprocess: function called on the validation cost
            every time before applying the logic of the early stopper

        :type l2_params: bool
        :param l2_params: save parameter norms at each step
        """
        ###################
        # Step 0. Set parameters
        ###################
        self.train_data = train_data
        self.valid_data = valid_data
        self.test_data = test_data
        self.state = state
        self.channel = channel
        self.model = model
        self.algo = algo
        self.valid_id = 0
        self.old_cost = 1e21
        self.validate_postprocess = validate_postprocess
        self.patience = state['patience']
        self.l2_params = l2_params

        self.train_cost = train_cost

        if hooks and not isinstance(hooks, (list, tuple)):
            hooks = [hooks]

        if self.state['validFreq'] < 0:
            self.state['validFreq'] = self.train_data.get_length()
            print 'Validation computed every', self.state['validFreq']
        elif self.state['validFreq'] > 0:
            print 'Validation computed every', self.state['validFreq']
        if self.state['trainFreq'] < 0:
            self.state['trainFreq'] = self.train_data.get_length()
            print 'Train frequency set to ', self.state['trainFreq']

        state['bvalidcost'] = 1e21
        for (pname, _) in model.properties:
            self.state[pname] = 1e20

        n_elems = state['loopIters'] // state['trainFreq'] + 1
        if self.state['rolling_vocab']:
            self.timings = {'step' : 0, 'super_step' : 0, 'next_offset' : -1}
        else:
            self.timings = {'step' : 0, 'next_offset' : -1}
        for name in self.algo.return_names:
            self.timings[name] = numpy.zeros((n_elems,), dtype='float32')
        if self.l2_params:
            for param in model.params:
                self.timings["l2_" + param.name] = numpy.zeros(n_elems, dtype="float32")
        n_elems = state['loopIters'] // state['validFreq'] + 1
        for pname in model.valid_costs:
            self.state['valid'+pname] = 1e20
            self.state['test'+pname] = 1e20
            self.timings['fulltrain'+pname] = numpy.zeros((n_elems,),
                                                          dtype='float32')
            self.timings['valid'+pname] = numpy.zeros((n_elems,),
                                                      dtype='float32')
            self.timings['test'+pname] = numpy.zeros((n_elems,),
                                                     dtype='float32')
        if self.channel is not None:
            self.channel.save()

        self.hooks = hooks
        self.reset = reset

        self.start_time = time.time()
        self.batch_start_time = time.time()

    def validate(self):
        rvals = self.model.validate(self.valid_data)
        msg = '**  %d     validation:' % self.valid_id
        self.valid_id += 1
        self.batch_start_time = time.time()
        pos = self.step // self.state['validFreq']
        for k, v in rvals:
            msg = msg + ' ' + k + ':%f ' % float(v)
            self.timings['valid'+k][pos] = float(v)
            self.state['valid'+k] = float(v)
        msg += 'whole time %s' % print_time(time.time() - self.start_time)
        msg += ' patience %d' % self.patience
        print msg

        if self.train_cost:
            valid_rvals = rvals
            rvals = self.model.validate(self.train_data, True)
            msg = '**  %d     train:' % (self.valid_id - 1)
            for k, v in rvals:
                msg = msg + ' ' + k + ':%6.3f ' % float(v)
                self.timings['fulltrain' + k] = float(v)
                self.state['fulltrain' + k] = float(v)
            print msg
            rvals = valid_rvals

        self.state['validtime'] = float(time.time() - self.start_time)/60.
        # Just pick the first thing that the cost returns
        cost = rvals[0][1]
        if self.state['bvalidcost'] > cost:
            self.state['bvalidcost'] = float(cost)
            for k, v in rvals:
                self.state['bvalid'+k] = float(v)
            self.state['bstep'] = int(self.step)
            self.state['btime'] = int(time.time() - self.start_time)
            self.test()
        else:
            print 'No testing', cost, '>', self.state['bvalidcost']
            for k, v in self.state.items():
                if 'test' in k:
                    print k, v
        print_mem('validate')
        if self.validate_postprocess:
            return self.validate_postprocess(cost)
        return cost

    def test(self):
        self.model.best_params = [(x.name, x.get_value()) for x in
                                  self.model.params]
        numpy.savez(self.state['prefix'] + '_best_params',
                    **dict(self.model.best_params))
        self.state['best_params_pos'] = self.step
        if self.test_data is not None:
            rvals = self.model.validate(self.test_data)
        else:
            rvals = []
        msg = '>>>         Test'
        pos = self.step // self.state['validFreq']
        for k, v in rvals:
            msg = msg + ' ' + k + ':%6.3f ' % v
            self.timings['test' + k][pos] = float(v)
            self.state['test' + k] = float(v)
        print msg
        self.state['testtime'] = float(time.time()-self.start_time)/60.

    def save(self):
        start = time.time()
        print "Saving the model..."

        # ignore keyboard interrupt while saving
        s = signal.signal(signal.SIGINT, signal.SIG_IGN)
        if self.state['overwrite']:
            numpy.savez(self.state['prefix']+'timing.npz',
                        **self.timings)
            self.model.save(self.state['prefix']+'model.npz')
            if self.state['algo'] == 'SGD_adadelta' and self.state['save_algo']:
                self.algo.save(self.state['prefix']+'algo.npz')
            if self.state['rolling_vocab'] and not self.state['fixed_embeddings']:
                self.save_large_params(self.state['prefix']+'large.npz')
        else:
            numpy.savez(self.state['prefix']+'timing' + str(self.save_iter) + '.npz',
                        **self.timings)
            self.model.save(self.state['prefix'] +
                            'model%d.npz' % self.save_iter)
            if self.state['algo'] == 'SGD_adadelta' and self.state['save_algo']:
                self.algo.save(self.state['prefix']+'algo%d.npz' % self.save_iter)
            if self.state['rolling_vocab'] and not self.state['fixed_embeddings']:
                self.save_large_params(self.state['prefix']+'large%d.npz' % self.save_iter)
        cPickle.dump(self.state, open(self.state['prefix']+'state.pkl', 'w'))
        self.save_iter += 1
        self.state['save_iter'] = self.save_iter # Increment after saving only
        signal.signal(signal.SIGINT, s)

        print "Model saved, took {}".format(time.time() - start)

    # FIXME
    def load(self, model_path=None, timings_path=None, algo_path=None, large_path=None):
        self.save_iter = self.state['save_iter']
        if model_path is None:
            if not self.state['overwrite']:
                model_path = self.state['prefix'] + 'model' + str(self.save_iter) + '.npz'
            else:
                model_path = self.state['prefix'] + 'model.npz'
        if timings_path is None:
            if not self.state['overwrite']:
                timings_path = self.state['prefix'] + 'timing' + str(self.save_iter) + '.npz'
            else:
                timings_path = self.state['prefix'] + 'timing.npz'
        if self.state['save_algo']:
            if algo_path is None:
                if not self.state['overwrite']:
                    algo_path = self.state['prefix'] + 'algo' + str(self.save_iter) + '.npz'
                else:
                    algo_path = self.state['prefix'] + 'algo.npz'
        if self.state['rolling_vocab']:
            if large_path is None:
                if not self.state['overwrite']:
                    large_path = self.state['prefix'] + 'large' + str(self.save_iter) + '.npz'
                else:
                    large_path = self.state['prefix'] + 'large.npz'
        try:
            self.model.load(model_path)
        except Exception:
            print 'mainLoop: Corrupted model file'
            traceback.print_exc()
        try:
            self.timings = dict(numpy.load(timings_path).iteritems())
        except Exception:
            print 'mainLoop: Corrupted timings file'
            traceback.print_exc()
        if self.state['save_algo']:
            try:
                self.algo.load(algo_path)
            except Exception:
                print 'mainLoop: Corrupted algo file'
                traceback.print_exc()
        if self.state['rolling_vocab']:
            try:
                self.load_large_params(large_path)
            except Exception:
                print 'mainLoop: Corrupted large parameters file'
                traceback.print_exc()

    def save_large_params(self, filename):
        """
        Save the large vocabulary params (and adadelta params) to file `filename`
        """
        vals = {}
        vals['large_W_0_enc_approx_embdr'] = self.model.large_W_0_enc_approx_embdr
        vals['large_W_0_dec_approx_embdr'] = self.model.large_W_0_dec_approx_embdr
        vals['large_W2_dec_deep_softmax'] = self.model.large_W2_dec_deep_softmax
        vals['large_b_dec_deep_softmax'] = self.model.large_b_dec_deep_softmax

        if self.state['save_algo']:
            vals['large_W_0_enc_approx_embdr_g2'] = self.algo.large_W_0_enc_approx_embdr_g2
            vals['large_W_0_dec_approx_embdr_g2'] = self.algo.large_W_0_dec_approx_embdr_g2
            vals['large_W2_dec_deep_softmax_g2'] = self.algo.large_W2_dec_deep_softmax_g2
            vals['large_b_dec_deep_softmax_g2'] = self.algo.large_b_dec_deep_softmax_g2

            vals['large_W_0_enc_approx_embdr_d2'] = self.algo.large_W_0_enc_approx_embdr_d2
            vals['large_W_0_dec_approx_embdr_d2'] = self.algo.large_W_0_dec_approx_embdr_d2
            vals['large_W2_dec_deep_softmax_d2'] = self.algo.large_W2_dec_deep_softmax_d2
            vals['large_b_dec_deep_softmax_d2'] = self.algo.large_b_dec_deep_softmax_d2

            if self.state['save_gs']:
                vals['large_W_0_enc_approx_embdr_gs'] = self.algo.large_W_0_enc_approx_embdr_gs
                vals['large_W_0_dec_approx_embdr_gs'] = self.algo.large_W_0_dec_approx_embdr_gs
                vals['large_W2_dec_deep_softmax_gs'] = self.algo.large_W2_dec_deep_softmax_gs
                vals['large_b_dec_deep_softmax_gs'] = self.algo.large_b_dec_deep_softmax_gs

        numpy.savez(filename, **vals)

    def load_large_params(self, filename):
        """
        Load the large vocabulary params (and adadelta params) from file `filename`
        """
        vals = numpy.load(filename)
        self.model.large_W_0_enc_approx_embdr = vals['large_W_0_enc_approx_embdr']
        self.model.large_W_0_dec_approx_embdr = vals['large_W_0_dec_approx_embdr']
        self.model.large_W2_dec_deep_softmax = vals['large_W2_dec_deep_softmax']
        self.model.large_b_dec_deep_softmax = vals['large_b_dec_deep_softmax']

        if self.state['save_algo'] and not self.state['fixed_embeddings']:
            self.algo.large_W_0_enc_approx_embdr_g2 = vals['large_W_0_enc_approx_embdr_g2']
            self.algo.large_W_0_dec_approx_embdr_g2 = vals['large_W_0_dec_approx_embdr_g2']
            self.algo.large_W2_dec_deep_softmax_g2 = vals['large_W2_dec_deep_softmax_g2']
            self.algo.large_b_dec_deep_softmax_g2 = vals['large_b_dec_deep_softmax_g2']

            self.algo.large_W_0_enc_approx_embdr_d2 = vals['large_W_0_enc_approx_embdr_d2']
            self.algo.large_W_0_dec_approx_embdr_d2 = vals['large_W_0_dec_approx_embdr_d2']
            self.algo.large_W2_dec_deep_softmax_d2 = vals['large_W2_dec_deep_softmax_d2']
            self.algo.large_b_dec_deep_softmax_d2 = vals['large_b_dec_deep_softmax_d2']

            if self.state['save_gs']:
                self.algo.large_W_0_enc_approx_embdr_gs = vals['large_W_0_enc_approx_embdr_gs']
                self.algo.large_W_0_dec_approx_embdr_gs = vals['large_W_0_dec_approx_embdr_gs']
                self.algo.large_W2_dec_deep_softmax_gs = vals['large_W2_dec_deep_softmax_gs']
                self.algo.large_b_dec_deep_softmax_gs = vals['large_b_dec_deep_softmax_gs']

    def roll_vocab_small2large(self):
        # Transfer from small to large parameters
        logger.debug("Called roll_vocab_small2large()")

        temp = self.model.params[self.model.name2pos['W_0_enc_approx_embdr']].get_value()
        if not self.state['fixed_embeddings']:
            temp_g2 = self.algo.gnorm2[self.model.name2pos['W_0_enc_approx_embdr']].get_value()
            temp_d2 = self.algo.dnorm2[self.model.name2pos['W_0_enc_approx_embdr']].get_value()
            if self.state['save_gs']:
                temp_gs = self.algo.gs[self.model.name2pos['W_0_enc_approx_embdr']].get_value()
        for large in self.model.large2small_src:
            small = self.model.large2small_src[large]
            self.model.large_W_0_enc_approx_embdr[large] = temp[small]
            if not self.state['fixed_embeddings']:
                self.algo.large_W_0_enc_approx_embdr_g2[large] = temp_g2[small]
                self.algo.large_W_0_enc_approx_embdr_d2[large] = temp_d2[small]
                if self.state['save_gs']:
                    self.algo.large_W_0_enc_approx_embdr_gs[large] = temp_gs[small]

        temp = self.model.params[self.model.name2pos['W_0_dec_approx_embdr']].get_value()
        if not self.state['fixed_embeddings']:
            temp_g2 = self.algo.gnorm2[self.model.name2pos['W_0_dec_approx_embdr']].get_value()
            temp_d2 = self.algo.dnorm2[self.model.name2pos['W_0_dec_approx_embdr']].get_value()
            if self.state['save_gs']:
                temp_gs = self.algo.gs[self.model.name2pos['W_0_dec_approx_embdr']].get_value()
        for large in self.model.large2small_trgt:
            small = self.model.large2small_trgt[large]
            self.model.large_W_0_dec_approx_embdr[large] = temp[small]
            if not self.state['fixed_embeddings']:
                self.algo.large_W_0_dec_approx_embdr_g2[large] = temp_g2[small]
                self.algo.large_W_0_dec_approx_embdr_d2[large] = temp_d2[small]
                if self.state['save_gs']:
                    self.algo.large_W_0_dec_approx_embdr_gs[large] = temp_gs[small]

        temp = self.model.params[self.model.name2pos['W2_dec_deep_softmax']].get_value()
        if not self.state['fixed_embeddings']:
            temp_g2 = self.algo.gnorm2[self.model.name2pos['W2_dec_deep_softmax']].get_value()
            temp_d2 = self.algo.dnorm2[self.model.name2pos['W2_dec_deep_softmax']].get_value()
            if self.state['save_gs']:
                temp_gs = self.algo.gs[self.model.name2pos['W2_dec_deep_softmax']].get_value()
        for large in self.model.large2small_trgt:
            small = self.model.large2small_trgt[large]
            self.model.large_W2_dec_deep_softmax[:,large] = temp[:,small]
            if not self.state['fixed_embeddings']:
                self.algo.large_W2_dec_deep_softmax_g2[:,large] = temp_g2[:,small]
                self.algo.large_W2_dec_deep_softmax_d2[:,large] = temp_d2[:,small]
                if self.state['save_gs']:
                    self.algo.large_W2_dec_deep_softmax_gs[:,large] = temp_gs[:,small]

        temp = self.model.params[self.model.name2pos['b_dec_deep_softmax']].get_value()
        if not self.state['fixed_embeddings']:
            temp_g2 = self.algo.gnorm2[self.model.name2pos['b_dec_deep_softmax']].get_value()
            temp_d2 = self.algo.dnorm2[self.model.name2pos['b_dec_deep_softmax']].get_value()
            if self.state['save_gs']:
                temp_gs = self.algo.gs[self.model.name2pos['b_dec_deep_softmax']].get_value()
        for large in self.model.large2small_trgt:
            small = self.model.large2small_trgt[large]
            self.model.large_b_dec_deep_softmax[large] = temp[small]
            if not self.state['fixed_embeddings']:
                self.algo.large_b_dec_deep_softmax_g2[large] = temp_g2[small]
                self.algo.large_b_dec_deep_softmax_d2[large] = temp_d2[small]
                if self.state['save_gs']:
                    self.algo.large_b_dec_deep_softmax_gs[large] = temp_gs[small]

    def roll_vocab_update_dicts(self, new_large2small_src, new_large2small_trgt):
        # Update dictionaries

        logger.debug("Called roll_vocab_update_dicts()")
        self.model.large2small_src = new_large2small_src
        self.model.large2small_trgt = new_large2small_trgt

        self.model.small2large_src = invert_dict(self.model.large2small_src)
        self.model.small2large_trgt = invert_dict(self.model.large2small_trgt)

        self.model.word_indxs_src = {} # small index to word
        self.model.word_indxs = {}
        for small in self.model.small2large_src:
            large = self.model.small2large_src[small]
            self.model.word_indxs_src[small] = self.model.large2word_src[large]
        for small in self.model.small2large_trgt:
            large = self.model.small2large_trgt[small]
            self.model.word_indxs[small] = self.model.large2word_trgt[large]

    def roll_vocab_large2small(self):
        # Transfer from large to small parameters
        logger.debug("Called roll_vocab_large2small()")

        self.state['n_sym_source'] = len(self.model.small2large_src)
        logger.debug("n_sym_source=%d" % self.state['n_sym_source'])
        temp = numpy.empty((self.state['n_sym_source'], self.state['rank_n_approx']), dtype='float32')
        if not self.state['fixed_embeddings']:
            temp_g2 = numpy.empty((self.state['n_sym_source'], self.state['rank_n_approx']), dtype='float32')
            temp_d2 = numpy.empty((self.state['n_sym_source'], self.state['rank_n_approx']), dtype='float32')
            if self.state['save_gs']:
                temp_gs = numpy.empty((self.state['n_sym_source'], self.state['rank_n_approx']), dtype='float32')
        for small in self.model.small2large_src:
            large = self.model.small2large_src[small]
            temp[small] = self.model.large_W_0_enc_approx_embdr[large]
            if not self.state['fixed_embeddings']:
                temp_g2[small] = self.algo.large_W_0_enc_approx_embdr_g2[large]
                temp_d2[small] = self.algo.large_W_0_enc_approx_embdr_d2[large]
                if self.state['save_gs']:
                    temp_gs[small] = self.algo.large_W_0_enc_approx_embdr_gs[large]
        self.model.params[self.model.name2pos['W_0_enc_approx_embdr']].set_value(temp)
        if not self.state['fixed_embeddings']:
            self.algo.gnorm2[self.model.name2pos['W_0_enc_approx_embdr']].set_value(temp_g2)
            self.algo.dnorm2[self.model.name2pos['W_0_enc_approx_embdr']].set_value(temp_d2)
            if self.state['save_gs']:
                self.algo.gs[self.model.name2pos['W_0_enc_approx_embdr']].set_value(temp_gs)
        elif self.state['var_src_len']: # When embeddings are fixed, gnorm2 and dnorm2 must still have the right shape
            self.algo.gnorm2[self.model.name2pos['W_0_enc_approx_embdr']].set_value(numpy.zeros((self.state['n_sym_source'], self.state['rank_n_approx']), dtype='float32'))
            self.algo.dnorm2[self.model.name2pos['W_0_enc_approx_embdr']].set_value(numpy.zeros((self.state['n_sym_source'], self.state['rank_n_approx']), dtype='float32'))
            if self.state['save_gs']:
                self.algo.gs[self.model.name2pos['W_0_enc_approx_embdr']].set_value(numpy.zeros((self.state['n_sym_source'], self.state['rank_n_approx']), dtype='float32'))

        temp = self.model.params[self.model.name2pos['W_0_dec_approx_embdr']].get_value()
        if not self.state['fixed_embeddings']:
            temp_g2 = self.algo.gnorm2[self.model.name2pos['W_0_dec_approx_embdr']].get_value()
            temp_d2 = self.algo.dnorm2[self.model.name2pos['W_0_dec_approx_embdr']].get_value()
            if self.state['save_gs']:
                temp_gs = self.algo.gs[self.model.name2pos['W_0_dec_approx_embdr']].get_value()
        for small in self.model.small2large_trgt:
            large = self.model.small2large_trgt[small]
            temp[small] = self.model.large_W_0_dec_approx_embdr[large]
            if not self.state['fixed_embeddings']:
                temp_g2[small] = self.algo.large_W_0_dec_approx_embdr_g2[large]
                temp_d2[small] = self.algo.large_W_0_dec_approx_embdr_d2[large]
                if self.state['save_gs']:
                    temp_gs[small] = self.algo.large_W_0_dec_approx_embdr_gs[large]
        self.model.params[self.model.name2pos['W_0_dec_approx_embdr']].set_value(temp)
        if not self.state['fixed_embeddings']:
            self.algo.gnorm2[self.model.name2pos['W_0_dec_approx_embdr']].set_value(temp_g2)
            self.algo.dnorm2[self.model.name2pos['W_0_dec_approx_embdr']].set_value(temp_d2)
            if self.state['save_gs']:
                self.algo.gs[self.model.name2pos['W_0_dec_approx_embdr']].set_value(temp_gs)

        temp = self.model.params[self.model.name2pos['W2_dec_deep_softmax']].get_value()
        if not self.state['fixed_embeddings']:
            temp_g2 = self.algo.gnorm2[self.model.name2pos['W2_dec_deep_softmax']].get_value()
            temp_d2 = self.algo.dnorm2[self.model.name2pos['W2_dec_deep_softmax']].get_value()
            if self.state['save_gs']:
                temp_gs = self.algo.gs[self.model.name2pos['W2_dec_deep_softmax']].get_value()
        for small in self.model.small2large_trgt:
            large = self.model.small2large_trgt[small]
            temp[:,small] = self.model.large_W2_dec_deep_softmax[:,large]
            if not self.state['fixed_embeddings']:
                temp_g2[:,small] = self.algo.large_W2_dec_deep_softmax_g2[:,large]
                temp_d2[:,small] = self.algo.large_W2_dec_deep_softmax_d2[:,large]
                if self.state['save_gs']:
                    temp_gs[:,small] = self.algo.large_W2_dec_deep_softmax_gs[:,large]
        self.model.params[self.model.name2pos['W2_dec_deep_softmax']].set_value(temp)
        if not self.state['fixed_embeddings']:
            self.algo.gnorm2[self.model.name2pos['W2_dec_deep_softmax']].set_value(temp_g2)
            self.algo.dnorm2[self.model.name2pos['W2_dec_deep_softmax']].set_value(temp_d2)
            if self.state['save_gs']:
                self.algo.gs[self.model.name2pos['W2_dec_deep_softmax']].set_value(temp_gs)

        temp = self.model.params[self.model.name2pos['b_dec_deep_softmax']].get_value()
        if not self.state['fixed_embeddings']:
            temp_g2 = self.algo.gnorm2[self.model.name2pos['b_dec_deep_softmax']].get_value()
            temp_d2 = self.algo.dnorm2[self.model.name2pos['b_dec_deep_softmax']].get_value()
            if self.state['save_gs']:
                temp_gs = self.algo.gs[self.model.name2pos['b_dec_deep_softmax']].get_value()
        for small in self.model.small2large_trgt:
            large = self.model.small2large_trgt[small]
            temp[small] = self.model.large_b_dec_deep_softmax[large]
            if not self.state['fixed_embeddings']:
                temp_g2[small] = self.algo.large_b_dec_deep_softmax_g2[large]
                temp_d2[small] = self.algo.large_b_dec_deep_softmax_d2[large]
                if self.state['save_gs']:
                    temp_gs[small] = self.algo.large_b_dec_deep_softmax_gs[large]
        self.model.params[self.model.name2pos['b_dec_deep_softmax']].set_value(temp)
        if not self.state['fixed_embeddings']:
            self.algo.gnorm2[self.model.name2pos['b_dec_deep_softmax']].set_value(temp_g2)
            self.algo.dnorm2[self.model.name2pos['b_dec_deep_softmax']].set_value(temp_d2)
            if self.state['save_gs']:
                self.algo.gs[self.model.name2pos['b_dec_deep_softmax']].set_value(temp_gs)

    def main(self):
        assert self.reset == -1

        print_mem('start')
        self.state['gotNaN'] = 0
        start_time = time.time()
        self.start_time = start_time
        self.batch_start_time = time.time()

        self.step = int(self.timings['step'])
        self.algo.step = self.step

        if self.state['save_iter'] < 0:
            self.save_iter = 0
            self.state['save_iter'] = 0
            self.save()
            if self.channel is not None:
                self.channel.save()
        else: # Fake saving
            self.save_iter += 1
            self.state['save_iter'] = self.save_iter
        self.save_time = time.time()

        last_cost = 1.
        self.state['clr'] = self.state['lr']
        self.train_data.start(self.timings['next_offset']
                if 'next_offset' in self.timings
                else -1)

        if self.state['rolling_vocab']:
            for i in xrange(self.timings['step'] - self.timings['super_step']):
                self.train_data.next()

        if self.state['rolling_vocab']:
            # Make sure dictionary is current.
            # If training is interrupted when the vocabularies are exchanged,
            # things may get broken.
            step_modulo = self.step % self.model.total_num_batches
            if step_modulo in self.model.rolling_vocab_dict: # 0 always in.
                cur_key = step_modulo
            else:
                cur_key = 0
                for key in self.model.rolling_vocab_dict:
                    if (key < step_modulo) and (key > cur_key): # Find largest key smaller than step_modulo
                        cur_key = key
            new_large2small_src = self.model.Dx_shelve[str(cur_key)]
            new_large2small_trgt = self.model.Dy_shelve[str(cur_key)]
            self.roll_vocab_update_dicts(new_large2small_src, new_large2small_trgt)
            self.zero_or_reload = True

        while (self.step < self.state['loopIters'] and
               last_cost > .1*self.state['minerr'] and
               (time.time() - start_time)/60. < self.state['timeStop'] and
               self.state['lr'] > self.state['minlr']):
            if self.step > 0 and (time.time() - self.save_time)/60. >= self.state['saveFreq']:
                self.save()
                if self.channel is not None:
                    self.channel.save()
                self.save_time = time.time()
            st = time.time()
            try:
                if self.state['rolling_vocab']:
                    step_modulo = self.step % self.model.total_num_batches
                    if step_modulo in self.model.rolling_vocab_dict:
                        if not self.zero_or_reload:
                            self.roll_vocab_small2large() # Not necessary for 0 or when reloading a properly saved model
                            new_large2small_src = self.model.Dx_shelve[str(step_modulo)]
                            new_large2small_trgt = self.model.Dy_shelve[str(step_modulo)]
                            self.roll_vocab_update_dicts(new_large2small_src, new_large2small_trgt) # Done above for 0 or reloaded model
                        self.roll_vocab_large2small()
                        tmp_batch = self.train_data.next(peek=True)
                        if (tmp_batch['x'][:,0].tolist(), tmp_batch['y'][:,0].tolist()) == self.model.rolling_vocab_dict[step_modulo]:
                            logger.debug("Identical first sentences. OK")
                        else:
                            logger.error("Batches do not correspond.")
                    elif self.state['hookFreq'] > 0 and \
                       self.step % self.state['hookFreq'] == 0 and \
                       self.hooks:
                        [fn() for fn in self.hooks]
                    # Hook first so that the peeked batch is the same as the one used in algo
                    # Use elif not to peek twice
                rvals = self.algo()
                self.state['traincost'] = float(rvals['cost'])
                self.state['step'] = self.step
                last_cost = rvals['cost']
                for name in rvals.keys():
                    self.timings[name][self.step] = float(numpy.array(rvals[name]))
                if self.l2_params:
                    for param in self.model.params:
                        self.timings["l2_" + param.name][self.step] =\
                            numpy.mean(param.get_value() ** 2) ** 0.5

                if (numpy.isinf(rvals['cost']) or
                   numpy.isnan(rvals['cost'])) and\
                   self.state['on_nan'] == 'raise':
                    self.state['gotNaN'] = 1
                    self.save()
                    if self.channel:
                        self.channel.save()
                    print 'Got NaN while training'
                    last_cost = 0
                if self.valid_data is not None and\
                   self.step % self.state['validFreq'] == 0 and\
                   self.step > 1:
                    valcost = self.validate()
                    if valcost > self.old_cost * self.state['cost_threshold']:
                        self.patience -= 1
                        if 'lr_start' in self.state and\
                           self.state['lr_start'] == 'on_error':
                                self.state['lr_start'] = self.step
                    elif valcost < self.old_cost:
                        self.patience = self.state['patience']
                        self.old_cost = valcost

                    if self.state['divide_lr'] and \
                       self.patience < 1:
                        # Divide lr by 2
                        self.algo.lr = self.algo.lr / self.state['divide_lr']
                        bparams = dict(self.model.best_params)
                        self.patience = self.state['patience']
                        for p in self.model.params:
                            p.set_value(bparams[p.name])

                if not self.state['rolling_vocab']: # Standard use of hooks
                    if self.state['hookFreq'] > 0 and \
                       self.step % self.state['hookFreq'] == 0 and \
                       self.hooks:
                        [fn() for fn in self.hooks]
                if self.reset > 0 and self.step > 1 and \
                   self.step % self.reset == 0:
                    print 'Resetting the data iterator'
                    self.train_data.reset()

                self.step += 1
                if self.state['rolling_vocab']:
                    self.zero_or_reload = False
                    self.timings['step'] = self.step # Step now
                    if (self.step % self.model.total_num_batches) % self.state['sort_k_batches'] == 0: # Start of a super_batch.
                        logger.debug("Set super_step and next_offset")
                        # This log shoud appear just before 'logger.debug("Start of a super batch")' in 'get_homogeneous_batch_iter()'
                        self.timings['super_step'] = self.step
                        # Step at start of superbatch. super_step < step
                        self.timings['next_offset'] = self.train_data.next_offset
                        # Where to start after reload. Will need to call next() a few times
                else:
                    self.timings['step'] = self.step
                    self.timings['next_offset'] = self.train_data.next_offset

            except KeyboardInterrupt:
                break

        if self.state['rolling_vocab']:
            self.roll_vocab_small2large()
        self.state['wholetime'] = float(time.time() - start_time)
        if self.valid_data is not None:
            self.validate()
        self.save()
        if self.channel:
            self.channel.save()
        print 'Took', (time.time() - start_time)/60., 'min'
        avg_step = self.timings['time_step'][:self.step].mean()
        avg_cost2expl = self.timings['log2_p_expl'][:self.step].mean()
        print "Average step took {}".format(avg_step)
        print "That amounts to {} sentences in a day".format(1 / avg_step * 86400 * self.state['bs'])
        print "Average log2 per example is {}".format(avg_cost2expl)
