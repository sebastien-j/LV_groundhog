import numpy

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("cur_model", type=str)
parser.add_argument("cur_large", type=str)
parser.add_argument("final_model", type=str)
args = parser.parse_args()

#cur_model = numpy.load('search_model9.npz')
#cur_large = numpy.load('search_large9.npz')
cur_model = numpy.load(args.cur_model)
cur_large = numpy.load(args.cur_large)

restricted_list = ['W_0_enc_approx_embdr', 'W_0_dec_approx_embdr', 'W2_dec_deep_softmax', 'b_dec_deep_softmax'] 

d = {}

for elt in cur_model:
    if elt not in restricted_list:
        d[elt] = cur_model[elt]
    else:
        d[elt] = cur_large['large_'+elt]

#numpy.savez('search_final_model.npz', **d)
numpy.savez(args.final_model, **d)