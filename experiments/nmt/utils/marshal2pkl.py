import marshal
import cPickle

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("marshal", type=str) #T-tables
parser.add_argument("pkl", type=str)

args = parser.parse_args()

with open(args.marshal, 'rb') as f:
    d = marshal.load(f)

with open(args.pkl, 'wb') as f:
    cPickle.dump(d, f, -1)