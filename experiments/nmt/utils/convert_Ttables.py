# Uses T-tables made by Chris Dyer's Fast Align

import numpy as np
import marshal # I often ran out of memory with cPickle

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--fname", type=str) #T-tables
parser.add_argument("--f1name", type=str)
parser.add_argument("--f1000name", type=str)

args = parser.parse_args()

d = {}

with open(args.fname, 'r') as f:
    i = -1
    cur_source = -1
    for line in f:
        line = line.split()
        if line[0] != cur_source:
            i += 1
            if (i%1000) == 0:
                print i
            if cur_source != -1:
                d[cur_source] = tmp_dict # Set dict for previous word
            cur_source = line[0]
            tmp_dict = {}
            tmp_dict[line[1]] = pow(np.e,float(line[2]))
        else:
            tmp_dict[line[1]] = pow(np.e,float(line[2]))
d[cur_source] = tmp_dict
del tmp_dict

#####
         
e = {}
j = 0
for elt in d:
    if (j%1000) == 0:
        print j
    j += 1
    e[elt] = sorted(d[elt], key=d[elt].get)[::-1]

f1 = {}
j = 0
for elt in e:
    if (j%1000) == 0:
        print j
    j += 1
    f1[elt] = e[elt][0]
    
f1000 = {}
j = 0
for elt in e:
    if (j%1000) == 0:
        print j
    j += 1
    f1000[elt] = e[elt][:1000]

# Use marshal (2) if OutOfMemory
with open(args.f1name, 'wb') as f:
    marshal.dump(f1, f, 2)

with open(args.f1000name, 'wb') as f:
    marshal.dump(f1000, f, 2)
