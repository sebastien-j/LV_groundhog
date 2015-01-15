# Filter bad sentences

import numpy
import cPickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("fname", type=str) # Use the tokenized text files
parser.add_argument("gname", type=str)
parser.add_argument("source_counts", type=str)
parser.add_argument("target_counts", type=str)
parser.add_argument("--good-f", type=str)
parser.add_argument("--bad-f", type=str)
parser.add_argument("--good-g", type=str)
parser.add_argument("--bad-g", type=str)
parser.add_argument("--max-length", type=int)
parser.add_argument("--max-count", type=int, default=1000)
args = parser.parse_args()

if args.good_f is None:
    good_fname = 'clean.' + args.fname
else:
    good_fname = args.good_fname
if args.bad_f is None:
    bad_fname = 'bad.' + args.fname
else:
    bad_fname = args.bad_fname
if args.good_g is None:
    good_gname = 'clean.' + args.gname
else:
    good_gname = args.good_gname
if args.bad_g is None:
    bad_gname = 'bad.' + args.gname
else:
    bad_gname = args.bad_gname
if args.max_length is None:
    args.max_length = numpy.inf

with open(args.source_counts, 'rb') as f:
    source_counts = cPickle.load(f)
with open(args.target_counts, 'rb') as f:
    target_counts = cPickle.load(f)

with open(args.fname, 'r') as f:
    with open(args.gname, 'r') as g:
        with open(good_fname, 'w') as good_f:
            with open(bad_fname, 'w') as bad_f:
                with open(good_gname, 'w') as good_g:
                    with open(bad_gname, 'w') as bad_g:
                        i = 0
                        j = 0
                        while True:
                            if (i+j) % 1000000 == 0:
                                print i+j,
                            elif (i+j) % 100000 == 0:
                                print '.',
                            fline = f.readline()
                            gline = g.readline()
                            if fline == '' or gline == '': # EOF
                                print i, j, i+j
                                break
                            fsplit = fline.split()
                            gsplit = gline.split()
                            if len(fsplit) == 0 or len(gsplit) == 0: # Empty line
                                bad_f.write(fline)
                                bad_g.write(gline)
                                i += 1
                                continue                                  
                            if len(fsplit) >= min(args.max_length + 1, 2 * len(gsplit) + 10): # Source sentence too long
                                bad_f.write(fline)
                                bad_g.write(gline)
                                i += 1
                                continue
                            if len(gsplit) >= min(args.max_length + 1, 2 * len(fsplit) + 10): # Target sentence too long
                                bad_f.write(fline)
                                bad_g.write(gline)
                                i += 1
                                continue
                            g_source = 0
                            g_target = 0
                            for word in gsplit:
                                g_source += numpy.log2(min(source_counts.get(word, 0.), args.max_count) + 1.)
                                g_target += numpy.log2(min(target_counts.get(word, 0.), args.max_count) + 1.)
                            g_source /= len(gsplit)
                            g_target /= len(gsplit)
                            g_target += 1./len(gsplit) #Be more lenient with short sentences (About twice as much for a single word sentence)
                            if (g_source > g_target): # ie the target sentence is written in the source language
                                bad_f.write(fline)
                                bad_g.write(gline)
                                i += 1
                                continue                                
                            f_source = 0
                            f_target = 0
                            for word in fsplit:
                                f_source += numpy.log2(min(source_counts.get(word, 0.), args.max_count) + 1.)
                                f_target += numpy.log2(min(target_counts.get(word, 0.), args.max_count) + 1.)
                            f_source /= len(fsplit)
                            f_target /= len(fsplit)
                            f_source += 1./len(fsplit) #Be more lenient with short sentences
                            if (f_target > f_source): # ie the source sentence is written in the target language
                                bad_f.write(fline)
                                bad_g.write(gline)
                                i += 1
                                continue
                            good_f.write(fline)
                            good_g.write(gline)
                            j += 1