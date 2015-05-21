Neural Machine Translation
--------------------------

The folder experiments/nmt contains the implementations of RNNencdec and
RNNsearch translation models used for the paper [1,2]

####Code Structure

- encdec.py contains the actual models code
- train.py is a script to train a new model or continue training an existing one
- sample.py can be used to sample translations from the model (or to find the
  most probable translations)
- score.py is used to score sentences pairs, that is to compute log-likelihood
  of a translation to be generated from a source sentence
- state.py contains prototype states. In this project a *state* means in fact a
  full specification of the model and training process, including architectural
  choices, layer sizes, training data and vocabularies. The prototype states in
  the state.py files are base configurations from which one can start to train a
  model of his/her choice.  The *--proto* option of the train.py script can be
  used to start with a particular prototype.
- preprocess/*.py are the scripts used to preprocess a parallel corpus to obtain
  a dataset (see 'Data Preparation' below for more detail.)
- web-demo/ contains files for running a web-based demonstration of a trained
  neural machine translation model (see web-demo/README.me for more detail).

All the paths below are relative to experiments/nmt.
  
####Using training script

Simply running
```
train.py
```
would start training in the current directory. Building a model and compiling
might take some time, up to 20 minutes for us. When training starts, look for
the following files:

- search_model.npz contains all the parameters
- search_state.pkl contains the state
- search_timing.npz contains useful training statistics

The model is saved every 20 minutes.  If restarted, the training will resume
from the last saved model. 

The default prototype state used is *prototype_search_state* that corresponds to
the RNNsearch-50 model from [1].

The *--proto* options allows to choose a different prototype state, for instance
to train an RNNencdec-30 from [1] run
```
train.py --proto=prototype_encdec_state
```
To change options from the state use the positional argument *changes*. For
instance, to train RNNencdec-50 from [1] you can run
```
train.py --proto=prototype_encdec_state "prefix='encdec-50_',seqlen=50,sort_k_batches=20"
```
For explanations of the state options, see comments in the state.py file. If a
lot of changes to the prototype state are required (for instance you want to
train on another dataset), you might want put them in a file, e.g.
german-data.py, if you want to train the same model to translate to German: 
```
dict(
    source=["parallel-corpus/en-de/parallel.en.shuf.h5"],
    ...
    word_indx_trgt="parallel-corpus/en-de/vocab.de.pkl"
)
```
and run
```
train.py --proto=a_prototype_state_of_choice --state german-data.py 
```

####Using sampling script
The typical call is
```
sample.py --beam-search --state your_state.pkl your_model.npz 
```
where your_state.pkl and your_model.npz are a state and a model respectively
produced by the train.py script.  A batch mode is also supported, see the
sample.py source code.

####Data Preparation

In short, you need the following files:
- source sentence file in HDF5 format
- target sentence file in HDF5 format
- source dictionary in a pickle file (word -> id)
- target dictionary in a pickle file (word -> id)
- source inverse dictionary in a pickle file (id -> word)
- target inverse dictionary in a pickle file (id -> word)

In experiments/nmt/preprocess, we provide scripts that we use to generate these
data files from a parallel corpus saved in .txt files. 

```
python preprocess.py -d vocab.en.pkl -v 30000 -b binarized_text.en.pkl -p *en.txt.gz
```
This will create a dictionary (vocab.en.pkl) of 30,000 most frequent words and a
pickle file (binarized_text.pkl) that contains a list of numpy arrays of which
each corresponds to each line in the text files. 
```
python invert-dict.py vocab.en.pkl ivocab.en.pkl
```
This will generate an inverse dictionary (id -> word).
```
python convert-pkl2hdf5.py binarized_text.en.pkl binarized_text.en.h5
```
This will convert the generated pickle file into an HDF5 format. 
```
python shuffle-hdf5.py binarized_text.en.h5 binarized_text.fr.h5 binarized_text.en.shuf.h5 binarized_text.fr.shuf.h5
```
Since it can be very expensive to shuffle the dataset each time you train a
model, we shuffle dataset in advance. However, note that we do keep the original
files for debugging purpose.

####Tests

Run
```
GHOG=/path/to/groundhog test/test.bash
```
to test sentence pairs scoring and translation generation. The script will start with creating 
a workspace directory and download test models there. You can keep using the same test workspace
and data.

####Known Issues

- float32 is hardcoded in many places, which effectively means that you can only 
  use the code with floatX=float32 in your .theanorc or THEANO_FLAGS
- In order to sample from the RNNsearch model you have to set the theano option on_unused_input to 'warn' 
  value via either .theanorc or THEANO_FLAGS

####References

Dzmitry Bahdanau, Kyunghyun Cho and Yoshua Bengio. 
Neural Machine Translation by Jointly Learning to Align and Translate
http://arxiv.org/abs/1409.0473

Kyunghyun Cho, Bart van Merrienboer, Caglar Gulcehre, Fethi Bougares, Holger Schwenk and Yoshua Bengio.
Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation.
EMNLP 2014. http://arxiv.org/abs/1406.1078

Large Vocabulary Neural Machine Translation
-------------------------------------------

####Training

Preprocess data as before. With a prototype/state such as 
`prototype_lv_state`, `rolling_dicts.py` will create 3 files that will 
tell the model when to update the current vocabularies. Afterwards, 
simply run `train.py` (with the same state).

```
python rolling_dicts.py --proto prototype_lv_state --state ${state}
python train.py --proto prototype_lv_state --state ${state}
```

####Using SMT word alignments (optional but highly recommended)

Although these steps are optional, they make decoding better and faster. 
They may be done while training the model, and therefore add very little 
if no overhead. However, you may need quite a lot of RAM. The following 
instructions will only consider the *fast_align* package, but either 
*giza++* or the *Berkeley aligner* could likely be used, although some 
commands may need to be altered. 

Put your data in a format suitable for *fast_align* (one sentence per 
line, source (left) and target (right) separated by triple pipes). You
may use `utils/format_fast_align.py` for this. Then run

```
[...]/fast_align -i ${data} -d -o -v -c ${ttables} > ${align}

python utils/convert_Ttables.py --fname ${ttables} \
                                --f1name ${ttables}.f1.marshal \
                                --f1000name ${ttables}.f1000.marshal
                                
python utils/marshal2pkl.py ${ttables}.f1.marshal ${mapping}.pkl
python utils/marshal2pkl.py ${ttables}.f1000.marshal ${ttables}.f1000.pkl

python utils/binarize_top_unigram.py --top-unigram ${ttables}.f1000.pkl \
    --src-w2i ${src_word2index}.pkl --trg-w2i ${trg_word2index}.pkl \
    --vocab-size ${vocab_size} --output ${topn}.pkl
```

This will allow the probability tables to be used during decoding, for 
either creating candidate lists or for unknown word replacement. It 
should be possible to use symmetrized alignment models, although in this 
case you may need to write custom scripts to get the `${mapping}.pkl` 
and `${topn}.pkl` files in the correct format. 

####Preparing models for evaluation

Before models may be evaluated, some mandatory steps must be taken. 
Convert the state generated by the model first. This needs to be done 
only once for all models using the same architecture and vocabularies, 
even if they are from different training runs. 

```
python utils/convert_state_local.py ${state}.pkl ${state}.final.pkl
```

To transfer the large vocabulary embeddings into the model file, 
you may run

```
python utils/build_final_model.py ${model}.npz ${large}.npz ${model}.final.npz
```

`utils/build_final_model.py` may be replaced by 
`utils/build_final_model_s2l.py`, which takes two additional arguments. 
The latter uses the current parameters while the first takes slightly 
outdated word embeddings (from the last time they were transferred from 
GPU to host). I don't know which one will work better for you, so you 
may want to try both. 

####Vocabulary filtering (optional)

As large vocabulary NMT models take up much memory, which may sometimes 
be an issue, especially if you want to cram many models on a GPU for 
fast decoding, you may filter the model according to the source side of 
the dev/test set.

For a given value of `K` and `Kp` (see the paper), run the following 
command once (not for each model). You may tune `K` and `Kp`  on the 
development set. If you are time-constrained, good default values are 
`K=state['n_sym_target']/1000` and `Kp=10`.

```
THEANO_FLAGS='device=cpu' python utils/filter_vocab.py --state ${state}.final.pkl \
                                 --source-file ${src} --num-common ${K}000 --num-ttables ${Kp} \
                                 --topn-file ${topn}.pkl --save-vocab-dir ${vocab_dir} \
                                 --ext ${ext} --new-state --new-topn
```

This will create new vocabulary files, filtered translation tables and 
states, with `${ext}` added to the filenames. You may then filter each 
model with 

```
    THEANO_FLAGS='device=cpu' python utils/filter_model.py \
        --old-state ${state}.final.pkl --new-state ${state}.final.${ext}.pkl \
        --old-model ${model}.final.npz --new-model ${model}.final.${ext}.npz
```

####Decoding

If you didn't use external word alignments, you may use `sample.py` as 
before. However, it will take a lot of time and may give sub-optimal 
results. Otherwise, use `fast_samply.py`, either with filtered models or 
not (if not, just remove ".${ext}" where appropriate). 

```
THEANO_FLAGS='device=gpu, on_unused_input=warn' fast_sample.py \
    --beam-search --beam-size ${beam} --normalize --less-transfer \
    --num-common ${K}000 --num-ttables ${Kp} --topn-file ${topn}.${ext}.pkl \
    --state ${state}.final.${ext}.pkl --models ${model}.final.${ext}.npz \
    --source ${src} --trans ${trans}
```

`--less-transfer`, which was only mentioned in passing in the paper, 
should allow for faster decoding, although the obtained translations 
will be slightly different. The `--normalize` flag generates longer 
output, which we found to often be beneficial (on the dev. sets) in the 
experiments we ran. If you wish to, you may tune `--normalize-p [float]` 
and/or `--wp [float]` on your development set, although we did not use 
these options in our paper. 

Moreover, if you want to use an ensemble, you simply have to put the 
paths to all models after `--models`. Currently, only models that share 
an identical state (ie same architecture and vocabularies) are 
supported, even though more diverse models could lead to greater 
improvements. 

####Replacing unknown words

If you want to replace unknown words, run

```
THEANO_FLAGS='device=gpu, on_unused_input=warn' replace_UNK.py --heuristic ${heuristic} \
    --num-common ${K}000 --num-ttables ${Kp} --topn-file ${topn}.${ext}.pkl \
    --state ${state}.final.${ext}.pkl --models ${model}.final.${ext}.npz \
    --mapping ${mapping}.pkl --source ${src} --trans ${trans} --new-trans ${new_trans}
```

There are currently three supported heuristics. Heuristic 0 copies 
unknown words, while heuristic 1 tries to use a mapping to replace 
unknown words and falls back to copying if no match is found. Finally, 
heuristic 2 tries to use the dictionary only for source words that start 
with a lowercase letter, and otherwise copies. If you remove 
`--num-common`, `--num-ttables` and `--topn-file` from your command, 
this script should also work with RNN-search. 

####Reloading a model

As before, is training is interrupted, you may reload a model and 
restart training. However, you need to take some quick additional steps. 
In your state file, you should change `state['save_iter']` to the id of 
the last saved model. Moreover, if you used `state['var_src_len']=True`, 
you need to set `state['n_sym_source']` correctly. To do so, you may 
load you model `model=dict(numpy.load(model_name))` and then use 
`len(model['W_0_enc_approx_embdr'])`. This quirk should probably be 
fixed... 

####Fixing the embeddings

Near the end of training, you can try fixing the embeddings of the 
models, which often seemed to help us get better BLEU scores once they 
had stopped improving on the development set. I usually choose a new 
prefix and set the iteration id to 0. Copy (or symlink) the `timings`, 
`algo` and either one of the `model` or `large` files
(eg `cp ${old_prefix}timings${id}.npz ${new_prefix}timings0.npz`).
Then run

```
python utils/l2s_or_s2l.py ${old_prefix}model${id}.npz ${old_prefix}large${id}.npz \
                           ${new_prefix}[model/large]0.npz \
                           ${old_prefix}timing${id}.npz ${old_prefix}state.pkl [--s2l]
```

Without the `--s2l` flag, you should have previously copied the *large* 
file, and the previous script will create a new *model* file. With it, 
you should do the opposite. This step is needed to sync the embeddings 
in the *model* and *large* files. 

To restart training, modify your prototype/state by changing the prefix, 
setting `save_iter` to `0` and `fixed_embeddings` to `True`. You may 
then run `train.py` normally. 

With this option, you won't save any *large* files anymore, and as such 
you may want to save more frequently. If you need to restart training, 
you should update the id of the *large* file to match the others. 

####References
```
Sébastien Jean, Kyunghyun Cho, Roland Memisevic and Yoshua Bengio. 
On Using Very Large Target Vocabulary for Neural Machine Translation
ACL 2015. http://www.arxiv.org/abs/1412.2007
```

If you use `replace_UNK.py`, you may also want to cite
```
Minh-Thang Luong, Ilya Sutskever, Quoc V. Le, Oriol Vinyals, Wojciech Zaremba
Addressing the Rare Word Problem in Neural Machine Translation
ACL 2015. http://www.arxiv.org/abs/1410.8206
```