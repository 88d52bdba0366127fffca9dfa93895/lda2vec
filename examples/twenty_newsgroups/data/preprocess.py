# Author: Chris Moody <chrisemoody@gmail.com>
# License: MIT

# This simple example loads the newsgroups data from sklearn
# and train an LDA-like model on it

# 0. Tokenizing done at 468.5s
# 1. corpus.update_word_count done at 1.4s
# 2. corpus.finalize done at 1.4s
# 3. corpus.to_compact done at 3.7s
# 4. corpus.filter_count done at 3.8s
# 5. corpus.compact_to_bow done at 4.4s
# 6. corpus.subsample_frequent done at 6.3s
# 7. corpus.compact_to_flat done at 6.8s
# 8. corpus.compact_word_vectors done at 19.9s

import time
import logging
import pickle

from sklearn.datasets import fetch_20newsgroups
import numpy as np

from lda2vec import preprocess, Corpus

logging.basicConfig()
start = time.time()

# Fetch data
remove = ('headers', 'footers', 'quotes')
texts = fetch_20newsgroups(subset='train', remove=remove).data
# Preprocess data
max_length = 1000 # Limit of 1k words per document
tokens, vocab = preprocess.tokenize(texts, max_length)
print '0. Tokenizing done at %.1fs' % (time.time() - start)

#del texts
corpus = Corpus()
# Make a ranked list of rare vs frequent words
corpus.update_word_count(tokens)
print '1. corpus.update_word_count done at %.1fs' % (time.time() - start)
corpus.finalize()
print '2. corpus.finalize done at %.1fs' % (time.time() - start)
# The tokenization uses spaCy indices, and so may have gaps
# between indices for words that aren't present in our dataset.
# This builds a new compact index
compact = corpus.to_compact(tokens)
print '3. corpus.to_compact done at %.1fs' % (time.time() - start)
# Remove extremely rare words
pruned = corpus.filter_count(compact, min_count=30)
print '4. corpus.filter_count done at %.1fs' % (time.time() - start)
# Convert the compactified arrays into bag of words arrays
bow = corpus.compact_to_bow(pruned)
print '5. corpus.compact_to_bow done at %.1fs' % (time.time() - start)
# Words tend to have power law frequency, so selectively
# downsample the most prevalent words
clean = corpus.subsample_frequent(pruned)
print '6. corpus.subsample_frequent done at %.1fs' % (time.time() - start)
# Now flatten a 2D array of document per row and word position
# per column to a 1D array of words. This will also remove skips
# and OoV words
doc_ids = np.arange(pruned.shape[0])
flattened, (doc_ids,) = corpus.compact_to_flat(pruned, doc_ids)
print '7. corpus.compact_to_flat done at %.1fs' % (time.time() - start)
assert flattened.min() >= 0
# Fill in the pretrained word vectors
fn_wordvc = 'GoogleNews-vectors-negative300.bin'
n_dim = 300
vectors, s, f = corpus.compact_word_vectors(vocab, filename=fn_wordvc)
print '8. corpus.compact_word_vectors done at %.1fs' % (time.time() - start)

# Save all of the preprocessed files
pickle.dump(vocab, open('vocab.pkl', 'w'))
pickle.dump(corpus, open('corpus.pkl', 'w'))
np.save("flattened", flattened)
np.save("doc_ids", doc_ids)
np.save("pruned", pruned)
np.save("bow", bow)
np.save("vectors", vectors)
