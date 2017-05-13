import re
#import math
import multiprocessing
from multiprocessing.dummy import Pool

import numpy as np
#from textblob import TextBlob as tb
from gensim.models import Phrases
from gensim.models.phrases import Phraser
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# def tf(word, blob):
#     return float(blob.words.count(word)) / len(blob.words)
#
# def n_containing(word, bloblist):
#     return sum(1 for blob in bloblist if word in blob.words)
#
# def idf(word, bloblist):
#     return math.log(len(bloblist) / (1 + n_containing(word, bloblist)))
#
# def tfidf(word, blob, bloblist):
#     return tf(word, blob) * idf(word, bloblist)

def tokenize(texts, max_length, skip=-2):
    """ Uses numpy to quickly tokenize text and return an array
    of indices.

    This method stores a global NLP directory in memory, and takes
    up to a minute to run for the time. Later calls will have the
    tokenizer in memory.

    Parameters
    ----------
    text : list of unicode strings
        These are the input documents. There can be multiple sentences per
        item in the list.
    max_length : int
        This is the maximum number of words per document. If the document is
        shorter then this number it will be padded to this length.
    skip : int, optional
        Short documents will be padded with this variable up until max_length.

    Returns
    -------
    arr : 2D array of ints
        Has shape (len(texts), max_length). Each value represents
        the word index.
    vocab : dict
        Keys are the word index, and values are the string. The pad index gets
        mapped to None

    >>> sents = [u"Do you recall a class action lawsuit", u"hello zombo.com"]
    >>> arr, vocab = tokenize(sents, 10)
    >>> arr.shape[0]
    2
    >>> arr.shape[1]
    10
    >>> w2i = {w: i for i, w in vocab.iteritems()}
    >>> arr[0, 0] == w2i[u'do']  # First word and its index should match
    True
    >>> arr[0, 1] == w2i[u'you']
    True
    """
    texts = [re.sub("[^a-zA-Z0-9]", " ", doc.strip()).lower() for doc in texts]
    phrases = Phrases([doc.split() for doc in texts])
    bigram = Phraser(phrases)
    #
    full_vocab = list(bigram[[d.split() for d in texts]])
    data_samples = [" ".join(doc) for doc in full_vocab]
    n_features = 20000
    tfidf_vectorizer = TfidfVectorizer(max_df=0.9, min_df=5,
        max_features=n_features)
    tfidf = tfidf_vectorizer.fit_transform(data_samples)
    tfidf_feature_names = tfidf_vectorizer.get_feature_names()
    # print tfidf_vectorizer.transform([data_samples[0]])
    # uniq_vocab = np.unique(flat_vocab)
    uniq_vocab = tfidf_feature_names
    vocab = dict(zip(uniq_vocab, range(len(uniq_vocab))))
    #
    len_texts = len(texts)
    n_threads = multiprocessing.cpu_count() - 1
    len_part = int(len_texts / n_threads) + 1
    data_parts = [texts[i*len_part : (i+1)*len_part] for i in range(n_threads)]
    #
    def mapping2int(texts_part):
        data = np.zeros((len(texts_part), max_length), dtype='int32')
        data[:] = skip
        for index, doc in enumerate(texts_part):
            doc_phrases = bigram[doc.split()]
            row = [vocab[w] for w in doc_phrases if w in uniq_vocab]
            if len(row) <= 0: continue
            length = min(len(row), max_length)
            data[index, :length] = row[:length]
        return data
    #
    pool = Pool(n_threads)
    total_data = pool.map(mapping2int, data_parts)
    data = total_data[0]
    for i in range(1, len(total_data)):
        data = np.append(data, total_data[i], axis=0)
    #
    vocab = dict((v,k) for k,v in vocab.iteritems())
    #vocab[skip] = '<SKIP>'
    return data, vocab

if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True)
