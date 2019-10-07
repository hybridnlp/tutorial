# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import mmap
import numpy as np
import os

from six import string_types


class Vecs(object):
  def __init__(self, vocab_filename, rows_filename, cols_filename=None):
    """Initializes the vectors from a text vocabulary and binary data."""
    with open(vocab_filename, 'r', encoding='utf_8') as lines:
      self.vocab = [line.split('\t')[0].strip() for line in lines]
      self.word_to_idx = {word: idx for idx, word in enumerate(self.vocab)}

    n = len(self.vocab)
    print('Opening vector with expected size %d from file %s' % (n, vocab_filename))
    print('vocab size %d (unique %d)' % (len(self.vocab), len(self.word_to_idx)))

    with open(rows_filename, 'rb') as rows_fh:
      rows_fh.seek(0, os.SEEK_END)
      size = rows_fh.tell()
 
      # Make sure that the file size seems reasonable.
      if size % (4 * n) != 0:
        raise IOError(
            'unexpected file size for binary vector file %s' % rows_filename)

      # Memory map the rows.
      dim = int(size / (4 * n))
      if (os.name == 'nt'):
          rows_mm = mmap.mmap(rows_fh.fileno(), 0, access=mmap.ACCESS_READ)
      else:
          rows_mm = mmap.mmap(rows_fh.fileno(), 0, prot=mmap.PROT_READ)
      rows = np.matrix(
          np.frombuffer(rows_mm, dtype=np.float32).reshape(n, dim))

      print('read rows')

      # If column vectors were specified, then open them and add them to the
      # row vectors.
      if cols_filename:
        with open(cols_filename, 'r') as cols_fh:
          if (os.name == 'nt'):
            cols_mm = mmap.mmap(cols_fh.fileno(), 0, access=mmap.ACCESS_READ)
          else:
            cols_mm = mmap.mmap(cols_fh.fileno(), 0, prot=mmap.PROT_READ)          
          cols_fh.seek(0, os.SEEK_END)
          if cols_fh.tell() != size:
            raise IOError('row and column vector files have different sizes')

          cols = np.matrix(
              np.frombuffer(cols_mm, dtype=np.float32).reshape(n, dim))

          rows += cols
          cols_mm.close()

      # Normalize so that dot products are just cosine similarity.
      self.vecs = rows / np.linalg.norm(rows, axis=1).reshape(n, 1)
      rows_mm.close()

  def similarity(self, word1, word2):
    """Computes the similarity of two tokens."""
    idx1 = self.word_to_idx.get(word1)
    idx2 = self.word_to_idx.get(word2)
    if not idx1 or not idx2:
      return None

    return float(self.vecs[idx1] * self.vecs[idx2].transpose())

  def random_word_in_vocab(self):
    """Returns a random word from the vocab"""
    return np.random.choice(self.vocab)
	
  def neighbors(self, query):
    """Returns the nearest neighbors to the query (a word or vector)."""
    if isinstance(query, string_types):
      idx = self.word_to_idx.get(query)
      if idx is None:
        print('"%s" is not in vocab, try "%s"' % (query, self.random_word_in_vocab()))
        return None

      query = self.vecs[idx]

    neighbors = self.vecs * query.transpose()

    return sorted(
      zip(self.vocab, neighbors.flat),
      key=lambda kv: kv[1], reverse=True)

  def lookup(self, word):
    """Returns the embedding for a token, or None if no embedding exists."""
    idx = self.word_to_idx.get(word)
    return None if idx is None else self.vecs[idx]
  
  def k_neighbors(self, word, k=10, result_key_suffix=''):
    """Returns the `k` nearest neighbors for the input word
    Returns a list of dicts with keys 'cosim' and 'word'
    """
    results = []
    res = self.neighbors(word)
    if not res:
      print('%s is not in the vocabulary, try e.g. %s' % (word, self.random_word_in_vocab()))
    else:
      for word, sim in res[:10]:
        results.append({
          'cosim'+result_key_suffix: sim,
          'word'+result_key_suffix: word
        })
    return results
