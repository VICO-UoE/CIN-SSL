from __future__ import print_function


import numpy as np
from scipy import spatial
import numpy as np
from data import coref_rules


class WordEmbeddings:
    def __init__(self, word_indexer, vectors):
        self.word_indexer = word_indexer
        self.vectors = vectors

    def get_embedding_length(self):
        return len(self.vectors[0])

    def get_embedding(self, word):
        word_idx = self.word_indexer.index_of(word)
        if word_idx != -1:
            return self.vectors[word_idx]
        else:
            return self.vectors[self.word_indexer.index_of("UNK")]

    def get_embeddings(self, word_list):
        emb_list = []
        for word in word_list:
            emb = self.get_embedding(word)
            emb_list.append(emb)
        return np.array(emb_list)

    def similarity(self, w1, w2):
        return 1 - spatial.distance.cosine(
            self.get_embedding(w1), np.mean(self.get_embeddings(w2), axis=0)
        )


class WordIndexer(object):
    def __init__(self):
        self.objs_to_ints = {}
        self.ints_to_objs = {}

    def __repr__(self):
        return str([str(self.get_object(i)) for i in range(0, len(self))])

    def __str__(self):
        return self.__repr__()

    def __len__(self):
        return len(self.objs_to_ints)

    def get_object(self, index):
        if index not in self.ints_to_objs:
            return None
        else:
            return self.ints_to_objs[index]

    def contains(self, object):
        return self.index_of(object) != -1

    def index_of(self, object):
        if object not in self.objs_to_ints:
            return -1
        else:
            return self.objs_to_ints[object]

    def add_and_get_index(self, object, add=True):
        if not add:
            return self.index_of(object)
        if object not in self.objs_to_ints:
            new_idx = len(self.objs_to_ints)
            self.objs_to_ints[object] = new_idx
            self.ints_to_objs[new_idx] = object
        return self.objs_to_ints[object]


def load_vocabulary(embeddings_file: str) -> WordEmbeddings:
    f = open(embeddings_file)
    word_indexer = WordIndexer()
    vectors = []

    word_indexer.add_and_get_index("PAD")

    word_indexer.add_and_get_index("UNK")
    for line in f:
        if line.strip() != "":
            space_idx = line.find(" ")
            word = line[:space_idx]
            numbers = line[space_idx + 1 :]
            float_numbers = [float(number_str) for number_str in numbers.split()]
            vector = np.array(float_numbers)
            word_indexer.add_and_get_index(word)

            if len(vectors) == 0:
                vectors.append(np.zeros(vector.shape[0]))
                vectors.append(np.zeros(vector.shape[0]))
            vectors.append(vector)
    f.close()
    print(
        "Read in "
        + repr(len(word_indexer))
        + " vectors of size "
        + repr(vectors[0].shape[0])
    )

    return WordEmbeddings(word_indexer, np.array(vectors))

def find_sublist_single(arr, sub):
    sublen = len(sub)
    first = sub[0]
    indx = -1
    while True:
        try:

            indx = arr.index(first, indx + 1)
        except ValueError:
            break
        if sub == arr[indx : indx + sublen]:
            return indx, indx + sublen - 1
    return -1, -1


# Find position of a given sublist
# return the index of the last token
def find_sublist(arr, sub, char_indx):
    sublen = len(sub)

    first = sub[0]
    if first == "the" or first == "a":
        try:
            indx = arr.index(sub[1], char_indx)
            indx = indx - 1
        except ValueError:
            indx = 0
    else:
        try:
            indx = arr.index(first, char_indx)
        except ValueError:
            indx = 0

    if sub == arr[indx : indx + sublen]:
        char_indx = indx + sublen - 1
        return indx, indx + sublen - 1, char_indx
    else:
        return 0, 0, char_indx
    

def find_duplicate_substring_indexes(arr, sub, start_index):
    sublen = len(sub)
    first = sub[0]
 
    while True:
        start_index = arr.index(first, start_index + 1)
        if start_index == -1:
            break
        end_index = start_index + sublen - 1

    return start_index, end_index, start_index


def find_sublist_n_occ(arr, sub):
    arr = arr.numpy().tolist()
    sub = sub.numpy().tolist()
    sublen = len(sub)
    first = sub[0]

    while True:
        try:
            indxs = [i for i, k in enumerate(arr) if k == first]
        except ValueError:
            break
        sublist_occ = []
        indx = -1
        for indx in indxs:

            if sub == arr[indx : indx + sublen]:
                sublist_occ.append([indx, indx + sublen - 1])
        return sublist_occ
    return -1, -1


def get_gt_coref_matrix(phrases, clusters):
    gt_coref_matrix = np.zeros((len(phrases), len(phrases)))
    for i in range(len(clusters)):
        for j in range(len(clusters)):
            if clusters[i] == clusters[j]:
                gt_coref_matrix[i][j] = 1.0

    return gt_coref_matrix

def get_flickr_gt_coref_matrix(entities_id, phrases):

    gt_coref_matrix = np.zeros((len(phrases), len(phrases)))
    for i in range(len(entities_id)):
        for j in range(len(entities_id)):
            if entities_id[i] == entities_id[j]:
                gt_coref_matrix[i][j] = 1.0

    return gt_coref_matrix


