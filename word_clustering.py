import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse import csr_matrix

from nltk.corpus import wordnet as wn

def erdos_renyi(n, p):
    A = lil_matrix((n, n), dtype = np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            if np.random.random() < p:
                A[i, j] = 1
                A[j, i] = 1

    return csr_matrix(A)

def find_related(word):
    return (word.hyponyms() + word.hypernyms()
            + word.instance_hyponyms() + word.instance_hypernyms()
            + word.member_holonyms() + word.member_meronyms()
            + word.part_holonyms() + word.part_meronyms()
            + word.substance_holonyms() + word.substance_meronyms()
            + word.entailments() + word.causes()
            + word.attributes() + word.also_sees())

def build_lexicon(word, depth):
    words = {word: 0}
    lexicon = {}
    while words:
        w, d = words.popitem()
        lexicon[w] = d
        if d < depth:
            related_words = find_related(w)
            for v in related_words:
                if words.has_key(v):
                    words[v] = min(words[v], d + 1)
                else:
                    words[v] = d + 1

    return lexicon
