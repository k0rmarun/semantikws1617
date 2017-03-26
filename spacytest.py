from spacy import load
from functools import lru_cache


def map_with_ili():
    parser_e = load('en')
    parser_g = lru_cache(128)(load('de'))
    with open("mapping.txt") as f:
        mapping = eval(f.read())

    @lru_cache(20000)
    def parse_g(word):
        return parser_g(word, True, True, False)[0].lemma_

    idx = 0

    def inner(sentence_e, sentence_g):
        sentence_e = parser_e(' '.join([word[0] for word in sentence_e]), True, True, False)
        sentence = []

        for i in range(len(sentence_e)):
            blank = sentence_e[i]
            lemma = sentence_e[i].lemma_
            sense = ''
            if sentence_e[i].pos_ in ('NOUN', 'VERB', 'ADJ'):
                word_g = sentence_g if isinstance(sentence_g, str) else sentence_g[0][0]
                if len(word_g) > 0:
                    #nonlocal idx
                    #idx += 1
                    #if idx % 10000 == 0:
                        #print(parse_g.cache_info())
                    mapping_key = (sentence_e[i].lemma_, parse_g(word_g))
                    #print(mapping_key)
                    sense = mapping[sentence_e[i].pos_].get(mapping_key, '')
            pos = sentence_e[i].pos_ if sense else ''
            sentence.append((blank, pos, lemma, sense))
        return tuple(sentence)

    return inner
