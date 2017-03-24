from spacy import load
from ili_mapping import create_mapping


def map_with_ili (text, mapping):
    def inner (text, mapping):
        parser_e = load('en')
        parser_g = load('de')
        
        while True:
            try:
                sentence_e, sentence_g = next(text)
            except StopIteration:
                break
            
            sentence = []
            
            for i in range(len(sentence_e)):
                blank = sentence_e[i]
                lemma = sentence_e[i].lemma_
                sense = ''
                if sentence_e[i].pos_ in ('NOUN', 'VERB', 'ADJ'):
                    word_g = []
                    for word in sentence_g:
                        if i+1 in word:
                            word_g.append(word[0])
                    word_g = ' '.join(word_g)
                    if len(word_g) > 0:
                        sense = mapping[sentence_e[i].pos_ ].get((sentence_e[i].lemma_, parser_g(word_g)[0].lemma_), '')
                pos = sentence_e[i].pos_ if sense else ''
                sentence.append((blank, pos, lemma, sense))
            yield tupel(sentence)
    
    return tuple(inner(text, mapping))
