from spacy import load
from ili_mapping import create_mapping


def map_with_ili (text, mapping):
    def inner (text, mapping):
        parser_e = load('en')
        parser_g = load('de')
        print('kjdkjhkfj')

        sidx = 0
        while True:
            try:
                sentence_e, sentence_g = next(text)
            except StopIteration:
                break

            sentence_e = parser_e(' '.join([word[0] for word in sentence_e]))
            sentence = []
            
            for i in range(len(sentence_e)):
                sidx += 1
                if sidx % 100000 is 0:
                    print(sidx)
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
            yield tuple(sentence)
    
    return tuple(inner(text, mapping))
