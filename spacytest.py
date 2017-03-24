from spacy import load
from ili_mapping import create_mapping


print('en')
parser_e = load('en')
print('de')
parser_g = load('de')

text_e = (s for s in [[('This', 1), ('is',2), ('spacy',3), ('lemmatize',4), ('testing',5)], [('Programming',1), ('booklets',2), ('are',3), ('more',4), ('better',5), ('than',6), ('others',7)]])
text_g = (s for s in [[('Dies', 1), ('ist',2), ('ein',0), ('spacy',3), ('lemmatisierungs',4), ('Test',5)], [('Programmier',1), ('Heftchen',2), ('sind',3), ('besser',5), ('als',6), ('andere',7)]])

mapping = eval(open('mapping.txt').read())
#mapping = create_mapping('interLingualIndex_DE-EN_GN110.xml', 'sqlite-30.db', '/home/julian/Dropbox/Kurse/Semantik/project/germanet-11.0/GN_V110_XML/')


corpus = []

while True:
    try:
        sentence_e = parser_e(' '.join([word[0] for word in text_e.__next__()]))
        sentence_g = text_g.__next__()
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
    corpus.append(sentence)

print(corpus)
