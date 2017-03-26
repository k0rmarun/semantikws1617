from postGIZA import post_GIZA
from spacytest import map_with_ili
from SkipGram import SkipGram

from spacy import load

parser_e = load('en')
parser_g = load('de')
with open("mapping.txt") as f:
    mapping = eval(f.read())

print("DONE")

def make_corpus():
    output = map_with_ili(post_GIZA(), mapping, parser_e, parser_g)
    total_words = 0
    tagged_words = 0
    for sentence in output:
        _sentence = []
        for word in sentence:
            total_words += 1
            if word[1] or word[3]:
                tagged_words += 1
                _sentence.append(word[1:4])
            else:
                _sentence.append((-1,word[2],-1))
        yield _sentence

    print("Got {} words. Tagged {} ({}%)".format(total_words, tagged_words, tagged_words/total_words))

SkipGram(make_corpus)