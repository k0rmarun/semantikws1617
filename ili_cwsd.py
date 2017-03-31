from functools import lru_cache
import treetaggerwrapper
from ili_mapping import create_mapping


def map_with_ili():
    ili_path = "interLingualIndex_DE-EN_GN110.xml"
    wordnet_path = "sqlite-30.db"
    germanet_path = "germanet-11.0/GN_V110_XML/"

    mapping, pe, pg = create_mapping(ili_path, wordnet_path, germanet_path)
    parser_e = treetaggerwrapper.TreeTagger(TAGLANG="en", TAGDIR="/home/niels/tree-tagger", TAGPARFILE="/home/niels/tree-tagger/english.par")
    parser_g = treetaggerwrapper.TreeTagger(TAGLANG="de", TAGDIR="/home/niels/tree-tagger", TAGPARFILE="/home/niels/tree-tagger/german.par")

    tagger_args = {"notagdns":True, "nosgmlsplit":True, "notagemail":True, "notagip":True, "notagurl":True}

    @lru_cache(20000)
    def parse_g(word):
        return [x.split() for x in parser_g.tag_text(word, **tagger_args)][0][2].lower()

    def parse_e(word):
        return [x.split() for x in parser_e.tag_text(word, **tagger_args)]

    idx = 0

    def inner(sentence_e, sentence_g):
        sentence_e = parse_e(' '.join([word[0] for word in sentence_e]))
        sentence = []
        tags = {"NN": "NOUN", "VB": "VERB", "JJ": "ADJ"}

        for i in range(len(sentence_e)):
            blank = sentence_e[i]
            lemma = blank[2].lower()
            sense = ''
            tag = ""

            if lemma in ".,-;:_#'+*~!?()[]{}":
                continue

            if blank[1][:2] in tags.keys():
                tag = tags[blank[1][:2]]
                word_g = []
                for word in sentence_g:
                    if i + 1 in word:
                        word_g.append(word[0])
                word_g = ' '.join(word_g)
                if len(word_g) > 0:
                    sense_key = (lemma, parse_g(word_g))
                    sense = mapping[tag].get(sense_key, -1)

            sentence.append((blank[0], tag, lemma, sense))
        return tuple(sentence)

    return mapping, inner, pe, pg
