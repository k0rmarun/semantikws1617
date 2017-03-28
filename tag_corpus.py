from postGIZA import post_GIZA
def tag_corpus(sentence_mapper):
    giza = post_GIZA()
    try:
        while True:
            se, sg = next(giza)
            sentence = sentence_mapper(se, sg)
            _sentence = []
            for word in sentence:
                if word[1] or word[3]:
                    _sentence.append(word[1:4])
                else:
                    _sentence.append((-1, word[2], -1))
            yield _sentence
    except StopIteration:
        pass