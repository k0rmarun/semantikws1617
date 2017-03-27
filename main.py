from postGIZA import post_GIZA
from spacytest import map_with_ili
from SkipGram import SkipGram
from testData import get_test_data

from memory_profiler import profile


def main():
    mapper = map_with_ili()
    print("Setup done")

    def make_corpus():
        giza = post_GIZA()
        total_words = 0
        tagged_words = 0
        try:
            while True:
                se, sg = next(giza)
                sentence = mapper(se, sg)
                _sentence = []
                for word in sentence:
                    total_words += 1
                    if word[1] or word[3]:
                        tagged_words += 1
                        _sentence.append(word[1:4])
                    else:
                        _sentence.append((-1, word[2], -1))
                yield _sentence
        except StopIteration:
            pass
        print("Got {} words. Tagged {} ({}%)".format(total_words, tagged_words, tagged_words / total_words))

    SkipGram(make_corpus)
    #test_de, test_en = get_test_data()
    #for s_idx in range(len(test_en)):
    #    sentence_en = test_en[s_idx]
    #    sentence_de = test_de[s_idx]

if __name__ == "__main__":
    main()