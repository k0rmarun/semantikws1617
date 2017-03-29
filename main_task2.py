from ili_cwsd import map_with_ili
from testData import get_test_data
from SkipGram import SkipGram
from load_mapping import load


def iter_corpus():
    for sentence in load():
        _sentence = []
        for word in sentence:
            if word[1] or (word[3] and word[3] is not -1):
                _sentence.append(tuple(word[1:4]))
            else:
                _sentence.append(("UNK", word[2], -1))
        yield _sentence


def main():
    print("Calculating task 2")

    sg = SkipGram(iter_corpus)

    words = ("case", "be", "human", )
    for word in words:
        print("{}: {}".format(word, sg.choose(["he", "said", "so", "be", "it"],word)))

if __name__ == "__main__":
    main()