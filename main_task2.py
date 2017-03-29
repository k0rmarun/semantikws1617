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


def get_translation_in_mapping(mapping:dict, sense):
    def inner(mapping_:dict):
        ret = []
        for elem in mapping_.keys():
            if elem[0] == sense[1]:
                if sense[2] == -1:
                    ret.append(elem)
                else:
                    if mapping_[elem] == sense[2]:
                        ret.append(elem)
        return ret
    if not sense:
        return None
    if sense[0] == "UNK":
        ret = []
        ret.extend(inner(mapping["ADJ"]))
        ret.extend(inner(mapping["NOUN"]))
        ret.extend(inner(mapping["VERB"]))
        return ret
    else:
        return inner(mapping[sense[0]])


def main():
    print("Calculating task 2")

    sg = SkipGram(iter_corpus)

    mapping, _ = map_with_ili()

    de, en = get_test_data()
    for sidx in range(len(de)):
        sentence_de = {}
        for word in de[sidx][0]:
            if word[-1] not in sentence_de:
                sentence_de[word[-1]] = []
            sentence_de[word[-1]].append(word)

        sentence_en = {}
        for word in en[sidx][0]:
            if word[-1] not in sentence_en:
                sentence_en[word[-1]] = []
            sentence_en[word[-1]].append(word)

        for widx in sentence_de.keys():
            if widx not in sentence_en:
                continue
            words_en = sentence_en[widx]
            words_de = sentence_de[widx]
            context = tuple(w[0].lower() for w in en[sidx][0])
            for word_en in words_en:
                choice = sg.choose(context, word_en.lower())
                trans = get_translation_in_mapping(mapping, choice)
                print(words_de, trans)


    words = ("case", "be", "human", )
    for word in words:
        print("{}: {}".format(word, sg.choose(["he", "said", "so", "be", "it"],word)))

if __name__ == "__main__":
    main()