from ili_cwsd import map_with_ili
from testData import get_test_data
from SkipGram import SkipGram
from load_mapping import load


def iter_corpus():
    for sentence in load():
        _sentence = []
        for word in sentence:
            if word[1] or (word[3] and word[3] is not -1):
                if isinstance(word[3], list):
                    for t in word[3]:
                        _sentence.append((word[1], word[2], t))
                else:
                    _sentence.append(tuple(word[1:4]))
            else:
                _sentence.append(("UNK", word[2], -1))
        yield _sentence


def get_translation_in_mapping(mapping: dict, sense, from_="de"):
    idx = 1 if from_ == "en" else 0

    def inner(mapping_: dict):
        ret = []
        for elem in mapping_.keys():
            if elem[idx] == sense[1]:
                if sense[2] == -1 or sense[2] == "":
                    ret.append((elem[0], elem[1], mapping_[elem]))
                else:
                    if list(mapping_[elem])[0] == sense[2]:
                        ret.append((elem[0], elem[1], list(mapping_[elem])[0]))
        return ret

    if not sense:
        return None
    if sense[0] == "UNK":
        ret = []
        ret.extend(inner(mapping["ADJ"]))
        ret.extend(inner(mapping["NOUN"]))
        ret.extend(inner(mapping["VERB"]))
        return "UNK", ret
    else:
        return sense[0], inner(mapping[sense[0]])


def main():
    print("Calculating task 2")

    sg = SkipGram(iter_corpus, load=True)
    # sg.save()

    mapping, _ = map_with_ili()

    de, en = get_test_data()
    count_unknown = 0
    count_not_ili = {}
    count_polysemous = {}
    count_disambiguated = {}
    count_correct = {}
    count_total = {}

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
            context = tuple((w[1] if w[1] else w[0]).lower() for w in en[sidx][0])
            for word_en in words_en:
                choice = sg.choose(context, (word_en[1] if word_en[1] else word_en[0]).lower())
                trans = get_translation_in_mapping(mapping, choice)

                if trans is None:
                    count_unknown += 1
                    if "UNK" not in count_total:
                        count_total["UNK"] = 0
                    count_total["UNK"] += 1
                    continue
                if trans[0] not in count_total:
                    count_total[trans[0]] = 0
                print("---", trans)
                count_total[trans[0]] += 1
                if len(trans[1]) == 0:
                    if trans[0] not in count_not_ili:
                        count_not_ili[trans[0]] = 0
                    count_not_ili[trans[0]] += 1
                elif len(trans[1]) == 1:
                    if trans[0] not in count_disambiguated:
                        count_disambiguated[trans[0]] = 0
                    count_disambiguated[trans[0]] += 1
                    w_de = (words_de[0][1] if words_de[0][1] else words_de[0][0]).lower()
                    if w_de == trans[1][0][1]:
                        if trans[0] not in count_correct:
                            count_correct[trans[0]] = 0
                        count_correct[trans[0]] += 1
                        print("Got correct:")
                        print("En:", words_en)
                        print("De:", words_de)
                        print("Disambiguated:", trans)
                else:
                    if trans[0] not in count_polysemous:
                        count_polysemous[trans[0]] = 0
                    count_polysemous[trans[0]] += 1

    print("Evaluating test corpus")
    total = sum(count_total.values())
    print("Total ", total)
    for t in count_total:
        print("{}: count {}({}% of test corpus)".format( t, count_total[t], count_total[t]/total*100))
    print()

    print("Unknown {} ({}% of test corpus)".format(count_unknown, count_unknown/total*100))
    print()

    for t in count_not_ili:
        print("{}: Not in ili {}({}% of test corpus)".format(t, count_not_ili[t], count_not_ili[t]/total*100))
    print()

    for t in count_polysemous:
        print("{}: Polysemous {}({}% of test corpus)".format(t, count_polysemous[t], count_polysemous[t] / total*100))
    print()

    for t in count_disambiguated:
        print("{}: Disambiguated {}({}% of test corpus)".format(t, count_disambiguated[t], count_disambiguated[t] / total*100))
    print()

    for t in count_correct:
        print("{}: Correct {}({}% of test corpus)".format(t, count_correct[t], count_correct[t] / total*100))
    if len(count_correct.values()) == 0:
        print("No correct disambiguated words")
    print()


if __name__ == "__main__":
    main()
