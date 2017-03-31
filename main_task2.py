from ili_cwsd import map_with_ili
from testData import get_test_data
from SkipGram import SkipGram
from load_mapping import load


def iter_corpus():
    for sentence in load():
        _sentence = []
        for word in sentence:
            if (word[3] and word[3] is not -1 and word[3] is not ""): #word[1] or (word[3] and word[3] is not -1 and word[3] is not ""):
                if isinstance(word[3], list):
                    for t in word[3]:
                        _sentence.append((word[1], word[2], t))
                else:
                    _sentence.append(tuple(word[1:4]))
            else:
                _sentence.append(("UNK", word[2], -1))
        yield _sentence


def main():
    print("Calculating task 2")

    sg = SkipGram(iter_corpus, load=False)
    sg.save()

    def evaluate(sentences_from, sentences_to, poly_from, poly_to):
        count_total = {}
        count_polysemous = {}
        count_untagged_correct = {}
        count_untagged_wrong = {}
        count_tagged_correct = {}
        count_tagged_wrong = {}
        count_tagged_unexpected = {}

        tagged = []

        for sidx in range(len(sentences_from)):
            sentence_from = {}
            for word in sentences_from[sidx][0]:
                if word[-1] not in sentence_from:
                    sentence_from[word[-1]] = []
                sentence_from[word[-1]].append(word)

            sentence_to = {}
            for word in sentences_to[sidx][0]:
                if word[-1] not in sentence_to:
                    sentence_to[word[-1]] = []
                sentence_to[word[-1]].append(word)

            for widx in sentence_from.keys():
                words_from = sentence_from[widx]
                if widx not in sentence_to.keys():
                    continue
                context = tuple((w[1] if w[1] else w[0]).lower() for w in sentences_from[sidx][0])
                for word_from in words_from:
                    _type = word_from[2] if word_from[2] else "UNK"
                    if _type not in count_total:
                        count_total[_type] = 0
                    count_total[_type] += 1

                    poly = word_from[1], word_from[2]
                    poly = (poly in poly_from)
                    if poly:
                        if _type not in count_polysemous:
                            count_polysemous[_type] = 0
                        count_polysemous[_type] += 1

                        choice = sg.choose(context, (word_from[1] if word_from[1] else word_from[0]).lower())

                        if choice is not None:
                            lemma = (word_from[1] if word_from[1] else word_from[0]).lower()
                            #         lemma                      Word-Type                        Sense
                            if lemma == choice[1]:
                                if (word_from[3] == "" and choice[2] == -1) or (word_from[3] == choice[2]):#word_from[2] == choice[0] and word_from[3] == choice[2]:
                                    if _type not in count_tagged_correct:
                                        count_tagged_correct[_type] = 0
                                    count_tagged_correct[_type] += 1
                                    continue

                            if word_from[3] == "" or word_from[3] == -1:
                                if _type not in count_tagged_unexpected:
                                    count_tagged_unexpected[_type] = 0
                                count_tagged_unexpected[_type] += 1
                            else:
                                if _type not in count_tagged_wrong:
                                    count_tagged_wrong[_type] = 0
                                count_tagged_wrong[_type] += 1

                            if _type not in count_tagged_wrong:
                                count_tagged_wrong[_type] = 0
                            count_tagged_wrong[_type] += 1
                        else:
                            if word_from[3] == "" or word_from[3] == -1:
                                if _type not in count_untagged_correct:
                                    count_untagged_correct[_type] = 0
                                count_untagged_correct[_type] += 1
                            else:
                                if _type not in count_untagged_wrong:
                                    count_untagged_wrong[_type] = 0
                                count_untagged_wrong[_type] += 1

        return count_total, count_polysemous, count_untagged_correct, count_untagged_wrong, count_tagged_correct, count_tagged_wrong, count_tagged_unexpected

    mapping, _, pe, pg = map_with_ili()
    pe = list(pe)
    pg = list(pg)

    de, en = get_test_data()
    count_total, count_polysemous, count_untagged_correct, count_untagged_wrong, count_tagged_correct, count_tagged_wrong, count_tagged_unexpected = evaluate(en,de,pe,pg)



    print("Evaluating test corpus")
    total = sum(count_total.values())

    print("Total words in test corpora", total)
    for t in count_total:
        print("Count of {}: {}({}% of test corpus)".format( t, count_total[t], count_total[t]/total*100))
    print()

    for t in count_polysemous:
        print("{}: Polysemous {}({}% of test corpus)".format(t, count_polysemous[t], count_polysemous[t] / total*100))
    print()

    for t in count_untagged_correct:
        print("{}: Correct Untagged {}({}% of test corpus, {}% of polysemous)".format(t, count_untagged_correct[t], count_untagged_correct[t] / total*100, count_untagged_correct[t]/count_polysemous[t]*100))
    for t in count_untagged_wrong:
        print("{}: Incorrect Untagged {}({}% of test corpus, {}% of polysemous)".format(t, count_untagged_wrong[t], count_untagged_wrong[t] / total*100, count_untagged_wrong[t]/count_polysemous[t]*100))

    for t in count_tagged_correct:
        print("{}: Correct tagged {}({}% of test corpus, {}% of polysemous)".format(t, count_tagged_correct[t], count_tagged_correct[t] / total*100, count_tagged_correct[t]/count_polysemous[t]*100))
    for t in count_untagged_wrong:
        print("{}: Incorrect tagged {}({}% of test corpus, {}% of polysemous)".format(t, count_tagged_wrong[t], count_tagged_wrong[t] / total*100, count_tagged_wrong[t]/count_polysemous[t]*100))
    for t in count_tagged_unexpected:
        print("{}: Unexpected tagged {}({}% of test corpus, {}% of polysemous)".format(t, count_tagged_unexpected[t], count_tagged_unexpected[t] / total*100, count_tagged_unexpected[t]/count_polysemous[t]*100))


if __name__ == "__main__":
    main()
