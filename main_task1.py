from ili_cwsd import map_with_ili
from testData import get_test_data
from tag_corpus import tag_corpus

def main():
    ili_mapping, sentence_mapper = map_with_ili()
    print("ILI setup done")
    print("Sense tagging corpus")

    tagged_by_type = {}
    total_by_type = {}
    total_words = 0
    tagged = list(tag_corpus(sentence_mapper))

    for sentence in tagged:
        for word in sentence:
            total_words += 1
            if word[1]:
                if word[1] not in total_by_type:
                    total_by_type[word[1]] = 0
                total_by_type[word[1]] += 1
            if word[3]:
                if word[1] not in tagged_by_type:
                    tagged_by_type[word[1]] = 0
                tagged_by_type[word[1]] += 1

    print("tagging corpus completed")

    print("Total projected words ", total_words)
    for type in total_by_type.keys():
        t = total_by_type.get(type, 0)
        c = tagged_by_type.get(type, 0)
        print("Got {} words as {}. Tagged {} ({}%)".format(t, type, c, c / t * 100))

    test_data = get_test_data()



if __name__ == "__main__":
    main()