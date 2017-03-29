from load_mapping import load


def main():
    tagged_by_type = {}
    total_by_type = {}
    total_words = 0

    print("Calculating statistics for task 1")

    for sentence in load():
        for word in sentence:
            total_words += 1
            if word[1]:
                if word[1] not in total_by_type:
                    total_by_type[word[1]] = 0
                total_by_type[word[1]] += 1
            if word[3] and word[3] is not -1:
                if word[1] not in tagged_by_type:
                    tagged_by_type[word[1]] = 0
                tagged_by_type[word[1]] += 1

    print("loaded corpus completed")

    print("Total projected words {}. Tagged {} ({})".format(total_words, sum(tagged_by_type.values()), sum(tagged_by_type.values())/total_words*100))
    for type in total_by_type.keys():
        t = total_by_type.get(type, 0)
        c = tagged_by_type.get(type, 0)
        print("Got {} words as {}. Tagged {} ({}%)".format(t, type, c, c / t * 100))

    #test_data = get_test_data()


if __name__ == "__main__":
    main()