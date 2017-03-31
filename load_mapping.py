import json
from cleaner import cleaner
from ili_mapping import db_to_wordnet

def load():
    _, polysemous = db_to_wordnet("sqlite-30.db")

    num_lines = 10000
    lidx = 0
    with open('mapping-inv.txt') as f:
        while True:
            lidx +=1
            if lidx > num_lines:
                break
            sentence_ = json.loads(f.readline())
            sentence = []
            for word in sentence_:
                if word[3] not in ['', -1]:
                    if (word[2], word[1]) in polysemous:
                        sentence.append(word)
                    else:
                        sentence.append([word[1], word[2], '', -1])
                else:
                    sentence.append(word)
            yield sentence

    #for sentence in cleaner(num_lines):
        #yield sentence

def main():
    for sentence in load():
        print(sentence)

if __name__ == "__main__":
    main()