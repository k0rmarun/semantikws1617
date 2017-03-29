from ili_cwsd import map_with_ili
from postGIZA import post_GIZA
import json


def save():
    print("Setting up ili")
    ili_mapping, sentence_mapper = map_with_ili()
    print("ILI setup done")
    print("Sense tagging corpus")

    giza = post_GIZA()
    try:
        with open("mapping.txt", "w") as f:
            while True:
                se, sg = next(giza)
                sentence = sentence_mapper(se, sg)
                f.write(json.dumps(sentence))
                f.write("\r\n")
    except StopIteration:
        pass

    print("saved corpus completed")


if __name__ == "__main__":
    save()