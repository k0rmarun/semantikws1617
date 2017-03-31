import json


def load():
    with open("mapping.txt") as f:
        lidx = 0
        while True:
            lidx += 1
            if lidx > 5000000:
                break
            line = f.readline()
            if not line:
                return
            yield json.loads(line)


def main():
    for sentence in load():
        print(sentence)

if __name__ == "__main__":
    main()