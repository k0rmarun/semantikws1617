from collections import defaultdict
from re import compile, search, findall, DOTALL
import sqlite3

connection = sqlite3.connect("sqlite-30.db")
cursor = connection.cursor()


# cursor.execute("PRAGMA table_info(words);")
# cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")


def get_sensenum_wn(lemma, ID):
    cursor.execute('SELECT sensenum, synsetid FROM words NATURAL JOIN senses WHERE lemma = ?', (lemma,))
    # aus irgendeinem Grund sind die wn synset ids um eine stelle länger, als die ili synset ids.
    # das wird hier sehr unschön abgefangen, die genaue ursache für dieses problem muss auf jeden fall geklärt werden,
    # könnte beim vollen durchlauf zu unabsehbaren konsequenzen führen.
    return [x[0] for x in cursor.fetchall() if str(x[1]).endswith(ID)]


def read_ili(path):
    corpus = open(path).read()
    ili = []
    lemmaRegex = compile('pwnWord="(.+?)"')
    wordnetRegex = compile('pwn30Id="ENG30-(.+?)-(.)"')
    germanetRegex = compile('lexUnitId="l(.+?)"')
    for entry in findall('<iliRecord .*?>.*?</iliRecord>', corpus, DOTALL):
        lemma = search(lemmaRegex, entry).group(1)
        wordnet = search(wordnetRegex, entry)
        germanet = search(germanetRegex, entry).group(1)
        ili.append((wordnet.group(2), lemma, wordnet.group(1), germanet))
    return ili[:10]  # später begrenzung entfernen


def create_EN_DE(ili):
    tagDict = {
        'n': defaultdict(lambda: defaultdict(set)),
        'v': defaultdict(lambda: defaultdict(set)),
        'a': defaultdict(lambda: defaultdict(set)),
        'r': defaultdict(lambda: defaultdict(set))
    }
    for entry in ili:
        for senseID in get_sensenum_wn(entry[1], entry[2]):
            tagDict[entry[0]][entry[1]][entry[3]].add(senseID)
    return {k1:
                {k2:
                     {
                         k3: tagDict[k1][k2][k3] for k3 in tagDict[k1][k2]
                         } for k2 in tagDict[k1]
                 } for k1 in tagDict
            }


if __name__ == '__main__':
    print(create_EN_DE(read_ili('interLingualIndex_DE-EN_GN110.xml')))
