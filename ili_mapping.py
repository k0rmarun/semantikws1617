from sqlite3 import connect
from collections import defaultdict
from os import listdir
from re import compile, search, findall, DOTALL
#cursor.execute("PRAGMA table_info(words);")
#cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")

def wordnet_to_dict (path):
    connection = connect(path)
    cursor = connection.cursor()
    cursor.execute('SELECT synsetid, sensenum, lemma FROM words NATURAL JOIN senses')
    return {(str(synsetid)[1:], lemma):sense for synsetid, sense, lemma in cursor.fetchall()}

def germanet_to_dict (directory):
    paths = [directory + path for path in listdir(directory) if 'nomen' in path or 'verben' in path or 'adj' in path]
    germanet = {}
    for path in paths:
        words = findall('<lexUnit.+?id="(.+?)".+?>.+?<orthForm>(.+?)</orthForm>.+?</lexUnit>', open(path).read(), DOTALL)
        for word in words:
            germanet[word[0]] = word[1].lower()
    return germanet


def ili_to_list(path):
    corpus = open(path).read()
    ili = []
    tag_converter = {'n':'NOUN', 'v':'VERB', 'a':'ADJ'}
    lemmaRegex = compile('pwnWord="(.+?)"')
    wordnetRegex = compile('pwn30Id="ENG30-(.+?)-(.)"')
    germanetRegex = compile('lexUnitId="(.+?)"')
    for entry in findall('<iliRecord .*?>.*?</iliRecord>', corpus, DOTALL):
        lemma = search(lemmaRegex, entry).group(1).lower().replace('_', ' ').strip(' ')
        wordnet = search(wordnetRegex, entry)
        germanet = search(germanetRegex, entry).group(1)
        if wordnet.group(2) in tag_converter:
            ili.append((tag_converter[wordnet.group(2)], lemma, germanet, wordnet.group(1)))
    return ili


def create_EN_DE (ili, wordnet, germanet):
    tagDict = {'NOUN':defaultdict(set), 'VERB':defaultdict(set), 'ADJ':defaultdict(set)}
    for entry in ili:
        try:
            if entry[0] in tagDict and entry[1] in tagDict[entry[0]] and germanet[entry[2]] in tagDict[entry[0]][entry[1]]:
                print('error')
            tagDict[entry[0]][(entry[1], germanet[entry[2]])].add(wordnet[(entry[3], entry[1])])
        except KeyError:
            pass
    return {k1:{k2:tagDict[k1][k2].pop() for k2 in tagDict[k1] if len(tagDict[k1][k2]) == 1} for k1 in tagDict}


def create_mapping (ili, wordnet, germanet):
    ili = ili_to_list(ili)
    wordnet = wordnet_to_dict(wordnet)
    germanet = germanet_to_dict (germanet)
    return create_EN_DE(ili, wordnet, germanet)

if __name__ == '__main__':
    ili = ili_to_list('interLingualIndex_DE-EN_GN110.xml')
    wordnet = wordnet_to_dict('sqlite-30.db')
    germanet = germanet_to_dict ('/home/julian/Dropbox/Kurse/Semantik/project/germanet-11.0/GN_V110_XML/')
    mapping = create_EN_DE(ili, wordnet, germanet)
    open('mapping.txt', 'w').write(str(mapping))
    print('Einträge in ili:', len(ili))
    print('Einträge mit nur einem möglichen sense tag:', sum([len(mapping[key]) for key in mapping]))
    print('Davon unterschiedliche englische lemmata:', len({k2[0] for k1 in mapping for k2 in mapping[k1]}))
    print('Davon Nomen:', len({word[0] for word in mapping['NOUN']}))
    print('Davon Verben:', len({word[0] for word in mapping['VERB']}))
    print('Davon Adjektive:', len({word[0] for word in mapping['ADJ']}))
