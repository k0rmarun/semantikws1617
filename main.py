from postGIZA import post_GIZA
from spacytest import map_with_ili

with open("mapping.txt") as f:
    mapping = eval(f.read())
output = map_with_ili(post_GIZA(), mapping)
total_words = 0
tagged_words = 0
for sentence in output:
    for word in sentence:
        total_words += 1
        if word[1] or word[3]:
            tagged_words += 1
            print(word)

print("Got {} words. Tagged {} ({}%)".format(total_words, tagged_words, tagged_words/total_words))