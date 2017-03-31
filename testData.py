data_dir = "/home/niels/PycharmProjects/semantikws1617/"


def get_test_data():
    def load_file(file):
        with open(data_dir + file) as f:
            data = f.readlines()
            data = filter(lambda x: x.strip(), data)
            data = [eval(x) for x in data]
            return data

    pairs = [
        ("Korpus_DE-ENGDE.txt", "Korpus_ENG-ENGDE.txt"),
        ("Korpus2_DE-ENGDE.txt", "Korpus2_ENG-ENGDE.txt"),
        ("Korpus_DE-DEENG.txt", "Korpus_ENG-DEENG.txt"),
        ("Korpus2_DE-DEENG.txt", "Korpus2_ENG-DEENG.txt"),
    ]

    de = []
    en = []
    for pair in pairs:
        de.extend(load_file(pair[0]))
        en.extend(load_file(pair[1]))

    return de, en
