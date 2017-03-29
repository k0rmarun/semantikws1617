data_dir = "/home/niels/PycharmProjects/semantikws1617/"

def get_test_data():
    with open(data_dir + "Korpus_DE-ENGDE.txt") as f:
        de = f.readlines()
        de = filter(lambda x: x.strip(), de)
        de = [eval(x) for x in de]

    with open(data_dir + "Korpus_ENG-ENGDE.txt") as f:
        en = f.readlines()
        en = filter(lambda x: x.strip(), en)
        en = [eval(x) for x in en]
    return de, en
