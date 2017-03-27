data_dir = "/home/niels/PycharmProjects/semantikws1617/"
def get_test_data():
    with open(data_dir+"test_de") as f:
        de = list(map(eval, f.readlines()))
    with open(data_dir + "test_en") as f:
        en = list(map(eval, f.readlines()))
    return de, en