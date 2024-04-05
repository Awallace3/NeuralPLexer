from torchdrug import datasets

def pdbbind_load():
    datasets.PDBBind("pdbbind_v2019")
    return


def main():
    pdbbind_load()
    return


if __name__ == "__main__":
    main()
