CLUSTERING_ATTRIBUTE = "survived"


def load_table(table_file_path):
    table = []
    with open(table_file_path, "r") as table_file:
        attributes_names = table_file.readline().split()
        for data_line in table_file.readlines():
            data_line_dictionary = dict(zip(attributes_names, data_line.split()))
            cluster = data_line_dictionary.pop(CLUSTERING_ATTRIBUTE)
            table.append((data_line_dictionary, cluster))
    return table


def main():
    table = load_table("train.txt")
    print(table)


if __name__ == "__main__":
    main()
