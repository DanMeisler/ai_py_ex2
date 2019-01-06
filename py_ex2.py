
def load_table(table_file_path):
    table = []
    with open(table_file_path, "r") as table_file:
        attributes_names = table_file.readline().split()
        for data_line in table_file.readlines():
            data_line_dictionary = dict(zip(attributes_names, data_line.split()))
            _, cluster = data_line_dictionary.popitem()
            table.append((data_line_dictionary, cluster))
    return table


def calculate_hamming_distance(dictionary1, dictionary2):
    hamming_distance = 0
    for attribute_name in dictionary1.keys():
        if dictionary1[attribute_name] != dictionary2[attribute_name]:
            hamming_distance += 1
    return hamming_distance


def main():
    table = load_table("train.txt")
    print(table)
    print(calculate_hamming_distance(table[0][0], table[3][0]))


if __name__ == "__main__":
    main()
