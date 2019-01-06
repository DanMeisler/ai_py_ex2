from collections import Counter


def load_table(table_file_path):
    """
    Loads table from a file.
    The resulting table form is [(instance1, class_name), (instance2, class_name), ...]
    where each instance is a dictionary of the attributes.
    :param table_file_path: the path to the file to be loaded
    :return: The table
    """
    table = []
    with open(table_file_path, "r") as table_file:
        attributes_names = table_file.readline().split()
        for data_line in table_file.readlines():
            data_line_dictionary = dict(zip(attributes_names, data_line.split()))
            _, class_name = data_line_dictionary.popitem()
            table.append((data_line_dictionary, class_name))
    return table


def calculate_hamming_distance(instance1, instance2):
    """
    Given two table instances (dictionaries), returns the hamming distance between them.
    :param instance1: first instance (See load_table function doc for more information)
    :param instance2: second instance (See load_table function doc for more information)
    :return: Hamming distance
    """
    hamming_distance = 0
    for attribute_name in instance1.keys():
        if instance1[attribute_name] != instance2[attribute_name]:
            hamming_distance += 1
    return hamming_distance


def knn_classify(table, instance, k=5):
    """
    Classify instance given a table.
    :param table: A table (See load_table function doc for more information)
    :param instance: An instance (See load_table function doc for more information)
    :param k: the k parameter in the knn algorithm
    :return: the instance class_name
    """
    knn = sorted(table, key=lambda x: calculate_hamming_distance(x[0], instance))[:k]
    knn_class_name_counter = Counter(map(lambda x: x[1], knn))
    return sorted(knn_class_name_counter, key=knn_class_name_counter.get)[-1]


def main():
    table = load_table("train.txt")
    print(table)
    print(calculate_hamming_distance(table[0][0], table[3][0]))
    print(knn_classify(table, table[25][0]))


if __name__ == "__main__":
    main()
