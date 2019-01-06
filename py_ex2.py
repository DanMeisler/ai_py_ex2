from collections import Counter

def load_table(table_file_path):
    table = []
    with open(table_file_path, "r") as table_file:
        attributes_names = table_file.readline().split()
        for data_line in table_file.readlines():
            data_line_dictionary = dict(zip(attributes_names, data_line.split()))
            _, cluster = data_line_dictionary.popitem()
            table.append((data_line_dictionary, cluster))
    return table


def calculate_hamming_distance(instance1, instance2):
    hamming_distance = 0
    for attribute_name in instance1.keys():
        if instance1[attribute_name] != instance2[attribute_name]:
            hamming_distance += 1
    return hamming_distance


def knn_classify(table, instance, k=5):
    knn = sorted(table, key=lambda x: calculate_hamming_distance(x[0], instance))[:k]
    knn_class_counter = Counter(map(lambda x: x[1], knn))
    return sorted(knn_class_counter, key=knn_class_counter.get)[-1]


def main():
    table = load_table("train.txt")
    print(table)
    print(calculate_hamming_distance(table[0][0], table[3][0]))
    print(knn_classify(table, table[3][0]))


if __name__ == "__main__":
    main()
