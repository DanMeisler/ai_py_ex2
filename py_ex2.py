from itertools import zip_longest
from collections import Counter
from functools import reduce
import operator


MODEL_TYPES = ["Decision tree", "KNN", "Naive base"]


def load_table(table_file_path):
    """
    Loads table from a file.
    The resulting table form is [(instance1, class_name), (instance2, class_name), ...]
    where each instance is a dictionary of the attributes.
    :param table_file_path: path to the file to be loaded
    :return: a table
    """
    table = []
    with open(table_file_path, "r") as table_file:
        attributes_names = table_file.readline().split()
        for data_line in table_file.readlines():
            data_line_dictionary = dict(zip(attributes_names, data_line.split()))
            _, class_name = data_line_dictionary.popitem()
            table.append((data_line_dictionary, class_name))
    return table


def get_hamming_distance(instance1, instance2):
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


def knn_classify(training_table, instance, k=5):
    """
    Classify instance given a training table.
    :param training_table: a table (See load_table function doc for more information)
    :param instance: an instance (See load_table function doc for more information)
    :param k: the k parameter in the knn algorithm
    :return: the instance class_name
    """
    knn = sorted(training_table, key=lambda x: get_hamming_distance(x[0], instance))[:k]
    knn_class_name_counter = Counter(map(lambda x: x[1], knn))
    return sorted(knn_class_name_counter, key=knn_class_name_counter.get)[-1]


def get_classes_names(table):
    return set(map(lambda x: x[1], table))


def get_possible_attribute_value_count(table, attribute_name):
    return len(set(map(lambda x: x[0][attribute_name], table)))


def get_instance_count(table, filter_function):
    return len(list(filter(filter_function, table)))


def get_smoothed_attribute_probability(table, class_name, attribute_name, attribute_value):
    k = get_possible_attribute_value_count(table, attribute_name)
    n1 = get_instance_count(table, lambda x: ((x[1] == class_name) and (x[0][attribute_name] == attribute_value)))
    n2 = get_instance_count(table, lambda x: x[1] == class_name)
    return (n1 + 1) / float(n2 + k)


def get_class_probability(table, class_name):
    return get_instance_count(table, lambda x: x[1] == class_name) / float(len(table))


def get_smoothed_probability(table, instance, class_name):
    smoothed_attributes_probabilities = \
        [get_smoothed_attribute_probability(table, class_name, x, y) for x, y in instance.items()]
    return reduce(operator.mul, smoothed_attributes_probabilities, get_class_probability(table, class_name))


def naive_base_classify(training_table, instance):
    # TODO - what in the case of equal probability
    return max(get_classes_names(training_table), key=lambda x: get_smoothed_probability(training_table, instance, x))


def create_prediction(training_table, instances, model_type):
    """
    Create prediction given a training table, instances to predict and model type to use.
    Returned prediction is in the form of [instance1's class_name, instance2's class_name, ...]
    :param training_table: a table (See load_table function doc for more information)
    :param instances: instances to be predicted (See load_table function doc for more information)
    :param model_type: a ModelTypes value
    :return: a prediction
    """
    prediction = []
    for instance in instances:
        if model_type == "Decision tree":
            pass
        elif model_type == "KNN":
            prediction.append(knn_classify(training_table, instance))
        elif model_type == "Naive base":
            prediction.append(naive_base_classify(training_table, instance))
        else:
            raise NotImplementedError()
    return prediction


def get_prediction_accuracy(prediction, real):
    """
    Calculates the accuracy of a prediction.
    :param prediction: a prediction (See create_prediction function doc for more information)
    :param real: a prediction like structure containing the real data
    :return: the accuracy the prediction is write
    """
    if len(prediction) != len(real):
        return 0.0

    return round(sum(x == y for x, y in zip(prediction, real)) / float(len(prediction)), 2)


def get_table_instances(table):
    return list(map(lambda x: x[0], table))


def output_predictions(training_table, test_table, output_file_path="output.txt"):
    """
    Creates output file given both training and test tables.
    :param training_table: a table (See load_table function doc for more information)
    :param test_table: a table (See load_table function doc for more information)
    :param output_file_path: path to a file to write the output into
    """
    header_line = "\t".join(["Num", "DT ", "KNN>", "naiveBase"])
    test_instances = get_table_instances(test_table)
    predictions = list(map(lambda x: create_prediction(training_table, test_instances, x), MODEL_TYPES))
    real = list(map(lambda x: x[1], test_table))
    predictions_accuracies = map(lambda x: str(get_prediction_accuracy(x, real)), predictions)

    with open(output_file_path, "w") as output_file:
        output_file.write(header_line + "\n")
        for instance_number, instance_prediction in enumerate(zip_longest(*predictions, fillvalue="")):
            output_file.write("%d\t" % (instance_number + 1) + "\t".join(instance_prediction) + "\n")

        output_file.write("\t" + "\t".join(predictions_accuracies) + "\n")


def main():
    training_table = load_table("train.txt")
    test_table = load_table("test.txt")
    output_predictions(training_table, test_table)


if __name__ == "__main__":
    main()
