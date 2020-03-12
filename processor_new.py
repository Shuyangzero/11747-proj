import csv
import os
import pandas as pd
import tokenization


class InputExample(object):
    # This is to wrap every input data to a class object
    def __init__(self, id, context, question, label):
        self.id = id
        self.text_a = context
        self.text_b = question
        self.label = label

class DataProcessor(object):
    # This is just a parent class for all types of subclass
    pass

class Sentihood_QA_M_Processor(DataProcessor):
    # This is to process input data to InputExample
    # Read tsv file
    # Method to implement: get_train, get_dev, get_test, get_labels
    # Output format:
    def get_train_examples(self, data_dir):
        data = pd.read_csv(data_dir + 'train_QA_M.tsv', sep = '\t').to_numpy()
        return self._create_examples(data, "train")

    def get_dev_examples(self, data_dir):
        data = pd.read_csv(data_dir + 'dev_QA_M.tsv', sep = '\t').to_numpy()
        return self._create_examples(data, "dev")

    def get_test_examples(self, data_dir):
        data = pd.read_csv(data_dir + 'test_QA_M.tsv', sep = '\t').to_numpy()
        return self._create_examples(data, "test")

    def get_labels(self):
        """See base class."""
        return ['None', 'Positive', 'Negative']

    def _create_examples(self, lines, set_type):
        # from numpy matrix creating list of samples
        examples = []
        for i, line in enumerate(lines):
            id = set_type + '-' + str(i)
            context = line[1]
            question = line[2]
            label = line[3]
            examples.append(InputExample(id, context, question, label))
            if i % 1000 == 0:
                print (id, context, question, label)

        return examples
