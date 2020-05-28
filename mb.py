import os
import io 
import numpy
from pandas import DataFrame
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

def readFiles(path):
    for root, dirnames, filenames in os.walk(path):
        for filename in filenames:
            path = os.path.join(root,filename)

            # inBody = False
            lines = []
            f = io.open(path, 'r', encoding='latin1')
            for line in f:
                # if inBody:
                lines.append(line)
                # elif line == "\n":
                #     inBody = True
            f.close()
            message = '\n'.join(lines)
            yield path, message

def dataFrameFromDirectory(path, classification):
    rows = []
    index = []
    for filename, message in readFiles(path):
        rows.append({'message': message, 'class': classification})
        index.append(filename)

    return DataFrame(rows, index=index)


def build_model(spam_dict):
    data = DataFrame({'message': [], 'class': []})
    for classif in spam_dict:
        for train_data_path in spam_dict[classif]:
            data = data.append(dataFrameFromDirectory(train_data_path, classif), sort=True)
    vectorizer = CountVectorizer()
    counts = vectorizer.fit_transform(data['message'].values)
    classifier = MultinomialNB()
    targets = data['class'].values
    classifier.fit(counts, targets)
    return vectorizer, classifier


def test_model(spam_dict, spam_test, test_path):
    vectorizer, classifier = build_model(spam_dict)
    num_messages = 0
    correct_count = 0
    for root, dirnames, filenames in os.walk(test_path):
        for filename in filenames:
            path = os.path.join(root,filename)
            num_messages += 1
            lines = []
            f = io.open(path, 'r', encoding='latin1')
            for line in f:
                lines.append(line)
            f.close()
            message = '\n'.join(lines)
            examples = [message]
            example_counts = vectorizer.transform(examples)
            predictions = classifier.predict(example_counts)
            if(predictions[0] == spam_test):
                correct_count += 1

    print(f"\n{spam_test} Count:")            
    print(f"The actual count of {spam_test} was: " + str(correct_count))
    print(f"The expected count of {spam_test} was: " + str(num_messages))
    print(f"The percentage of {spam_test} caught was: " + str(correct_count/num_messages))

