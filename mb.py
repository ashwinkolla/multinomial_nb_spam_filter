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

data = DataFrame({'message': [], 'class': []})
data = data.append(dataFrameFromDirectory('C:/Users/Ashwin/Documents/EnronDataSets/enron1/spam', 'spam'), sort=True)
data = data.append(dataFrameFromDirectory('C:/Users/Ashwin/Documents/EnronDataSets/enron1/ham', 'ham'), sort=True)
vectorizer = CountVectorizer()
counts = vectorizer.fit_transform(data['message'].values)
classifier = MultinomialNB()
targets = data['class'].values
classifier.fit(counts, targets)


numSpamMessages = 0
spamCount = 0
testSpamPath = 'C:/Users/Ashwin/Documents/EnronDataSets/enron2/spam'
for root, dirnames, filenames in os.walk(testSpamPath):
    for filename in filenames:
        path = os.path.join(root,filename)
        numSpamMessages += 1
        lines = []
        f = io.open(path, 'r', encoding='latin1')
        for line in f:
            lines.append(line)
        f.close()
        message = '\n'.join(lines)
        examples = [message]
        example_counts = vectorizer.transform(examples)
        predictions = classifier.predict(example_counts)
        if(predictions[0] == "spam"):
            spamCount += 1
print("The actual count of spam was: " + str(spamCount))
print("The expected count of spam was: " + str(numSpamMessages))
print("The percentage of spam caught was: " + str(spamCount/numSpamMessages))

numHamMessages = 0
hamCount = 0
testHamPath = 'C:/Users/Ashwin/Documents/EnronDataSets/enron2/ham'
for root, dirnames, filenames in os.walk(testHamPath):
    for filename in filenames:
        path = os.path.join(root,filename)
        numHamMessages += 1
        lines = []
        f = io.open(path, 'r', encoding='latin1')
        for line in f:
            lines.append(line)
        f.close()
        message = '\n'.join(lines)
        examples = [message]
        example_counts = vectorizer.transform(examples)
        predictions = classifier.predict(example_counts)
        if(predictions[0] == "ham"):
            hamCount += 1
print("The actual count of ham was: " + str(hamCount))
print("The expected count of ham was: " + str(numHamMessages))
print("The percentage of ham caught was: " + str(hamCount/numHamMessages))