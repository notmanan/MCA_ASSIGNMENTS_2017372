import pickle
from sklearn import svm
from sklearn.metrics import classification_report
import numpy as np

size = 15000
names = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
print("Getting Training Data")
Xtrain = []
Ytrain = []
for i in range(len(names[:2])):
    data = pickle.load(open(names[i] + "Spectrograms.p", "rb"))
    for j in range(len(data)):
        print(i,j)
        appender = list(np.ravel(data[j]))
        # print(len(appender))

        if(len(appender) >= size):
            Xtrain.append(appender[:size])
            Ytrain.append(i)
    
print("Getting Test Data")

print("Training")
clf = svm.SVC(kernel='rbf', gamma='scale')
clf.fit(Xtrain, Ytrain)

print("Evaluation")

Xtest = []
Ytest = []
for i in range(len(names[:2])):
    data = pickle.load(open(names[i] + "validationSpectrograms.p", "rb"))
    for j in range(len(data)):
        print(i,j)
        appender = list(np.ravel(data[j]))
        if(len(appender) >= size):
            Xtest.append(appender[:size])
            Ytest.append(i)


y_true = Ytest
y_pred = clf.predict(Xtest)

print(classification_report(y_true, y_pred))