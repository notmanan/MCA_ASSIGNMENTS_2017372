import pickle
from sklearn import svm
from sklearn.metrics import classification_report
import numpy as np

size = 1100
names = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
print("Getting Training Data")
Xtrain = []
Ytrain = []
for i in range(len(names)):
    data = pickle.load(open(names[i] + "MFCC.p", "rb"))
    for j in range(len(data)):
        # print(i,j)
        appender = list(np.ravel(data[j]))
        print(len(appender))

        if(len(appender) >= size):
            Xtrain.append(appender[:size])
            Ytrain.append(i)
    
print("Getting Test Data")
Xtest = []
Ytest = []
for i in range(len(names)):
    data = pickle.load(open(names[i] + "validationMFCC.p", "rb"))
    for j in range(len(data)):
        print(i,j)
        appender = list(np.ravel(data[j]))
        if(len(appender) >= size):
            Xtest.append(appender[:size])
            Ytest.append(i)

print("Training")
clf = svm.SVC(kernel='rbf', gamma='scale')
clf.fit(Xtrain, Ytrain)

print("Evaluation")
y_true = Ytest
y_pred = clf.predict(Xtest)

print(classification_report(y_true, y_pred))