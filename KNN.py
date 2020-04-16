import pandas as pd
import sklearn
from sklearn import preprocessing, linear_model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import shuffle
import numpy as np

df = pd.read_csv('car.data')
#print(df.head())

le = preprocessing.LabelEncoder()
buying = le.fit_transform(list(df['buying']))
maint = le.fit_transform(list(df['maint']))
door = le.fit_transform(list(df['door']))
persons = le.fit_transform(list(df['persons']))
safety = le.fit_transform(list(df['safety']))
lug_boot = le.fit_transform(list(df['lug_boot']))
cls = le.fit_transform(list(df['class']))

predict = 'class'

X = list(zip(buying,maint,door,persons,safety,lug_boot))
y = list(cls)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

'''
#find the best possible value of k
best1 = 0
best = 0
needed = 1
k_value = 1
for k in range(100):
    for j in range(30):
        clf = KNeighborsClassifier(n_neighbors=needed)
        clf.fit(x_train, y_train)
        acc = clf.score(x_test, y_test)
        if acc>best:
            best1 = acc
    if best1 > best:
        best = best1
        k_value = needed
    needed = needed + 2

print(k_value)
'''
clf = KNeighborsClassifier(n_neighbors=9)
clf.fit(x_train, y_train)
acc = clf.score(x_test, y_test)
print(acc)
print(y_test)

predictions = clf.predict(x_test)

#for data viewing use this
'''names= ['unacc', 'acc', 'good', 'vgood']
for i in range(len(predictions)):
    def reply_name(prediction):
        if prediction == 0:
            return (names[0])
        elif prediction == 1:
            return (names[1])
        elif prediction == 2:
            return (names[2])
        elif prediction == 3:
            return (names[3])
    print("Real value : {} and Predicted value : {}".format(reply_name(y_test[i]),reply_name(predictions[i])))
'''
#for ecludian distances
for x in range(len(predictions)):
    n= clf.kneighbors([x_test[x]], 9, True)
    print(n)