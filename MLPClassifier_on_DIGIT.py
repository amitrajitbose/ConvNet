#importing dataset
from sklearn import datasets
digit=datasets.load_digits()

x=digit.data
y=digit.target

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.5)


from sklearn.neural_network import MLPClassifier
my_classifier=MLPClassifier()

my_classifier.fit(xtrain,ytrain)

predictions=my_classifier.predict(xtest)

from sklearn.metrics import accuracy_score
print (accuracy_score(ytest,predictions))