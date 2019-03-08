import numpy as np
# Using high-level API Keras for creating TensorFlow NN
# https://keras.io/
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers

# fix random seed for reproducibility
seed = 42
np.random.seed(seed)

# Read the data
data = np.genfromtxt('Iris.csv', delimiter=',', usecols=(0,1,2,3))
labels = np.genfromtxt('Iris.csv', delimiter=',', usecols=4,dtype="|U")

indices = [i for i,_ in enumerate(data)]
i_setosa = indices[:50]
i_versicolor = indices[50:100]
i_virginica = indices[100:]

# Shuffle and split test and training sets
np.random.shuffle(i_setosa)
np.random.shuffle(i_versicolor)
np.random.shuffle(i_virginica)

trainSize = .3
size = int(trainSize * len(i_setosa))

train_setosa = i_setosa[:size]
test_setosa = i_setosa[size:]
train_versicolor = i_versicolor[:size]
test_versicolor = i_versicolor[size:]
train_virginica = i_virginica[:size]
test_virginica = i_virginica[size:]

train = train_setosa + train_versicolor + train_virginica
np.random.shuffle(train)
test = test_setosa + test_versicolor + test_virginica
np.random.shuffle(test)

# Create training and test set.
# One-hot encoding labels: https://hackernoon.com/what-is-one-hot-encoding-why-and-when-do-you-have-to-use-it-e3c6186d008f
X_train = []
y_train = []
for x in train:
    X_train.append(data[x])
    y = [1 if labels[x] == "Iris-setosa" else 0,
         1 if labels[x] == "Iris-versicolor" else 0,
         1 if labels[x] == "Iris-virginica" else 0]
    y_train.append(y)

X_test = []
y_test = []
for x in test:
    X_test.append(data[x])
    y = [1 if labels[x] == "Iris-setosa" else 0,
         1 if labels[x] == "Iris-versicolor" else 0,
         1 if labels[x] == "Iris-virginica" else 0]
    y_test.append(y)

X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

num_features = 4
num_labels = 3

num_hidden1 = 5
num_hidden2 = 4

# create model / Here the 'magic' happens
model = Sequential()
model.add(Dense(num_hidden1, input_dim=num_features, activation='relu'))
model.add(Dense(num_hidden2, activation='relu'))
model.add(Dense(num_labels, activation='softmax'))

# create a stochastic gradient descent optimizer
sgd = optimizers.SGD(lr=0.01)

# Compile model
model.compile(loss="mean_squared_error",optimizer=sgd, metrics=['accuracy'])

# train in batches
model.fit(X_train, y_train, epochs=500, batch_size=5)

# Evaluate
loss_and_metrics = model.evaluate(X_test, y_test, verbose=1)

prediction=model.predict(X_test)
length=len(prediction)
y_label=np.argmax(y_test,axis=1)
predict_label=np.argmax(prediction,axis=1)

accuracy=np.sum(y_label==predict_label)/length * 100
print("Accuracy of the validation:",round(accuracy,2),"%")