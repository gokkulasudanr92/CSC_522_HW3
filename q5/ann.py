from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy
import matplotlib.pyplot as plt

numpy.random.seed(7)

EPOCHS = 5
BATCH_SIZE = 1
hidden_neurons = [2, 4, 6, 8, 10]
training_scores = []
validate_scores = []

# Define a model
def defineModel(hidden_neurons):
    model = Sequential()
    # First input layer
    model.add(Dense(1, input_dim=64))
    # Second hidden layer with 2 neurons with relu activation
    model.add(Dense(hidden_neurons, activation="relu"))
    # One output layer with sigmoid activation
    model.add(Dense(1, activation="sigmoid"))
    model.compile(optimizer="rmsprop", loss="mse", metrics=["accuracy"])
    return model

# Training Dataset
X_train = numpy.loadtxt("./hw3q5/X_train.csv", delimiter=",")
Y_train = numpy.loadtxt("./hw3q5/Y_train.csv", delimiter=",")

# Validation Dataset
X = numpy.loadtxt("./hw3q5/X_val.csv", delimiter=",")
Y = numpy.loadtxt("./hw3q5/Y_val.csv", delimiter=",")


for hidden in hidden_neurons:
    # Define the model
    model = defineModel(hidden)
    # Fit the model
    model.fit(X_train, Y_train, epochs=EPOCHS, batch_size=BATCH_SIZE)
    
    #Evaluate
    training_score = model.evaluate(X_train, Y_train)
    validate_score = model.evaluate(X, Y)
    training_scores.append(training_score[1] * 100)
    validate_scores.append(validate_score[1] * 100)

    del(model)

plt.title("Accuracy(%) vs. Hidden Neurons")
plt.xlabel("Hidden Neurons")
plt.ylabel("Accuracy(%)")
plt.plot(hidden_neurons, training_scores, marker="o", color="red", label="Training Accuracy")
plt.plot(hidden_neurons, validate_scores, marker="o", color="blue", label="Validation Accuracy")
plt.legend()
plt.draw()

max_index = -1
max_acc = -1
for i in range(len(hidden_neurons)):
    if (max_acc < validate_scores[i]):
        max_acc = validate_scores[i]
        max_index = i

X_test = numpy.loadtxt("./hw3q5/X_test.csv", delimiter=",")
Y_test = numpy.loadtxt("./hw3q5/Y_test.csv", delimiter=",")

m = defineModel(hidden_neurons[max_index])
m.fit(X_train, Y_train, epochs=EPOCHS, batch_size=BATCH_SIZE)
test_score = m.evaluate(X_test, Y_test)
print("\n\n%s: %.2f%%" % (m.metrics, test_score[1] * 100))
plt.show()
