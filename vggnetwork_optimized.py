import keras
from keras.datasets import fashion_mnist
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from keras.models import Sequential
from keras.utils import to_categorical
import optuna
import joblib


# load all the data
# fashion_mnist = keras.datasets.fashion_mnist
train, test = fashion_mnist.load_data()
(train_images, train_labels), (test_images, test_labels) = train, test

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


# Normalize images
train_images = train_images / 255.0
test_images = test_images / 255.0

# convert name labels into form [0,0,0,1]
train_labels_one_hot = to_categorical(train_labels)
test_labels_one_hot = to_categorical(test_labels)

# reshape them since its black/white so last channel is 1 (RGB is 3)
train_images = train_images.reshape(-1, 28, 28, 1)
test_images = test_images.reshape(-1, 28, 28, 1)


# lrelu = lambda x: keras.activations.relu(x, alpha=0.1)
def lrelu(x): return keras.activations.relu(x, alpha=0.1)


def objective(trial):
    optimizer = trial.suggest_categorical('optimizer', [
        'SGD',
        'RMSprop',
        'Adam',
        'Adadelta',
        'Adagrad',
        'Adamax',
        'Nadam',
        'Ftrl'
    ])
    dropout = trial.suggest_float('dropout', 0, 1)
    # l1 = trial.suggest_loguniform('l1', 0.000001, 0.1)

    # create model
    model = Sequential()

    model.add(Conv2D(
        32,
        (3, 3),
        strides=1,
        padding="same",
        activation=lrelu,
        input_shape=(28, 28, 1)
    ))
    model.add(keras.layers.BatchNormalization())
    model.add(Dropout(dropout))
    model.add(Conv2D(32, (3, 3), strides=1, padding="same", activation=lrelu))
    model.add(keras.layers.BatchNormalization())
    model.add(Dropout(dropout))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(64, (3, 3), strides=1, padding="same", activation=lrelu))
    model.add(keras.layers.BatchNormalization())
    model.add(Dropout(dropout))
    model.add(Conv2D(64, (3, 3), strides=1, padding="same", activation=lrelu))
    model.add(keras.layers.BatchNormalization())

    model.add(Dropout(dropout))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(128, (3, 3), strides=1, padding="same", activation=lrelu))
    model.add(keras.layers.BatchNormalization())
    model.add(Dropout(dropout))
    model.add(Conv2D(128, (3, 3), strides=1, padding="same", activation=lrelu))
    model.add(keras.layers.BatchNormalization())
    model.add(Dropout(dropout))
    model.add(MaxPooling2D((2, 2)))

    model.add(Flatten())
    model.add(Dense(10, activation="softmax"))

    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    # train model on training set
    model.fit(
        train_images,
        train_labels_one_hot,
        epochs=50,
        validation_split=0.2,
        batch_size=128,
        verbose=1
    )

    # test trained model
    test_loss, test_acc = model.evaluate(test_images, test_labels_one_hot)
    print('Test loss', test_loss)
    print('Test accuracy', test_acc)
    return test_acc


study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=200)

print(study.best_params)
joblib.dump(study, 'study.pkl')
