import keras
from keras.datasets import fashion_mnist
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from keras.models import Sequential
from keras.utils import to_categorical


# load all the data
# fashion_mnist = keras.datasets.fashion_mnist
train, test = fashion_mnist.load_data()
(train_images, train_labels), (test_images, test_labels) = train, test

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


# Normalize images
train_images = train_images / 255.0
test_images = test_images / 255.0

# Show image...completely just to check
'''
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()
'''
# check shape so that we can see if necessary to change it
print(train_images.shape)

# coonvert name labels into form [0,0,0,1]
train_labels_one_hot = to_categorical(train_labels)
test_labels_one_hot = to_categorical(test_labels)


# reshape them since its black/white so last channel is 1 (RGB is 3)
train_images = train_images.reshape(-1, 28, 28, 1)
test_images = test_images.reshape(-1, 28, 28, 1)


# lrelu = lambda x: keras.activations.relu(x, alpha=0.1)
def lrelu(x): return keras.activations.relu(x, alpha=0.1)


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
model.add(Dropout(0.4))
model.add(Conv2D(32, (3, 3), strides=1, padding="same", activation=lrelu))
model.add(keras.layers.BatchNormalization())
model.add(Dropout(0.4))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(64, (3, 3), strides=1, padding="same", activation=lrelu))
model.add(keras.layers.BatchNormalization())
model.add(Dropout(0.4))
model.add(Conv2D(64, (3, 3), strides=1, padding="same", activation=lrelu))
model.add(keras.layers.BatchNormalization())

model.add(Dropout(0.4))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(128, (3, 3), strides=1, padding="same", activation=lrelu))
model.add(keras.layers.BatchNormalization())
model.add(Dropout(0.4))
model.add(Conv2D(128, (3, 3), strides=1, padding="same", activation=lrelu))
model.add(keras.layers.BatchNormalization())
model.add(Dropout(0.4))
model.add(MaxPooling2D((2, 2)))

model.add(Flatten())
model.add(Dense(10, activation="softmax"))

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

callbacks = [keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=3,
    verbose=1, mode='auto'
)]

# train model on training set
model.fit(
    train_images,
    train_labels_one_hot,
    epochs=50,
    validation_split=0.2,
    verbose=1
)


# test trained model
test_loss, test_acc = model.evaluate(test_images, test_labels_one_hot)
print('Test loss', test_loss)
print('Test accuracy', test_acc)
