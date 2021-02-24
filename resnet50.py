from keras.datasets import fashion_mnist
from keras.applications.resnet import preprocess_input
from keras.applications.resnet50 import ResNet50
from keras.utils import to_categorical
from keras.layers import Flatten, Dense
from keras.callbacks import EarlyStopping
from keras import Model
# import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

train, test = fashion_mnist.load_data()
(train_images, train_labels), (test_images, test_labels) = train, test

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


# Resize images to 32x32 for ResNet50
def resize(x):
    im = Image.fromarray(x)
    im = im.resize((32, 32), Image.ANTIALIAS)
    return np.array(im)


train_images = np.asarray(list(map(resize, train_images)))
test_images = np.asarray(list(map(resize, test_images)))

# Normalize images
train_images = train_images / 255.0
test_images = test_images / 255.0

# Onehot encoding for class labels
train_labels_onehot = to_categorical(train_labels)
test_labels_onehot = to_categorical(test_labels)


# make images fake rgb for ResNet 50
def fake_rgb(x): return np.repeat(x[..., np.newaxis], 3, -1)


train_images = fake_rgb(train_images)
test_images = fake_rgb(test_images)

# preprocess data for resnet
train_images = preprocess_input(train_images)

# Load the model
model = ResNet50(
    include_top=False,
    weights=None,
    input_shape=train_images[0].shape,
    classes=10
    )

# Make it ready for classification
flat1 = Flatten()(model.layers[-1].output)
class1 = Dense(1024, activation='relu')(flat1)
output = Dense(10, activation='softmax')(class1)

# define new model
model = Model(inputs=model.inputs, outputs=output)

# Finish up the model and train it
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

callbacks = [EarlyStopping(
    monitor='val_loss',
    patience=3,
    verbose=1,
    mode='auto'
)]

# train model on training set
model.fit(
    train_images,
    train_labels_onehot,
    epochs=50,
    validation_split=0.2,
    verbose=1
)

# test trained model
test_loss, test_acc = model.evaluate(test_images, test_labels_onehot)
print('Test loss', test_loss)
print('Test accuracy', test_acc)
