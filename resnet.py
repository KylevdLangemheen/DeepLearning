from tensorflow import Tensor
from tensorflow.keras.layers import Input, Conv2D, ReLU, BatchNormalization,\
                                    Add, AveragePooling2D, Flatten, Dense
from tensorflow.keras.models import Model
from keras.datasets import fashion_mnist
from keras.applications.resnet import preprocess_input
from keras.utils import to_categorical
from keras.layers import Flatten, Dense
from keras.callbacks import EarlyStopping
from keras import Model
# import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import keras


def relu_bn(inputs: Tensor) -> Tensor:
    relu = ReLU()(inputs)
    bn = BatchNormalization()(relu)
    return bn

def residual_block(x: Tensor, downsample: bool, filters: int, kernel_size: int = 3) -> Tensor:
    y = Conv2D(kernel_size=kernel_size,
               strides= (1 if not downsample else 2),
               filters=filters,
               padding="same")(x)
    y = relu_bn(y)
    y = Conv2D(kernel_size=kernel_size,
               strides=1,
               filters=filters,
               padding="same")(y)

    if downsample:
        x = Conv2D(kernel_size=1,
                   strides=2,
                   filters=filters,
                   padding="same")(x)
    out = Add()([x, y])
    out = relu_bn(out)
    return out

def create_res_net():
    
    inputs = Input(shape=(28, 28, 1))
    num_filters = 64
    
    t = BatchNormalization()(inputs)
    t = Conv2D(kernel_size=3,
               strides=1,
               filters=num_filters,
               padding="same")(t)
    t = relu_bn(t)
    
    num_blocks_list = [2, 5, 5, 2]
    for i in range(len(num_blocks_list)):
        num_blocks = num_blocks_list[i]
        for j in range(num_blocks):
            t = residual_block(t, downsample=(j==0 and i!=0), filters=num_filters)
        num_filters *= 2
    
    t = AveragePooling2D(4)(t)
    t = Flatten()(t)
    outputs = Dense(10, activation='softmax')(t)
    
    model = Model(inputs, outputs)

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

def main():
    model = create_res_net()
    train, test = fashion_mnist.load_data()

    (train_images, train_labels), (test_images, test_labels) = train, test

    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    # Normalize images
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    # Onehot encoding for class labels
    train_labels_onehot = to_categorical(train_labels)
    test_labels_onehot = to_categorical(test_labels)

    # Reshaping for grayscale images
    train_images = train_images.reshape(-1, 28, 28, 1)
    test_images = test_images.reshape(-1, 28, 28, 1)

    callbacks = [keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=3,
    verbose=1, mode='auto'
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




if __name__ == '__main__':
    main()