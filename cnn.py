import tensorflow as tf
from tensorflow.keras import layers, models

NUM_CLASSES = 43



def build_cnn(dropout_rate=0.2, num_filters =64, kernel_size=3, optimizer='adam'):
    """ Creates a CNN model for our image classification task.
    """

    model = models.Sequential()
    model.add(layers.Input(shape=(64, 64, 3)))
    model.add(layers.Conv2D(num_filters, (kernel_size, kernel_size), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(num_filters * 2, (kernel_size, kernel_size), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(num_filters * 2, (kernel_size, kernel_size), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(NUM_CLASSES, activation='softmax'))

    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model



