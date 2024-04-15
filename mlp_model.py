import tensorflow as tf
from tensorflow.keras import models, layers, optimizers, regularizers


def create_mlp_model(input_shape, num_classes, learning_rate, steps_per_epoch):
    model = models.Sequential([
        layers.Dense(128, activation='relu', input_shape=(input_shape,)),
        layers.Dropout(0.2),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation='softmax')
    ])

    # Define an exponential decay learning rate scheduler
    lr_schedule = optimizers.schedules.ExponentialDecay(
        initial_learning_rate=learning_rate,
        decay_steps=steps_per_epoch,
        decay_rate=0.99,
        staircase=True)

    model.compile(optimizer=optimizers.Adam(learning_rate=lr_schedule),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model