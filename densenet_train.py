# Imports
import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
from keras.models import Sequential
from tensorflow.python.keras.layers import Dropout, Dense, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.optimizers import RMSprop
from keras import regularizers
from datetime import datetime

# -------------------------------------------------------------------------------------------------------------------

# Preparing Data
num_classes = 93
img_height, img_width = 64, 64
batch_size = 64

# Loading data from directories
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    'ascii_dataset',
    validation_split=0.15,
    subset="training",
    seed=123,
    shuffle=True,
    image_size=(img_height, img_width),
    batch_size=batch_size,
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    'ascii_dataset',
    validation_split=0.15,
    subset="validation",
    seed=123,
    shuffle=False,
    image_size=(img_height, img_width),
    batch_size=batch_size,
)

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    'ascii_dataset',
    validation_split=0.15,
    subset="validation",
    seed=123,
    shuffle=False,
    image_size=(img_height, img_width),
    batch_size=batch_size,
)

# Printing names of folders (ascii value of characters) in dataset
class_names = train_ds.class_names

# Create a pickle file to write the class names into
with open('class_names.pkl', 'wb') as f:
    pickle.dump(class_names, f)

# Read the contents of the pickle file
with open('class_names.pkl', 'rb') as f:
    # Load the data from the file
    data = pickle.load(f)

print("Printing the contents of the pickle file: \n", data)

# Creating model
densenet_model = Sequential(name='DenseNet_Model')

# First Convolutional Layer
densenet_model.add(Conv2D(64, (7, 7), strides=(2, 2), padding='same', input_shape=(img_height, img_width, 3), activation='relu'))
densenet_model.add(MaxPooling2D((3, 3), strides=(2, 2), padding='same'))

# Dense Block 1
densenet_model.add(Conv2D(128, (1, 1), padding='same', activation='relu'))
densenet_model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
densenet_model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))

# Transition Layer 1
densenet_model.add(Conv2D(128, (1, 1), padding='same', activation='relu'))
densenet_model.add(MaxPooling2D((2, 2), strides=(2, 2), padding='same'))

# Add more Dense Blocks and Transition Layers as needed

# Global Average Pooling
densenet_model.add(GlobalAveragePooling2D())

# Fully Connected Layer
densenet_model.add(Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
densenet_model.add(Dropout(0.5))

# Output Layer
densenet_model.add(Dense(num_classes, activation='softmax', kernel_regularizer=regularizers.l2(0.01)))

# Explicitly build the model
densenet_model.build(input_shape=(batch_size, img_height, img_width, 3))

# Display the model summary
densenet_model.summary()


# Compiling model
densenet_model.compile(optimizer=RMSprop(learning_rate=0.0001), loss='sparse_categorical_crossentropy',
                       metrics=['accuracy'])

# Training model
start = datetime.now()

epochs = 600
history = densenet_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)

# Save model
densenet_model.save("models/densenet_model")

duration = datetime.now() - start
print("Training completed in time: ", duration)

# Evaluating model
fig1 = plt.gcf()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.axis(ymin=0.4, ymax=1)
plt.grid()
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.savefig('DenseNet_model_accuracy.png')
plt.show()

fig2 = plt.gcf()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.axis(ymin=0, ymax=3)
plt.grid()
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')
plt.savefig('DenseNet_model_loss.png')
plt.show()

# Testing model
test_loss, test_acc = densenet_model.evaluate(test_ds, verbose=2)
print('\nTest accuracy:', test_acc)