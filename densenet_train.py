# Imports
import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
from keras.applications import DenseNet121
from keras.models import Sequential
from tensorflow.python.keras.layers import Flatten, Dropout, Dense
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

# Creating a pre-trained DenseNet model on the imagenet dataset
pretrained_model = DenseNet121(
    include_top=False,
    input_shape=(img_height, img_width, 3),
    weights='imagenet'
)

# Freeze the weights of the pre-trained layers
for layer in pretrained_model.layers:
    layer.trainable = False

# Define the DenseNet model architecture
densenet_model.add(pretrained_model)
densenet_model.add(Flatten())
densenet_model.add(Dropout(0.5))
densenet_model.add(Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
# densenet_model.add(Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
# densenet_model.add(Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
densenet_model.add(Dropout(0.5))
densenet_model.add(Dense(num_classes, activation='softmax', kernel_regularizer=regularizers.l2(0.01)))
    
densenet_model.summary()

# Learning rates that are too high can cause the model to oscillate around the minimum, while rates that are too low can slow down convergence
# A learning rate scheduler can adjust the learning rate during training
# lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
#     initial_learning_rate=0.01,
#     decay_steps=10000,
#     decay_rate=0.9
# )

# Compiling model
densenet_model.compile(optimizer=RMSprop(learning_rate=0.001), loss='sparse_categorical_crossentropy',
                       metrics=['accuracy'])

# Training model
start = datetime.now()

epochs = 80
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
plt.savefig('VGG19_model_accuracy.png')
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
plt.savefig('VGG19_model_loss.png')
plt.show()

# Testing model
test_loss, test_acc = densenet_model.evaluate(test_ds, verbose=2)
print('\nTest accuracy:', test_acc)