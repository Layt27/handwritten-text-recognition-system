# Imports
import matplotlib.pyplot as plt
import pickle
import tensorflow as tf
from keras import regularizers
from tensorflow.python.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.models import Sequential
from keras.optimizers import RMSprop
from datetime import datetime

# -------------------------------------------------------------------------------------------------------------------

# Preparing Data
num_classes = 93              # Value is the number of folders in the dataset folder

img_height, img_width = 64, 64
batch_size = 64                     # Number of samples processed before the model is updated

# Loading data from directories
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  'ascii_dataset',
  validation_split = 0.15,
  subset = "training",
  seed = 123,
  shuffle = True,       # Shuffle order of data during training to prevent memorization and learn general patterns in data better
  image_size = (img_height, img_width),
  batch_size = batch_size,
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  'ascii_dataset',
  validation_split = 0.15,
  subset = "validation",
  seed = 123,
  shuffle = False,
  image_size = (img_height, img_width),
  batch_size = batch_size,
)

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
  'ascii_dataset',
  validation_split = 0.15,
  subset = "validation",
  seed = 123,
  shuffle = False,
  image_size = (img_height, img_width),
  batch_size = batch_size,
)

# Printing names of folders (student IDs) in dataset
class_names=train_ds.class_names

# Create a pickle file to write the class names into
with open('class_names.pkl', 'wb') as f:
  pickle.dump(class_names, f)

# Read the contents of the pickle file
with open('class_names.pkl', 'rb') as f:
  # Load the data from the file
  data = pickle.load(f)

print("Printing the contents of the pickle file: \n", data)


# Creating model
vgg19_model=Sequential(name='VGG19_Model')

# Define the VGG19 model architecture

# First Convolutional Layer
vgg19_model.add(Conv2D(64, (7, 7), strides=(2, 2), padding='same', input_shape=(img_height, img_width, 3), activation='relu'))
vgg19_model.add(MaxPooling2D((3, 3), strides=(2, 2), padding='same'))

# Dense Block 1
vgg19_model.add(Conv2D(128, (1, 1), padding='same', activation='relu'))
vgg19_model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
vgg19_model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))

# Transition Layer 1
vgg19_model.add(Conv2D(128, (1, 1), padding='same', activation='relu'))
vgg19_model.add(MaxPooling2D((2, 2), strides=(2, 2), padding='same'))

# Add more Dense Blocks and Transition Layers as needed

# Global Average Pooling
vgg19_model.add(GlobalAveragePooling2D())

# Fully Connected Layer
vgg19_model.add(Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
vgg19_model.add(Dropout(0.5))

# Output Layer
vgg19_model.add(Dense(num_classes, activation='softmax', kernel_regularizer=regularizers.l2(0.01)))

# Explicitly build the model
vgg19_model.build(input_shape=(batch_size, img_height, img_width, 3))

# Display the model summary
vgg19_model.summary()

# Compiling model
vgg19_model.compile(optimizer=RMSprop(learning_rate=0.0001),loss='sparse_categorical_crossentropy',metrics=['accuracy'])

# Training model
start = datetime.now()

epochs=600         # Number of iterations through the dataset
history = vgg19_model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)

# Save model
vgg19_model.save("models/vgg19_model")

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
test_loss, test_acc = vgg19_model.evaluate(test_ds, verbose=2)
print('\nTest accuracy:', test_acc)