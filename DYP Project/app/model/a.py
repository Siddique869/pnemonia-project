import os
import numpy as np
import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import image_dataset_from_directory

# Define paths to the train and test datasets
train_path = 'xray_dataset_covid19/train/'
test_path = 'xray_dataset_covid19/test/'

# Load the training and testing datasets
train_ds = image_dataset_from_directory(train_path, batch_size=148)
test_ds = image_dataset_from_directory(test_path, batch_size=40)

# Convert the datasets to numpy arrays and normalize pixel values
train_x, train_y = next(train_ds.as_numpy_iterator())
test_x, test_y = next(test_ds.as_numpy_iterator())
train_x, test_x = train_x / 255.0, test_x / 255.0

# Load ResNet50 model without the top layer and freeze its layers
ResNet = ResNet50(include_top=False, input_shape=train_x[0].shape)
for layer in ResNet.layers:
    layer.trainable = False

# Add custom layers (Flatten + Dense) for binary classification
x = Flatten()(ResNet.output)
output = Dense(1, activation='sigmoid')(x)
model = Model(ResNet.input, output)

# Compile the model with Adam optimizer and binary cross-entropy loss
model.compile(optimizer=Adam(0.01), loss='binary_crossentropy')

# Train the model
model.fit(train_x, train_y, batch_size=64, epochs=15, validation_data=(test_x, test_y))

# Evaluate the model
model.evaluate(test_x, test_y)

# Make predictions on the test set
y_pred = (model.predict(test_x).reshape(-1,) >= 0.5).astype(int)

# Calculate accuracy
accuracy = np.sum(test_y == y_pred) / len(y_pred)
print(f"The model has an accuracy of: {accuracy * 100:.2f}%")

# Save the trained model
model.save('pneumonia_model.h5')  # Save in .h5 format
model.save('pneumonia_model.keras')  # Save in .keras format
