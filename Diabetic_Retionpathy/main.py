import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import shutil
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam

# Load the dataset
df = pd.read_csv('train.csv')
#https://www.kaggle.com/datasets/sovitrath/diabetic-retinopathy-224x224-gaussian-filtered

# info about the dataset
# print(df.info())   # id_code , diagnosis
# print(df.head()

# Map numerical diagnosis to categorical labels
multiple= { 0: 'No_DR', 1: 'Mild',  2: 'Moderate', 3: 'Severe', 4: 'Proliferate_DR'}
binary = { 0: 'No', 1: 'YES', 2: 'YES', 3: 'YES', 4: 'YES' }

# Add new columns for binary and multiple classification
df['detection'] =  df['diagnosis'].map(binary.get)
df['levels'] = df['diagnosis'].map(multiple.get)

# updated DataFrame
#print(df.info())
#print(df.head())
#print(df['detection'].value_counts())
"""YES    1857
   No     1805"""
#print(df['levels'].value_counts())
"""levels
No_DR             1805
Moderate           999
Mild               370
Proliferate_DR     295
Severe             193"""
#The dataset is imbalanced on levels

# Stratified split (bcz of imbalanced dataset(levels) ) of the data into train, validation, and test sets
train_test, val = train_test_split(df, test_size = 0.15, stratify = df['levels'])
train, test = train_test_split(train_test, test_size = 0.15 / 0.85 , stratify = train_test['levels'])

#print(train['level'].value_counts())
#print(val['level'].value_counts())
#print(test['level'].value_counts())

# Function to copy images into respective folders
def copy_images(dataframe, src_dir, dest_base_dir):
    for index, row in dataframe.iterrows():
        level = row['levels']
        detect = row['detection']
        img_id = row['id_code'] + ".png"
        source = os.path.join(src_dir, level, img_id)
        dest_dir = os.path.join(dest_base_dir, detect)

        # Create the destination directory if it doesn't exist
        os.makedirs(dest_dir, exist_ok=True)

        # Copy the image to the destination
        destination = os.path.join(dest_dir, img_id)
        shutil.copy(source, destination)

# Define source and destination directories
src_dir = r'C:\Users\Rohit Negi\Desktop\Diabetic_Retionpathy\gaussian_filtered_images\gaussian_filtered_images'
train_dir = r'C:\Users\Rohit Negi\Desktop\Diabetic_Retionpathy\train'
test_dir = r'C:\Users\Rohit Negi\Desktop\Diabetic_Retionpathy\test'
val_dir = r'C:\Users\Rohit Negi\Desktop\Diabetic_Retionpathy\val'

# Uncomment to copy images
""" 
# Copy images for the train, val, and test sets
copy_images(train, src_dir, train_dir)
copy_images(val, src_dir, val_dir)
copy_images(test, src_dir, test_dir)
"""

# Setting up ImageDataGenerator for train/val/test
train_directory = 'train'
validation_directory = 'val'
test_directory = 'test'

train_data = ImageDataGenerator(rescale=1./255).flow_from_directory(train_directory, target_size=(224, 224), shuffle=True)
val_data = ImageDataGenerator(rescale=1./255).flow_from_directory(validation_directory, target_size=(224, 224), shuffle=True)
test_data = ImageDataGenerator(rescale=1./255).flow_from_directory(test_directory, target_size=(224, 224), shuffle=False)

# Set random seeds for reproducibility
tf.random.set_seed(42)
"""It ensures that the operations in TensorFlow that generate random numbers (like the initialization of weights in 
a neural network) will produce the same sequence of numbers every time I run this  code."""
np.random.seed(42)
#Similarly, this line sets the seed for NumPy's random number generator.

# Build the model
model = Sequential()

model.add(Conv2D(16, (3, 3), padding="valid", input_shape=(224, 224, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())#this layer normalizes the activations from the previous layer,
# which can help in speeding up training and reducing the sensitivity to network initialization.

model.add(Conv2D(32, (3, 3), padding="valid", activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())

model.add(Conv2D(64, (4, 4), padding="valid", activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())

model.add(Flatten()) #This layer flattens the 3D output of the last pooling layer into a 1D array to be fed into the dense layers.
model.add(Dense(64, activation='relu'))  # These are fully connected layers.
# The first Dense layer with 64 units for learning non-linear combinations, followed by a Dropout layer to prevent overfitting.
model.add(Dropout(0.3))  # Add dropout layer to prevent overfitting
model.add(Dense(2, activation='softmax')) # 2 units for binary classification


# Compile the model with the Adam optimizer and appropriate learning rate
model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model with early stopping to avoid overfitting
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=5, restore_best_weights=True
)
final = model.fit( train_data, epochs=15,validation_data=val_data,callbacks=[early_stopping])
# Save the model to a file
model.save("dr_model.h5")

# Evaluate the model on the test set
test_results = model.evaluate(test_data)
print(f"Test Loss: {test_results[0]}, Test Accuracy: {test_results[1]}")
#Test Loss: 0.14322102069854736, Test Accuracy: 0.9509090781211853