import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import pickle

# Load your data
df = pd.read_excel('data.xlsx')

# Prepare your data
X = df[['TEMP', 'HUM', 'MOISTURE']].values
y = df['VALVE STATE'].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train your deep learning model using TensorFlow
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(3,)),  # Input layer with 3 features (temperature, humidity, moisture)
    tf.keras.layers.Dense(64, activation='relu'),  # Hidden layer with 64 units and ReLU activation
    tf.keras.layers.Dense(32, activation='relu'),  # Hidden layer with 32 units and ReLU activation
    tf.keras.layers.Dense(1, activation='sigmoid')  # Output layer with 1 unit and sigmoid activation (for binary classification)
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the deep learning model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the deep learning model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Deep Learning Model Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

# Create and train your SVM model using scikit-learn
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)

# Save the SVM model to a file using pickle
with open('svm_model.pkl', 'wb') as model_file:
    pickle.dump(svm_model, model_file)

# Now you have both your deep learning and SVM models saved

