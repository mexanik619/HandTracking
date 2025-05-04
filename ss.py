# train_asl_model.py
import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

DATA_DIR = "asl_dataset"
labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

# Load data
X, y = [], []
for file in os.listdir(DATA_DIR):
    if file.endswith(".npy"):
        label = file.split("_")[0]
        X.append(np.load(os.path.join(DATA_DIR, file)))
        y.append(labels.index(label))

X = np.array(X)
y = to_categorical(y, num_classes=len(labels))

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build model
model = Sequential([
    Dense(128, activation='relu', input_shape=(63,)),
    Dropout(0.4),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(len(labels), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=20, batch_size=16, validation_data=(X_test, y_test))

# Save model
model.save("asl_model.h5")
