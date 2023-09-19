import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import load_model

# Load the saved model
model = load_model("tezine_rock_country_not_cropped.h5")

# Now you can use loaded_model for predictions without retraining


spectrogram_dir = './Data/spectrograms_3sec/'
genres = os.listdir(spectrogram_dir)
print(genres)
num_classes = len(genres)
print(num_classes)

X = []
y = []

for i, genre in enumerate(genres):
    genre_dir = os.path.join(spectrogram_dir, genre)
    for filename in os.listdir(genre_dir):
        if filename.endswith('.png'):
            img_path = os.path.join(genre_dir, filename)
            img = keras.preprocessing.image.load_img(img_path, target_size=(128, 128))
            img_array = keras.preprocessing.image.img_to_array(img)
            X.append(img_array)
            y.append(i)

X = np.array(X)
y = np.array(y)

X, y = shuffle(X, y, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)


test_loss, test_acc = model.evaluate(X_test, y_test_encoded)
print(f'Test accuracy: {test_acc}')

y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)


# Get the indices of the top two predicted classes for each sample
top2_indices = np.argsort(y_pred, axis=1)[:, -2:]

# Check if the true class is in the top two predictions for each sample
top2_correct = np.array([y_test_encoded[i] in top2_indices[i] for i in range(len(y_test_encoded))])

# Calculate Top-2 accuracy
top2_accuracy = np.mean(top2_correct)
print(f'Top-2 accuracy: {top2_accuracy * 100:.2f}%')


# Get the indices of the top three predicted classes for each sample
top3_indices = np.argsort(y_pred, axis=1)[:, -3:]

# Check if the true class is in the top three predictions for each sample
top3_correct = np.array([y_test_encoded[i] in top3_indices[i] for i in range(len(y_test_encoded))])

# Calculate Top-3 accuracy
top3_accuracy = np.mean(top3_correct)
print(f'Top-3 accuracy: {top3_accuracy * 100:.2f}%')



y_test_class_names = [genres[label] for label in y_test]
y_pred_class_names = [genres[label] for label in y_pred_classes]

print(classification_report(y_test_class_names, y_pred_class_names, target_names=genres))