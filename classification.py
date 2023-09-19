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


spectrogram_dir = './Data/spectrograms_3sec_cropped/'
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
            img = keras.preprocessing.image.load_img(img_path, target_size=(256, 256))
            img_array = keras.preprocessing.image.img_to_array(img)
            X.append(img_array)
            y.append(i)

X = np.array(X)
y = np.array(y)

X, y = shuffle(X, y, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


train_set_size = len(X_train)
print("Size of the train set:", train_set_size)

print(set(y_test))
print(set(genres))



label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

model = keras.Sequential([
    layers.Conv2D(8, (3, 3), activation='relu', input_shape=(256, 256, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(16, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(256, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])



optimizer = keras.optimizers.Adam(learning_rate=0.0005)
model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy']
              )

early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=25,
    restore_best_weights=True
)

lr_scheduler = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=8,
    min_lr=1e-7
)
#class_weights = {0: 1.5, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.5, 5: 1.0, 6: 1.0, 7: 1.0, 8: 1.0, 9: 1.0}


model.fit(X_train, y_train_encoded, epochs=80, batch_size=32, validation_split=0.2,
     callbacks=[early_stopping, lr_scheduler])
model.save("najbolji_model_dodat256.h5")

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


