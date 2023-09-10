import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.metrics import classification_report

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


train_set_size = len(X_train)
print("Size of the train set:", train_set_size)

print(set(y_test))
print(set(genres))



label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

model = keras.Sequential([
    layers.Conv2D(8, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(16, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])



optimizer = keras.optimizers.Adam(learning_rate=0.0005)
model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

lr_scheduler = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-7
)

model.fit(X_train, y_train_encoded, epochs=80, batch_size=32, validation_split=0.2, steps_per_epoch=len(X_train) // 32,
     callbacks=[early_stopping, lr_scheduler])

test_loss, test_acc = model.evaluate(X_test, y_test_encoded)
print(f'Test accuracy: {test_acc}')

y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)


y_test_class_names = [genres[label] for label in y_test]
y_pred_class_names = [genres[label] for label in y_pred_classes]

print(classification_report(y_test_class_names, y_pred_class_names, target_names=genres))


