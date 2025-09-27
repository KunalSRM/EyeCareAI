# # # # src/train.py
# # # """
# # # Train script using transfer learning (EfficientNetB0).
# # # Saves model to models/eye_care_model.h5
# # # """
# # # import os
# # # import pandas as pd
# # # import numpy as np
# # # from tensorflow import keras
# # # from tensorflow.keras import layers
# # # from tensorflow.keras.applications import EfficientNetB0
# # # from tensorflow.keras.applications.efficientnet import preprocess_input
# # # from tensorflow.keras.preprocessing.image import ImageDataGenerator
# # # from sklearn.utils.class_weight import compute_class_weight

# # # ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# # # PROCESSED_DIR = os.path.join(ROOT, "data", "processed")
# # # MODELS_DIR = os.path.join(ROOT, "models")
# # # os.makedirs(MODELS_DIR, exist_ok=True)

# # # TRAIN_CSV = os.path.join(PROCESSED_DIR, "train_labels.csv")
# # # VAL_CSV = os.path.join(PROCESSED_DIR, "val_labels.csv")
# # # IMG_SIZE = (224, 224)
# # # BATCH_SIZE = 16
# # # EPOCHS = 12
# # # MODEL_PATH = os.path.join(MODELS_DIR, "eye_care_model.h5")

# # # def make_generators():
# # #     train_df = pd.read_csv(TRAIN_CSV)
# # #     val_df = pd.read_csv(VAL_CSV)

# # #     # Ensure string labels for flow_from_dataframe
# # #     train_df['diagnosis'] = train_df['diagnosis'].astype(str)
# # #     val_df['diagnosis'] = val_df['diagnosis'].astype(str)

# # #     train_datagen = ImageDataGenerator(
# # #         preprocessing_function=preprocess_input,
# # #         rotation_range=20,
# # #         width_shift_range=0.1,
# # #         height_shift_range=0.1,
# # #         shear_range=0.05,
# # #         zoom_range=0.1,
# # #         horizontal_flip=True,
# # #         brightness_range=(0.8,1.2),
# # #         fill_mode='nearest'
# # #     )
# # #     val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

# # #     train_gen = train_datagen.flow_from_dataframe(
# # #         dataframe=train_df,
# # #         x_col='filepath',
# # #         y_col='diagnosis',
# # #         target_size=IMG_SIZE,
# # #         batch_size=BATCH_SIZE,
# # #         class_mode='categorical',
# # #         shuffle=True
# # #     )
# # #     val_gen = val_datagen.flow_from_dataframe(
# # #         dataframe=val_df,
# # #         x_col='filepath',
# # #         y_col='diagnosis',
# # #         target_size=IMG_SIZE,
# # #         batch_size=BATCH_SIZE,
# # #         class_mode='categorical',
# # #         shuffle=False
# # #     )

# # #     # Compute class weights
# # #     classes = np.unique(train_df['diagnosis'].astype(int))
# # #     y_integers = train_df['diagnosis'].astype(int)
# # #     class_weights = compute_class_weight('balanced', classes=classes, y=y_integers)
# # #     class_weights = {int(c): w for c, w in zip(classes, class_weights)}
# # #     print("Class weights:", class_weights)
# # #     return train_gen, val_gen, class_weights

# # # def build_model(num_classes=5):
# # #     base = EfficientNetB0(include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3), weights='imagenet')
# # #     base.trainable = False  # fine-tune later if wanted

# # #     inp = keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
# # #     x = base(inp, training=False)
# # #     x = layers.GlobalAveragePooling2D()(x)
# # #     x = layers.Dropout(0.4)(x)
# # #     x = layers.Dense(256, activation='relu')(x)
# # #     x = layers.Dropout(0.3)(x)
# # #     out = layers.Dense(num_classes, activation='softmax')(x)
# # #     model = keras.Model(inputs=inp, outputs=out)
# # #     model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
# # #                   loss='categorical_crossentropy',
# # #                   metrics=['accuracy'])
# # #     return model

# # # def train():
# # #     train_gen, val_gen, class_weights = make_generators()
# # #     model = build_model(num_classes=train_gen.num_classes)

# # #     # Callbacks
# # #     callbacks = [
# # #         keras.callbacks.ModelCheckpoint(MODEL_PATH, monitor='val_accuracy', save_best_only=True, verbose=1),
# # #         keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1),
# # #         keras.callbacks.EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True, verbose=1)
# # #     ]

# # #     history = model.fit(
# # #         train_gen,
# # #         epochs=EPOCHS,
# # #         validation_data=val_gen,
# # #         class_weight=class_weights,
# # #         callbacks=callbacks
# # #     )

# # #     # Optionally fine-tune: unfreeze some layers and train more (commented)
# # #     # base.trainable = True
# # #     # for layer in base.layers[:-30]:
# # #     #     layer.trainable = False
# # #     # model.compile(optimizer=keras.optimizers.Adam(1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
# # #     # model.fit(...)

# # #     print("Saved model to:", MODEL_PATH)

# # # if __name__ == "__main__":
# # #     if not os.path.exists(TRAIN_CSV):
# # #         raise FileNotFoundError("Run src/preprocess.py first to create train/val CSVs.")
# # #     train()
# # # train.py
# # import os
# # import pandas as pd
# # import numpy as np
# # from tensorflow.keras.preprocessing.image import ImageDataGenerator
# # from tensorflow.keras.models import Sequential
# # from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
# # from tensorflow.keras.optimizers import Adam
# # from tensorflow.keras.callbacks import ModelCheckpoint

# # # Paths
# # TRAIN_CSV = "../data/raw/train.csv"
# # TRAIN_DIR = "../data/raw/train_images"
# # MODEL_PATH = "../models/eye_care_model.h5"

# # # Parameters
# # IMG_HEIGHT, IMG_WIDTH = 224, 224
# # BATCH_SIZE = 32
# # EPOCHS = 15

# # # Build simple CNN
# # def build_model(num_classes):
# #     model = Sequential([
# #         Conv2D(32, (3,3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
# #         MaxPooling2D(2,2),
# #         Conv2D(64, (3,3), activation='relu'),
# #         MaxPooling2D(2,2),
# #         Flatten(),
# #         Dense(128, activation='relu'),
# #         Dropout(0.5),
# #         Dense(num_classes, activation='softmax')
# #     ])
# #     model.compile(optimizer=Adam(learning_rate=1e-4),
# #                   loss='sparse_categorical_crossentropy',
# #                   metrics=['accuracy'])
# #     return model

# # # Load train CSV
# # df = pd.read_csv(TRAIN_CSV)

# # # Train-validation split
# # from sklearn.model_selection import train_test_split
# # train_df, val_df = train_test_split(df, test_size=0.15, stratify=df['diagnosis'], random_state=42)

# # # Data generators
# # train_datagen = ImageDataGenerator(rescale=1./255, horizontal_flip=True, vertical_flip=True)
# # val_datagen = ImageDataGenerator(rescale=1./255)

# # train_gen = train_datagen.flow_from_dataframe(
# #     train_df,
# #     directory=TRAIN_DIR,
# #     x_col='id_code',
# #     y_col='diagnosis',
# #     target_size=(IMG_HEIGHT, IMG_WIDTH),
# #     batch_size=BATCH_SIZE,
# #     class_mode='raw'  # Use 'raw' for regression style sparse labels
# # )

# # val_gen = val_datagen.flow_from_dataframe(
# #     val_df,
# #     directory=TRAIN_DIR,
# #     x_col='id_code',
# #     y_col='diagnosis',
# #     target_size=(IMG_HEIGHT, IMG_WIDTH),
# #     batch_size=BATCH_SIZE,
# #     class_mode='raw'
# # )

# # # Calculate number of classes dynamically
# # num_classes = df['diagnosis'].nunique()
# # print(f"Number of classes: {num_classes}")

# # # Calculate class weights
# # from sklearn.utils import class_weight
# # class_weights = class_weight.compute_class_weight(
# #     'balanced',
# #     classes=np.unique(train_df['diagnosis']),
# #     y=train_df['diagnosis']
# # )
# # class_weights_dict = dict(enumerate(class_weights))
# # print("Class weights:", class_weights_dict)

# # # Build model
# # model = build_model(num_classes=num_classes)

# # # Checkpoint
# # os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
# # checkpoint = ModelCheckpoint(MODEL_PATH, monitor='val_accuracy', save_best_only=True, verbose=1)

# # # Train
# # history = model.fit(
# #     train_gen,
# #     validation_data=val_gen,
# #     epochs=EPOCHS,
# #     class_weight=class_weights_dict,
# #     callbacks=[checkpoint]
# # )

# # print(f"✅ Model saved at: {MODEL_PATH}")

# import os
# import pandas as pd
# import numpy as np
# from sklearn.utils.class_weight import compute_class_weight
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# # Paths
# TRAIN_DIR = "../train_images"
# TEST_DIR = "../test_images"
# TRAIN_CSV = "../train.csv"
# TEST_CSV = "../test.csv"
# MODEL_PATH = "../models/eye_care_model.h5"

# # Parameters
# IMG_HEIGHT = 224
# IMG_WIDTH = 224
# BATCH_SIZE = 32
# EPOCHS = 10  # adjust as needed
# NUM_CLASSES = 5

# # Load CSVs
# train_df = pd.read_csv(TRAIN_CSV)
# test_df = pd.read_csv(TEST_CSV)

# # ✅ Fix: Append '.png' to match actual filenames
# train_df['id_code'] = train_df['id_code'].apply(lambda x: f"{x}.png")
# test_df['id_code'] = test_df['id_code'].apply(lambda x: f"{x}.png")

# # Split train into train/validation
# from sklearn.model_selection import train_test_split
# train_df, val_df = train_test_split(train_df, test_size=0.15, random_state=42, stratify=train_df['diagnosis'])

# # Data augmentation
# train_datagen = ImageDataGenerator(
#     rescale=1./255,
#     rotation_range=15,
#     zoom_range=0.1,
#     horizontal_flip=True
# )

# val_datagen = ImageDataGenerator(rescale=1./255)

# # Generators
# train_gen = train_datagen.flow_from_dataframe(
#     train_df,
#     directory=TRAIN_DIR,
#     x_col='id_code',
#     y_col='diagnosis',
#     target_size=(IMG_HEIGHT, IMG_WIDTH),
#     batch_size=BATCH_SIZE,
#     class_mode='raw'
# )

# val_gen = val_datagen.flow_from_dataframe(
#     val_df,
#     directory=TRAIN_DIR,
#     x_col='id_code',
#     y_col='diagnosis',
#     target_size=(IMG_HEIGHT, IMG_WIDTH),
#     batch_size=BATCH_SIZE,
#     class_mode='raw'
# )

# # Compute class weights
# class_weights = compute_class_weight(
#     class_weight='balanced',
#     classes=np.unique(train_df['diagnosis']),
#     y=train_df['diagnosis']
# )
# class_weights = dict(enumerate(class_weights))
# print(f"Class weights: {class_weights}")

# # Build a simple CNN
# model = Sequential([
#     Conv2D(32, (3,3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
#     MaxPooling2D(2,2),
#     Conv2D(64, (3,3), activation='relu'),
#     MaxPooling2D(2,2),
#     Conv2D(128, (3,3), activation='relu'),
#     MaxPooling2D(2,2),
#     Flatten(),
#     Dense(128, activation='relu'),
#     Dropout(0.5),
#     Dense(NUM_CLASSES, activation='softmax')
# ])

# model.compile(optimizer=Adam(learning_rate=1e-4),
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])

# # Callbacks
# checkpoint = ModelCheckpoint(MODEL_PATH, monitor='val_accuracy', save_best_only=True, verbose=1)
# early_stop = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)

# # Train
# history = model.fit(
#     train_gen,
#     validation_data=val_gen,
#     epochs=EPOCHS,
#     class_weight=class_weights,
#     callbacks=[checkpoint, early_stop]
# )

# print(f"✅ Model saved at: {MODEL_PATH}")

# src/train.py
"""
Train script for EyeCareAI – Diabetic Retinopathy Detection.
Uses a simple CNN (can upgrade to EfficientNetB0 later).
Saves model to models/eye_care_model.h5
"""

import os
import pandas as pd
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# ---------------- Paths ----------------
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TRAIN_DIR = os.path.join(ROOT_DIR, "data", "raw", "train_images")
TEST_DIR = os.path.join(ROOT_DIR, "data", "raw", "test_images")
TRAIN_CSV = os.path.join(ROOT_DIR, "data", "raw", "train.csv")
TEST_CSV = os.path.join(ROOT_DIR, "data", "raw", "test.csv")
MODEL_PATH = os.path.join(ROOT_DIR, "models", "eye_care_model.h5")
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

# ---------------- Parameters ----------------
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32
EPOCHS = 15  # increase if GPU available
NUM_CLASSES = 5

# ---------------- Load CSV ----------------
train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)

# Fix filenames to include .png
train_df['id_code'] = train_df['id_code'].apply(lambda x: f"{x}.png")
test_df['id_code'] = test_df['id_code'].apply(lambda x: f"{x}.png")

# Split into train and validation
train_df, val_df = train_test_split(
    train_df,
    test_size=0.15,
    stratify=train_df['diagnosis'],
    random_state=42
)

# ---------------- Data Generators ----------------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_dataframe(
    train_df,
    directory=TRAIN_DIR,
    x_col='id_code',
    y_col='diagnosis',
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='raw'
)

val_gen = val_datagen.flow_from_dataframe(
    val_df,
    directory=TRAIN_DIR,
    x_col='id_code',
    y_col='diagnosis',
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='raw'
)

# ---------------- Class Weights ----------------
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_df['diagnosis']),
    y=train_df['diagnosis']
)
class_weights_dict = dict(enumerate(class_weights))
print("Class weights:", class_weights_dict)

# ---------------- Build Model ----------------
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(NUM_CLASSES, activation='softmax')
])

model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# ---------------- Callbacks ----------------
checkpoint = ModelCheckpoint(MODEL_PATH, monitor='val_accuracy', save_best_only=True, verbose=1)
early_stop = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)

# ---------------- Train ----------------
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    class_weight=class_weights_dict,
    callbacks=[checkpoint, early_stop]
)

print(f"✅ Model saved at: {MODEL_PATH}")
