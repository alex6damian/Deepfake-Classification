import tensorflow as tf
import pandas as pd
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_data(file):
    # read CSV file
    dataFrame = pd.read_csv(f'{file}.csv')

    images, labels = [], []

    # iterate through the CSV file
    for _, row in dataFrame.iterrows():
        # transform images into np arrays
        img_id = row["image_id"]
        img_path = f"{file}/{img_id}.png"
        img_array = np.array(Image.open(img_path)) / 255.0 # normalize pixel values to [0, 1]
        images.append(img_array)


        # test files do not have labels
        if "label" in dataFrame.columns:
            labels.append(row["label"])

    # np array with dtype float32
    images = np.array(images, dtype=np.float32)

    if len(labels) > 0:
        labels = np.array(labels)

    # convert lists to numpy arrays
    return np.array(images) if file == "test" else (images, labels)

def augment_data(images, labels):
    # create a data augmentation layer
    augmented_data = ImageDataGenerator(
        rotation_range=20,
        height_shift_range=0.2,
        width_shift_range=0.2,
        zoom_range=0.2,
        shear_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    return augmented_data.flow(images, labels, batch_size=32)

def CNN():
    model = tf.keras.models.Sequential([
        # first convolutional layer
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)),
        tf.keras.layers.MaxPooling2D((2, 2)),

        # second convolutional layer
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),

        # third convolutional layer
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),

        # flatten the output
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(5, activation='softmax') # 5 classes (0-4 labels)
    ])

    # compile the model
    model.compile(optimizer='adam', 
                  loss='sparse_categorical_crossentropy', # for int labels
                  metrics=['accuracy'])
    
    # return the model summary
    model.summary()
    return model


if __name__ == "__main__":
    # load data
    train_images, train_labels = load_data("train")
    val_images, val_labels = load_data("validation")
    test_images = load_data("test")
    
    # get the CNN model
    model = CNN()

    # augment training data
    train_data_generator = augment_data(train_images, train_labels)

    # fit the model 
    model.fit(train_images, train_labels,
              validation_data=(val_images, val_labels),
              epochs=10, batch_size=32,
              callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)])

    model.fit(train_data_generator,
              validation_data=(val_images, val_labels),
              epochs=10, batch_size=32,
              callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)])
    
    # save predictions
    predictions = model.predict(test_images)

    # get highest probability labels
    label_predictions = np.argmax(predictions, axis=1) 

    # dataframe for output
    output_dataFrame = pd.DataFrame({
        "image_id": pd.read_csv("test.csv")["image_id"], 
        "label": label_predictions
    })

    # save to csv
    output_dataFrame.to_csv("predictions.csv", index=False)