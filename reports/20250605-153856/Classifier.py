import tensorflow as tf
import pandas as pd
from datetime import datetime
import csv
import sys
import os
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_data(file):
    label4 = False
    if file == "validation1":
        file = "validation"
        label4 = True
    print(f"loading {file}")
    # read CSV file
    dataFrame = pd.read_csv(f'{file}.csv')

    images, labels = [], []

    if file == "train_balanced":
        file = "train"

    # iterate through the CSV file
    for _, row in dataFrame.iterrows():
        # transform images into np arrays
        img_id = row["image_id"]
        img_path = f"{file}/{img_id}.png"
        img_array = np.array(Image.open(img_path)) / 255.0 # normalize pixel values to [0, 1]
        images.append(img_array)


        # test files do not have labels
        if "label" in dataFrame.columns:
            if label4 == True:
                if row["label"] == 4:
                    labels.append(1)
                else:
                    labels.append(0)
            else:
                labels.append(row["label"])

    # np array with dtype float32
    images = np.array(images, dtype=np.float32)

    if len(labels) > 0:
        labels = np.array(labels)

    # convert lists to numpy arrays
    return np.array(images) if file == "test" else (images, labels)

def CNN4():
    input_shape = (100,100,3)
    pool_size = (2, 2) # pool size
    
    model = tf.keras.models.Sequential([
        # first convolutional layer
        tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D(pool_size=pool_size),

        # second convolutional layer
        tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=pool_size),

        # second convolutional layer
        tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=pool_size),

        # third convolutional layer
        tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=pool_size),

        # flatten the output
        tf.keras.layers.Flatten(), # transform into 1D vector
        tf.keras.layers.Dropout(0.3), # dropout layer to prevent overfitting
        tf.keras.layers.Dense(256, activation='relu'), # fully connected layer with 256 neurons, relu activated
        tf.keras.layers.Dropout(0.3), # another dropout layer
        tf.keras.layers.Dense(1, activation='sigmoid') # 2 classes (1 for label 4, 0 for others)
    ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy', # for binary labels
                  metrics=['accuracy'])
    
    model.summary()
    return model

def CNN():
    input_shape = (100, 100, 3) # 100x100 RGB image
    kernel_size = (3, 3) # filter size
    pool_size = (2, 2) # pool size

    model = tf.keras.models.Sequential([
        # first convolutional layer
        tf.keras.layers.Conv2D(filters=32, kernel_size=kernel_size, activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D(pool_size=pool_size),

        # second convolutional layer
        tf.keras.layers.Conv2D(filters=64, kernel_size=kernel_size, activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=pool_size),

        # second convolutional layer
        tf.keras.layers.Conv2D(filters=128, kernel_size=kernel_size, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=pool_size),

        # third convolutional layer
        tf.keras.layers.Conv2D(filters=256, kernel_size=kernel_size, activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=pool_size),

        # flatten the output
        tf.keras.layers.Flatten(), # transform into 1D vector
        tf.keras.layers.Dropout(0.3), # dropout layer to prevent overfitting
        tf.keras.layers.Dense(256, activation='relu'), # fully connected layer with 256 neurons, relu activated
        tf.keras.layers.Dropout(0.3), # another dropout layer
        tf.keras.layers.Dense(5, activation='softmax') # 5 classes (0-4 labels)
    ])

    # compile the model
    model.compile(optimizer='adam', 
                  loss='sparse_categorical_crossentropy', # for int labels
                  metrics=['accuracy'])
    
    # return the model summary
    model.summary()
    return model

def generate_confusion_matrix(real, prediction, names, precision):
    # generate a confusion matrix
    matrix = confusion_matrix(real, prediction)
    plt.figure(figsize=(12, 12))
    sns.heatmap(matrix, annot=True, fmt='d', cmap='Purples', xticklabels=names, yticklabels=names)
    plt.title("Confusion Matrix")
    plt.ylabel("Real labels")
    plt.xlabel("Predicted labels")
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    # Create a new folder with the timestamp
    folder_name = f"reports/{timestamp}"
    os.makedirs(folder_name, exist_ok=True)

    # Save the confusion matrix figure
    plt.savefig(f"{folder_name}/confusion_matrix{precision}.png")

    # Save the current code to the folder
    with open(__file__, 'r') as file:
        code_content = file.read()
    with open(f"{folder_name}/Classifier.py", 'w') as backup_file:
        backup_file.write(code_content)
    plt.close()  # close the plot

if __name__ == "__main__":

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3),
        tf.keras.callbacks.ModelCheckpoint('best_model.keras', save_best_only=True)
    ]

    if len(sys.argv) != 3:
        print("Example: python3 Classifier.py <epochs> <batch_size>")
        sys.exit(1)
    
    # save epochs and batch_size params
    epochs = int(sys.argv[1])
    batch_size = int(sys.argv[2])

    # load data 
    train_images, train_labels = load_data("train")
    val_images, val_labels = load_data("validation")
    test_images = load_data("test")

    # load data for label 4 cnn
    train_images_label4, train_labels_label4 = load_data("train_balanced")
    val_images_label4, val_labels_label4 = load_data("validation1")
    model = CNN4()
    model.fit(train_images_label4, train_labels_label4,
        validation_data=(val_images_label4, val_labels_label4),
        epochs=epochs, batch_size=batch_size,
        callbacks=callbacks,
        class_weight={0:1.0, 1:3.0})

    validation_loss4, validation_accuracy4 = model.evaluate(val_images_label4, val_labels_label4, verbose=0)
    print(f"Accuracy: {round(validation_accuracy4, 4)}, Loss: {round(validation_loss4, 4)}")

    # creating reports folder if not exists
    os.makedirs("reports", exist_ok=True)
    reports_csv = "reports/reports.csv"
    file_exists = os.path.isfile(reports_csv)

    with open(reports_csv, mode="a", newline="") as file:
        writer = csv.writer(file)

        # create the header
        if not file_exists:
            writer.writerow(["Epochs", "Batch_Size", "Accuracy", "Loss"])
        
        # write the results
        writer.writerow([epochs, batch_size, round(validation_accuracy4, 4), round(validation_loss4, 4)])
    
    val_predictions4 = model.predict(val_images_label4)
    val_label_predictions4 = (val_predictions4 > 0.5).astype(int).flatten()
    generate_confusion_matrix(val_labels_label4, val_label_predictions4, ['0', '1'], int(validation_accuracy4 * 100))

    # label names
    names = ['0', '1', '2', '3', '4']

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

'''
TODO: 
BATCH NORMALIZATION pe fiecare layer, pornind de la mai multe filtre pentru detalii fine
ZERO AUGMENTARE
EPOCI MICI, BATCH_SIZE 16
GlobalAveragePooling2D() in loc de flatten+dropout
'''
