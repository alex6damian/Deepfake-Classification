import pandas as pd
import csv
import os
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier as RFC

def load_data(file):
    print("---------------------------------------")
    print(f"Loading {file} data...")
    print("---------------------------------------")
    # read CSV file
    dataFrame = pd.read_csv(f'{file}.csv')

    images, labels = [], []

    # iterate through the CSV file
    for _, row in dataFrame.iterrows():
        # transform images into np arrays
        img_id = row["image_id"]
        img_path = f"{file}/{img_id}.png"
        try:
            img_array = np.array(Image.open(img_path)) / 255.0 # normalize pixel values to [0, 1]
            images.append(img_array)

            # test files do not have labels
            if "label" in dataFrame.columns:
                labels.append(row["label"])
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            continue

    # np array with dtype float32
    images = np.array(images, dtype=np.float32)

    if len(labels) > 0:
        labels = np.array(labels)
        return (images, labels)
    else:
        return images

def RF():
    # random forest classifier
    model = RFC(n_estimators=500, random_state=69, n_jobs=-1)  # Added random_state and n_jobs for better performance
    return model

def RF_train(train_data):
    print("---------------------------------------")
    print("Training Random Forest model...")
    print("---------------------------------------")
    
    # create a random forest model
    model = RF()

    # reshape training data to 2D (flatten images)
    train_images_flat = train_data[0].reshape(train_data[0].shape[0], -1)
    
    print(f"Training data shape: {train_images_flat.shape}")
    print(f"Training labels shape: {train_data[1].shape}")

    # train the RF model
    model.fit(train_images_flat, train_data[1])

    print("Training completed!")
    return model

def create_report(trained_model, validation_data):
    print("---------------------------------------")
    print("Evaluating model...")
    print("---------------------------------------")
    
    # reshape validation data to 2D (flatten images)
    val_images_flat = validation_data[0].reshape(validation_data[0].shape[0], -1)
    
    # make predictions on validation data
    val_predictions = trained_model.predict(val_images_flat)
    
    # calculate accuracy using sklearn
    validation_accuracy = accuracy_score(validation_data[1], val_predictions)

    print("---------------------------------------")
    print(f"Validation Accuracy: {validation_accuracy:.4f}")
    print("---------------------------------------")

    # creating reports folder if not exists
    os.makedirs("RFreports", exist_ok=True)
    reports_csv = "RFreports/reports.csv"
    file_exists = os.path.isfile(reports_csv)

    with open(reports_csv, mode="a", newline="") as file:
        writer = csv.writer(file)

        # create the header
        if not file_exists:
            writer.writerow(["Accuracy"])
        
        # write the results
        writer.writerow([round(validation_accuracy, 4)])
    
    return validation_accuracy

def generate_confusion_matrix(real, prediction, names, precision):
    # generate a confusion matrix
    matrix = confusion_matrix(real, prediction)
    plt.figure(figsize=(12, 12))
    sns.heatmap(matrix, annot=True, fmt='d', cmap='Purples', xticklabels=names, yticklabels=names)
    plt.title("Confusion Matrix - Random Forest")
    plt.ylabel("Real labels")
    plt.xlabel("Predicted labels")
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    # folder creation
    folder_name = f"RFreports/{timestamp}"
    os.makedirs(folder_name, exist_ok=True)

    # save the matrix
    plt.savefig(f"{folder_name}/confusion_matrix_{round(precision,4)}.png")
    plt.show()  # Display the matrix
    plt.close()

    # save code configuration
    with open(__file__, 'r') as file:
        code_content = file.read()
    with open(f"{folder_name}/RandomForest_Classifier.py", 'w') as backup_file:
        backup_file.write(code_content)
    
def make_predictions(trained_model, validation_data, test_images, validation_accuracy):
    '''
    validation_data contains images and labels
    test_images contains only images
    '''
    print("---------------------------------------")
    print("Making predictions...")
    print("---------------------------------------")
    
    # label names
    names = ['0', '1', '2', '3', '4']

    # reshape validation data for prediction
    val_images_flat = validation_data[0].reshape(validation_data[0].shape[0], -1)
    
    # predict validation data
    val_label_predictions = trained_model.predict(val_images_flat)
    generate_confusion_matrix(validation_data[1], val_label_predictions, names, validation_accuracy)

    # reshape test data for prediction
    test_images_flat = test_images.reshape(test_images.shape[0], -1)
    
    # make predictions on test data
    test_predictions = trained_model.predict(test_images_flat)

    # no need for argmax
    label_predictions = test_predictions

    # dataframe for output
    output_dataFrame = pd.DataFrame({
        "image_id": pd.read_csv("test.csv")["image_id"], 
        "label": label_predictions
    })

    # save to csv
    output_dataFrame.to_csv("predictions.csv", index=False)
    print(f"Predictions saved! Shape: {output_dataFrame.shape}")

if __name__ == "__main__":
    try:
        # load data
        train_images, train_labels = load_data("train")
        val_images, val_labels = load_data("validation")
        test_images = load_data("test")
        
        print(f"Data loaded successfully:")
        print(f"Train: {train_images.shape}, Labels: {train_labels.shape}")
        print(f"Validation: {val_images.shape}, Labels: {val_labels.shape}")
        print(f"Test: {test_images.shape}")
        
        # train the model
        trained_model = RF_train((train_images, train_labels))

        # create a report and get the accuracy
        validation_accuracy = create_report(trained_model, (val_images, val_labels))
        
        # generate confusion matrix and save predictions
        make_predictions(trained_model, (val_images, val_labels), test_images, validation_accuracy)

        print("---------------------------------------")
        print("The model has been trained, predictions can be found in predictions.csv")
        print("---------------------------------------")
        
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()