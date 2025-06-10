# Machine Learning Models Documentation for Image Classification

## Table of Contents
1. [Overview](#overview)
2. [Convolutional Neural Network](#cnn-model-convolutional-neural-network)
3. [Random Forest Model](#random-forest-model)
4. [Installation and Dependencies](#installation-and-dependencies)
5. [Usage](#usage)
6. [Results and Reports](#results-and-reports)

## Overview

This documentation describes two machine learning models implemented for image classification into 5 classes (0-4):
- **CNN (Convolutional Neural Network)** - deep learning model
- **Random Forest** - ensemble learning model

Both models process 100x100 pixel RGB images and generate predictions along with performance reports.

## CNN Model (Convolutional Neural Network)

### Model Architecture

```python
# CNN Structure
input_shape = (100, 100, 3)  # 100x100 RGB images
kernel_size = (3, 3)         # Filter size
pool_size = (2, 2)           # Pooling size

Layers:
1. Conv2D(64 filters) + MaxPooling2D
2. Conv2D(128 filters) + MaxPooling2D  
3. Conv2D(256 filters) + BatchNormalization + MaxPooling2D
4. Conv2D(512 filters) + MaxPooling2D
5. Flatten + Dropout(0.5)
6. Dense(256, relu) + Dropout(0.5)
7. Dense(5, softmax) # Output layer
```

### CNN Characteristics

| Feature | Value |
|---------|-------|
| **Optimizer** | Adam |
| **Loss Function** | Sparse Categorical Crossentropy |
| **Dropout Rate** | 0.5 |
| **Batch Normalization** | Yes (layer 3) |
| **Early Stopping** | Yes (patience=10) |
| **Learning Rate Reduction** | Yes (factor=0.2, patience=3) |
| **Class Weights** | {0: 1.0, 1: 1.5, 2: 1.0, 3: 1.0, 4: 2.5} |

### CNN Training Parameters

- **Epochs**: Specified as command line argument
- **Batch Size**: Specified as command line argument
- **Callbacks**:
  - EarlyStopping to prevent overfitting
  - ReduceLROnPlateau for learning rate optimization
  - ModelCheckpoint to save the best model

### CNN Usage

```bash
python3 cnn.py <epochs> <batch_size>

# Example:
python3 cnn.py 50 32
```

### Main CNN Functions

#### `CNN()`
Defines and compiles the convolutional neural network architecture.

#### `CNN_train(train_data, validation_data, epochs, batch_size, class_weight)`
Trains the CNN model with the following parameters:
- `train_data`: Tuple (training_images, training_labels)
- `validation_data`: Tuple (validation_images, validation_labels)
- `epochs`: Number of training epochs
- `batch_size`: Batch size
- `class_weight`: Weights for class balancing

#### `create_report(trained_model, validation_data)`
Evaluates the model and saves results to `reports/reports.csv`.

## Random Forest Model

### Model Configuration

```python
# Random Forest Parameters
n_estimators = 500      # Number of trees
max_depth = 20         # Maximum tree depth
random_state = 69      # Seed for reproducibility
n_jobs = -1           # Use all available cores
```

### Random Forest Characteristics

| Feature | Value |
|---------|-------|
| **Number of Trees** | 500 |
| **Maximum Depth** | 20 |
| **Parallelization** | All cores (n_jobs=-1) |
| **Random State** | 69 |
| **Data Processing** | Flatten images to 1D |

### Random Forest Usage

```bash
python3 random_forest.py
```

### Main Random Forest Functions

#### `RF()`
Initializes the Random Forest classifier with optimized parameters.

#### `RF_train(train_data)`
Trains the Random Forest model:
- Reshapes images to 1D format (flatten)
- Trains the classifier on flattened data

#### `create_report(trained_model, validation_data)`
Evaluates the model and saves results to `RFreports/reports.csv`.

## Installation and Dependencies

### Complete Dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Execution Steps

1. **Running CNN**:
   ```bash
   python3 cnn.py 50 32  # 50 epochs, batch size 32
   ```

2. **Running RF**:
   ```bash
   python3 random_forest.py
   ```

### Common Functions

#### `load_data(file)`
Loads data from CSV files and corresponding images:
- **Input**: File name ('train', 'validation', 'test')
- **Output**: 
  - For train/validation: tuple (images, labels)
  - For test: images only
- **Processing**: Normalize pixels to [0,1]

#### `generate_confusion_matrix(real, prediction, names, precision)`
Generates and saves confusion matrix:
- Creates visualization with seaborn/matplotlib
- Saves image in reports directory
- Backs up source code

#### `make_predictions(trained_model, validation_data, test_images, validation_accuracy)`
Generates final predictions:
- Evaluates on validation data
- Generates confusion matrix
- Saves predictions to `predictions.csv`

### Evaluation Metrics

- **Accuracy**: Accuracy on validation set
- **Confusion Matrix**: Confusion matrix for detailed analysis
- **Loss** (CNN only): Loss on validation set

## Model Comparison

| Aspect | CNN | Random Forest |
|--------|-----|---------------|
| **Training Time** | Longer | Shorter |
| **Memory Usage** | Higher | Lower |
| **Image Performance** | Superior | Good |
| **Interpretability** | Low | High |
| **Hyperparameters** | Many | Few |
| **Parallelization** | GPU/CPU | CPU multi-core |

## Usage Recommendations

### CNN
- **Recommended for**: Large datasets, maximum performance
- **Required Resources**: GPU (recommended)
- **Training Time**: 2-3 minutes (on a powerful GPU droplet)

### Random Forest
- **Recommended for**: Rapid prototyping, limited resources
- **Required Resources**: Multi-core CPU
- **Training Time**: 2-3 minutes (on CPU)

## Results and Reports

## Troubleshooting

### Common CNN Issues

1. **Out of Memory**: Reduce batch_size
2. **Overfitting**: Enable early stopping (already implemented)
3. **Slow Training**: Use GPU if available

### Common Random Forest Issues

1. **Insufficient Memory**: Reduce n_estimators
2. **Long Training Time**: Reduce max_depth or n_estimators
3. **Poor Performance**: Try feature engineering

## Implementation Details

### Data Loading Process

Both models use the same data loading function with error handling:

```python
def load_data(file):
    # Read CSV file
    dataFrame = pd.read_csv(f'{file}.csv')
    images, labels = [], []
    
    # Process each image
    for _, row in dataFrame.iterrows():
        img_id = row["image_id"]
        img_path = f"{file}/{img_id}.png"
        try:
            # Load and normalize image
            img_array = np.array(Image.open(img_path)) / 255.0
            images.append(img_array)
            
            # Add label if available
            if "label" in dataFrame.columns:
                labels.append(row["label"])
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            continue
    
    return (np.array(images), np.array(labels)) if labels else np.array(images)
```

### Report Generation

Both models generate comprehensive reports including:
- Performance metrics (accuracy, loss)
- Confusion matrices with visualization
- Timestamped report directories
- Source code backups for reproducibility

### Prediction Pipeline

1. **Model Training**: Train on training set
2. **Validation**: Evaluate on validation set
3. **Confusion Matrix**: Generate visual analysis
4. **Test Predictions**: Make final predictions on test set
5. **Output**: Save predictions to CSV file

## Best Practices

### For CNN Model
- Use appropriate batch sizes (16, 32, 64)
- Monitor validation loss to prevent overfitting
- Consider data augmentation for larger datasets
- Use GPU acceleration when available

### For Random Forest Model
- Balance between n_estimators and training time
- Consider feature selection for high-dimensional data
- Use cross-validation for hyperparameter tuning
- Monitor memory usage with large datasets

## Error Handling

Both models include comprehensive error handling:
- Image loading errors are caught and logged
- Missing files are skipped with warnings
- Model training errors are reported with stack traces
- Graceful degradation when optional features fail

## Performance Optimization

### CNN Optimizations
- BatchNormalization for training stability
- Dropout layers for regularization
- Early stopping to prevent overfitting
- Learning rate scheduling for convergence

### Random Forest Optimizations
- Parallel processing with all CPU cores
- Optimized tree depth to balance bias-variance
- Efficient memory usage with proper data types
- Fast prediction with sklearn optimizations

## Conclusion

Both models provide robust solutions for image classification with specific advantages and trade-offs. The CNN offers superior performance for complex computer vision tasks, while Random Forest provides a fast and interpretable solution for prototyping and resource-constrained applications.

The comprehensive reporting system, error handling, and optimization features make both models production-ready for various image classification scenarios.