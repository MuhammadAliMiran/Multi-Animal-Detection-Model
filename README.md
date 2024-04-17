# Animal Kingdom Image Classification

This repository contains code for building a convolutional neural network (CNN) to classify images of animals into 10 categories. The code includes data loading, preprocessing, model architecture, training, and evaluation.

## Libraries and Helper Functions

- Data science libraries: numpy, pandas, tensorflow, scikit-learn, itertools, random, matplotlib, cv2, seaborn
- Tensorflow libraries: keras, layers, models, callbacks, preprocessing, optimizers
- System libraries: pathlib, os.path
- Metrics: classification_report, confusion_matrix

Helper functions are imported from a separate file `helper_functions.py` which can be downloaded from the given URL. These functions include:

- `create_tensorboard_callback`
- `plot_loss_curves`
- `unzip_data`
- `compare_historys`
- `walk_through_dir`
- `pred_and_plot`

## Data Loading and Preprocessing

- The dataset is loaded from a directory containing 10 subdirectories, each with images of a different animal. The data is loaded into a pandas dataframe with two columns: `Filepath` and `Label`. The `Filepath` column contains the file path of each image and the `Label` column contains the corresponding animal category.
- The data is split into training and testing sets using `train_test_split` from scikit-learn. The training set is further split into training and validation sets. Data augmentation is applied using `ImageDataGenerator` from tensorflow.

## Model Architecture

- The model architecture consists of a pre-trained EfficientNetB7 model with top layers removed, followed by several dense layers with batch normalization and dropout. The model is compiled with the Adam optimizer and categorical cross-entropy loss.

## Training

- The model is trained using the `fit` method from keras. The training is monitored using several callbacks including `EarlyStopping`, `ModelCheckpoint`, `ReduceLROnPlateau`, and `create_tensorboard_callback`.

## Evaluation

- The model is evaluated using the `evaluate` method from keras. The test loss and accuracy are printed. Loss curves and classification reports are plotted.

## Grad-Cam Visualization

- Grad-Cam visualization is applied to the model to visualize the parts of the input images that contribute most to the model's predictions.
