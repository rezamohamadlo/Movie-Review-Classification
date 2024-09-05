## IMDB Movie Review Classification
(This project has been done as a programing assignment of course "Natural Language Processing in Tensorflow" in Coursera website.

### Overview

This project aims to classify IMDB movie reviews as either positive or negative using various deep learning architectures. It leverages the power of techniques like Embedding, Flatten, LSTM, GRU, and Convolutional Neural Networks (CNN) to generate coherent and stylistically similar text based on an initial seed input. The model is built using TensorFlow and Keras and includes components for data preprocessing, model training, and text generation.

### Key Components

1. **Data Preprocessing**:
   - The IMDB movie review dataset is loaded and preprocessed using TensorFlow Datasets (TFDS).
   - Text data is converted to lowercase, tokenized into sequences, and padded to ensure uniform input size.

2. **Model Architectures**:
   - **Embedding + Flatten**:
     - Embedding layer converts input text to dense vectors.
     - Flatten layer converts 2D embeddings to 1D vectors.
     - Dense layers for classification.
   - **Embedding + LSTM**:
     - Embedding layer converts input text to dense vectors.
     - Bidirectional LSTM layer captures contextual information.
     - Dense layers for classification.
   - **Embedding + GRU**:
     - Embedding layer converts input text to dense vectors.
     - Bidirectional GRU layer captures contextual information.
     - Dense layers for classification.
   - **Embedding + CNN**:
     - Embedding layer converts input text to dense vectors.
     - Convolutional layer extracts local features.
     - Global average pooling aggregates features.
     - Dense layers for classification.

3. **Training and Evaluation**:
   - Each model is trained on the preprocessed IMDB dataset for a specified number of epochs.
   - Training and validation accuracy and loss are plotted using Matplotlib to monitor model performance.

### Requirements

- TensorFlow
- TensorFlow Datasets (TFDS)
- NumPy
- Matplotlib


### Results

The accuracy and loss graphs for each model are plotted using Matplotlib. The graphs show the training and validation performance of the models over the specified number of epochs.



## Files

- **`IMDB-Review-Classification.ipynb`**: Jupyter Notebook containing the complete implementation of the text generation model.
- **`data-preprocessing.ipynb`**: Jupyter Notebook for data preprocessing tasks, including loading and preparing the Shakespearean sonnets dataset.
- **`readme.md`**: This file, providing an overview and details about the project.
