## IMDB Movie Review Classification

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

### Usage

1. Install the required libraries: `pip install tensorflow tensorflow-datasets numpy matplotlib`
2. Run the Python script containing the code.

### Results

The accuracy and loss graphs for each model are plotted using Matplotlib. The graphs show the training and validation performance of the models over the specified number of epochs.

### Future Improvements

- Experiment with different hyperparameters to improve model performance.
- Try other deep learning architectures or ensemble methods to further enhance the sentiment analysis capabilities.
- Explore transfer learning techniques using pre-trained word embeddings or language models.

### Contributing

Contributions to this project are welcome. If you find any issues or have suggestions for improvements, please feel free to create a new issue or submit a pull request.

### License

This project is licensed under the [MIT License](LICENSE).
