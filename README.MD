# Sentiment Analysis of Hilton Hotel London Reviews

## Overview

This project develops machine learning and deep learning models to classify Hilton Hotel London reviews into 1–5 star ratings using scraped TripAdvisor data. The solution includes:
- Data collection and comprehensive preprocessing
- Multiple ML model training and evaluation (Logistic Regression, Random Forest, SVM, etc.)
- Deep learning approaches with ANN and Bidirectional LSTM
- Detailed exploratory data analysis and visualization

Achievements:
- Achieved **~48% validation accuracy** with Logistic Regression
- Minimal overfitting with errors typically within ±1 star of true ratings
- Identified sentiment patterns across different hotel locations
- Built a reusable pipeline for review classification and business insight generation

## Features

### Core Analysis
- **Data Collection**: Scraped 17,538 reviews from 5 Hilton London hotels on TripAdvisor
- **Text Preprocessing**: Tokenization, lemmatization, stopword removal, and custom filtering
- **Feature Engineering**: TF-IDF vectorization for reviews and summaries
- **Class Balancing**: Strategic under-sampling to handle rating imbalance

### Machine Learning
- **Multiple Algorithms**: Logistic Regression, Decision Trees, Random Forest, SVM, Naive Bayes, KNN
- **Ensemble Methods**: AdaBoost, XGBoost, Voting and Stacking classifiers
- **Model Evaluation**: Comprehensive accuracy metrics and confusion matrices
- **Hyperparameter Tuning**: GridSearchCV for optimal parameter selection

### Deep Learning
- **Neural Network Architecture**: Embedding layers with GlobalAveragePooling and Dense layers
- **LSTM Implementation**: Bidirectional LSTM with dropout regularization
- **Text Processing**: Word indexing, sequence padding and one-hot encoding

### Visualization
- Rating distribution analysis across hotels
- Word frequency analysis and word clouds
- Model performance comparison and feature importance
- Interactive visualizations with Seaborn and Matplotlib

## Installation

### Prerequisites
- Python 3.7+
- Google Colab or Jupyter Notebook environment

### Setup
1. Mount Google Drive in Colab:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

2. Install required dependencies:
   ```bash
   pip install numpy pandas matplotlib seaborn nltk scikit-learn xgboost tensorflow wordcloud yellowbrick
   ```

3. Download NLTK resources:
   ```python
   import nltk
   nltk.download('stopwords')
   nltk.download('wordnet')
   ```

### Usage
- **Run the complete analysis**:
  ```python
  # Execute cells in the provided notebook sequentially
  # Sentiment_Analysis_of_Hilton_Hotel_London_Reviews.ipynb
  ```

- **Load pre-trained models** (optional):
  ```python
  import pickle
  model = pickle.load(open('/path/to/Logistic Regression.pkl', 'rb'))
  ```

## Project Structure
```
Hilton-Review-Analysis/
├── tripadvisor_scraped_hotel_reviews.csv      # Raw scraped data from Tripadvisor
├── hotel_reviews_structured.csv               # Cleaned and balanced dataset
├── Intermediate Data/                         # Preprocessed datasets
│   ├── X_train_data.csv                       # Training features
│   ├── X_test_data.csv                        # Testing features
│   ├── y_train_data.csv                       # Training labels
│   ├── y_test_data.csv                        # Testing labels
│   ├── The Hilton Hotel London Datay_test_ohe.csv    # One-hot encoded test labels
│   └── The Hilton Hotel London Datay_train_ohe.csv   # One-hot encoded training labels
├── Saved Model/                               # Trained models and artifacts
│   ├── Logistic Regression.pkl                # Best performing model (serialized)
│   ├── Neural_Network.keras                   # Artificial Neural Network model
│   ├── BidirectionalLSTM.keras                # Bidirectional LSTM model
│   └── word_index_dict.pkl                    # Vocabulary dictionary for text processing
├── all_models.csv                             # Model performance comparison results
├── Sentiment Analysis of Hilton Hotel London Reviews.ipynb  # Main analysis notebook
└── README.MD                                  # Project documentation
```
Some of the files where not shared due to github 25MBs storage. If you need the missing files, Please, Send an email via through the one provided below.
## Results

### Model Performance
| Model               | Validation Accuracy | Training Accuracy | Gap  |
| ------------------- | ------------------- | ----------------- | ---- |
| Logistic Regression | ~48%             | ~50%             | ~2% |
| Random Forest       | 45–46%              | ~99%             | 53%  |
| SVM                 | 44–45%              | \~99%             | 54%  |
| ANN                 | \~45–47%            | As for val.   | Low  |
| BiLSTM              | \~45–47%            | Similar to val.   | Low

### Business Insights
1. **Rating Distribution**: Most reviews clustered in 3-5 star range
2. **Hotel Variation**: Hilton London Metropole received more negative reviews compared to Gatwick Airport location
3. **Sentiment Indicators**: Words like "not", "good", "great", "clean", "poor" were strong predictors
4. **Error Pattern**: Misclassifications typically within ±1 star, rarely extreme (1 vs 5)

## Business Applications

### Hospitality Use Cases
- **Review Monitoring**: Automatically classify incoming reviews for quick response
- **Service Improvement**: Identify common complaints from low-rated reviews
- **Competitive Analysis**: Compare sentiment across different hotel locations
- **Trend Tracking**: Monitor sentiment changes over time after service improvements

### Operational Benefits
- Reduce manual review processing time
- Enable proactive customer service interventions
- Data-driven decision making for hotel management
- Benchmark performance against industry standards

## Contributing

Contributions are welcome! Please follow these steps:
1. Open an issue to discuss proposed changes or enhancements
2. Fork the repository and create a feature branch
3. Submit a pull request with:
   - Clear description of changes
   - Updated documentation
   - Test results if applicable

## License
MIT License. See LICENSE file for details.

## Contact

For inquiries, improvement or collaboration opportunities: [muthominicholus22@gmail.com]
