# Veo3-sentiment-analysis

## Overview
This project implements a robust sentiment analysis pipeline for Indonesian YouTube comments about VO3. The workflow includes data cleaning, augmentation, feature engineering, model training, and evaluation, with all major steps saved to CSV for traceability.

## Analysis Pipeline

### 1. Data Loading
- The labeled dataset (`komentar_vo3 _labeling.csv`) is loaded, using the columns: `comment`, `likes`, and `sentimen`.

### 2. Preprocessing
- Comments are cleaned by:
  - Lowercasing (case folding)
  - Removing URLs, mentions, hashtags, numbers, and punctuation
  - Removing Indonesian stopwords
  - Stemming using Sastrawi
- The cleaned data is saved to `preprocessed_comments.csv`.

### 3. Data Augmentation
- After preprocessing, the dataset is expanded to 10,000 samples using:
  - Random word swaps
  - Synonym replacement (with a small Indonesian synonym dictionary)
  - Random deletion of non-stopword tokens
- Augmented samples are appended to the cleaned data and saved to `preprocessed_comments.csv`.

### 4. Feature Engineering
- TF-IDF vectorization is applied to the cleaned text with:
  - Unigram and bigram support (`ngram_range=(1,2)`)
  - `min_df=5`, `max_df=0.85`, and up to 5000 features
- The resulting TF-IDF matrix is saved to `tfidf_matrix.csv`.
- The `likes` feature is combined with the TF-IDF features and saved to `tfidf_plus_likes.csv`.

### 5. Model Training & Hyperparameter Tuning
- An SVM classifier (One-vs-One) is trained using the combined features.
- Hyperparameters (`C`, `kernel`, `gamma`) are tuned using `GridSearchCV` with cross-validation.
- The best model and parameters are selected.

### 6. Evaluation
- The model is evaluated using cross-validation.
- Predictions and evaluation metrics (classification report, confusion matrix) are saved to `sentiment_predictions.csv`.

## Example: Head of Cleaned & Augmented Data
Below is a sample of the cleaned and augmented data (from `preprocessed_comments.csv`):

| comment                        | likes | sentimen | clean_text                |
|--------------------------------|-------|----------|--------------------------|
| Video ini sangat bagus         | 12    | 1        | video ini sangat bagus    |
| Kurang menarik dan lambat      | 3     | 0        | kurang menarik dan lambat |
| ...                            | ...   | ...      | ...                      |

## Model Evaluation Results
A sample classification report (accuracy, precision, recall, f1-score):

```
              precision    recall  f1-score   support

           0       0.85      0.83      0.84      2000
           1       0.87      0.89      0.88      3000
           2       0.80      0.81      0.80      5000

    accuracy                           0.84     10000
   macro avg       0.84      0.84      0.84     10000
weighted avg       0.84      0.84      0.84     10000
```

## Confusion Matrix Visualization
The confusion matrix is also visualized for better interpretability. Example code to generate the plot:

```python
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# y_true and y_pred are the true and predicted labels
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negatif','Netral','Positif'], yticklabels=['Negatif','Netral','Positif'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
```

Sample output:

![Confusion Matrix Example](confusion_matrix_example.png)

## Output Files
- `preprocessed_comments.csv`: Cleaned and augmented data
- `tfidf_matrix.csv`: TF-IDF features
- `tfidf_plus_likes.csv`: Combined features (TF-IDF + likes)
- `sentiment_predictions.csv`: Model predictions and evaluation

## Reproducibility
All major steps save their outputs to CSV, enabling traceability and reuse of intermediate results.

## Requirements
- Python 3.x
- pandas, numpy, scikit-learn, Sastrawi, nltk
- matplotlib, seaborn (for visualization)

## Usage
Run the notebook `kode.ipynb` step by step to reproduce the analysis and results.

## Team
- [Arizq](https://github.com/naufalarizq) - [rizqullohnaufal@apps.ipb.ac.id](mailto :rizqullohnaufal@apps.ipb.ac.id)