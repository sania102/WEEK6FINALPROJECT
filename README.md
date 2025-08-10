
Hi,
This is my final project for the AI/ML internship – a **Sentiment Analysis** model using Twitter data, enhanced with **data augmentation** techniques to improve performance.


## ** Project Title**

**Create synthetic credit card transactions for fraud detection**

---

## ** Objective**

The main goal of this project is to classify tweets as **Positive** or **Negative** sentiments and improve the model's accuracy by generating additional synthetic training data using **EDA (Easy Data Augmentation)** techniques like synonym replacement, random swap, insertion, and deletion.

---

## ** Dataset**

* **Name:** `training.1600000.processed.noemoticon.csv`
* **Source:** Twitter Sentiment Analysis Dataset
* **Size:** 1.6 million rows (for speed, I used a 20k subset during training)
* **Columns Used:** `text` (tweet content) and `target` (sentiment label)

---

## ** Steps I Followed**

1. **Loaded and cleaned** the dataset

   * Removed punctuation and stopwords
   * Applied **lemmatization** for better text normalization

2. **Converted text to features** using **TF-IDF vectorization**

   * Used **bigrams** and **trigrams** for better context capture

3. **Trained baseline models**

   * Logistic Regression
   * Random Forest Classifier

4. **Applied data augmentation**

   * Generated synthetic tweets for the minority class using synonym replacement, random swaps, insertions, and deletions

5. **Re-trained models on augmented dataset**

   * Compared accuracy and F1-score before and after augmentation

6. **Visualized** the distribution of real vs synthetic tweets using **t-SNE plots**

---

## ** Models & Files Generated**

* `logistic_augmented.pkl` – Trained Logistic Regression model (after augmentation)
* `rf_augmented.pkl` – Trained Random Forest model (after augmentation)
* `tfidf_vectorizer_augmented.pkl` – TF-IDF vectorizer used for features
* `augmented_train.csv` – Final training dataset after augmentation
* `generated_synthetic_samples.csv` – Only the synthetic data generated

---

## ** Results**

* Both models showed **improved accuracy and F1-score** after augmentation
* Logistic Regression performed slightly better than Random Forest on this dataset

---

## ** How to Run the Code**

1. Make sure you have **Python 3.x** installed.
2. Install required packages:

   ```
   pip install nltk scikit-learn pandas numpy seaborn matplotlib
   ```
3. Place the dataset (`training.1600000.processed.noemoticon.csv`) in the same folder as the notebook.
4. Open `FINALPROJECT.ipynb` in **Jupyter Notebook** or **Google Colab**.
5. Run all cells in order.
6. Check the results, plots, and saved models in the output folder.

---

## ** Future Improvements**

* Try **deep learning models** like LSTMs or Transformers (BERT)
* Expand augmentation to include **back-translation**
* Use the full 1.6M dataset on a higher computing setup for better performance

---


