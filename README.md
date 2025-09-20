# News Classification with N-grams  

## 📌 Project Overview  
This project applies **Natural Language Processing (NLP)** techniques to classify news articles into four categories:  
-  World  
-  Sports  
-  Business  
-  Science/Technology  

The approach focuses on using **N-gram features (Unigrams, Bigrams, Trigrams)** with traditional **machine learning classifiers** to evaluate the impact of different text representations and preprocessing techniques on classification performance.  

---

## 🎯 Research Questions  
1. What is the impact of different N-gram ranges on classification accuracy?  
2. Which classifier (KNN, Decision Tree, or Random Forest) performs best with N-gram features?  
3. How does text preprocessing (tokenization, stemming, stopword removal) affect results?  
4. What are the strengths and limitations of N-gram approaches for text classification?  

---

## 🛠 Methodology  

### 1. Dataset  
- **AG News Classification Dataset** (by Xiang Zhang, published on Kaggle by Aman Anand).  
- 4 categories, each with **3000 training samples** (12,000 total) and **1900 testing samples** (7,600 total).  

### 2. Preprocessing  
- Lowercasing, punctuation & stopword removal  
- Tokenization with NLTK  
- Snowball stemming  
- Label encoding of classes  

### 3. N-gram Feature Extraction  
- Unigrams `(1,1)`  
- Bigrams `(2,2)`  
- Trigrams `(3,3)`  
- Count Vectorizer used to build vocabulary  

### 4. Models  
- **Random Forest (RF)**  
- **K-Nearest Neighbors (KNN)**  
- **Decision Tree (DT)**  
- Hyperparameters tuned with **HalvingGridSearchCV**  

### 5. Evaluation Metrics  
- Accuracy  
- Precision  
- Recall  
- F1 Score  

### 6. Interface  
- A **Gradio-based web application** allows users to input news text and receive predictions with confidence scores.  

---

## 📊 Results  

- **Unigrams** consistently outperformed bigrams and trigrams.  
- **Random Forest (with Unigrams)** achieved the highest performance:  
  - **F1 Score: 0.827**  
- Text preprocessing improved accuracy significantly (raw word count reduced by ~34%).  

---

##  Key Insights  
- **Simplicity works:** Unigrams capture enough discriminative power for news classification.  
- **Random Forest is the best performer** among traditional classifiers for this dataset.  
- **Preprocessing is crucial** to reduce noise and improve feature quality.  
- N-grams are interpretable and efficient, but they suffer from sparsity and lack deep semantic understanding.  

---

##  Future Improvements  
- Use **TF-IDF weighting** instead of raw counts.  
- Combine **Unigrams + Bigrams** for richer context.  
- Explore **character-level N-grams** for robustness.  
- Incorporate **word embeddings (Word2Vec, GloVe, FastText)**.  
- Experiment with **deep learning (CNN, RNN, Transformers like BERT)**.  
- Build **hybrid models** mixing N-gram features with embeddings.  

---

##  Installation & Usage  

### Clone Repository  
```bash
git clone https://github.com/yourusername/news-classification-ngrams.git
cd news-classification-ngrams
``` 

---

## 📚 References  
- [AG News Dataset – Kaggle](https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset)  
- [Towards Data Science – F1 Score Explained](https://towardsdatascience.com/micro-macro-weighted-averages-of-f1-score-clearly-explained-b603420b292f)  
