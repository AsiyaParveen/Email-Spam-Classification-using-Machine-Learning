# Email-Spam-Classification-using-Machine-Learning

## Project Overview
This project is a **Machine Learning–based Email Spam Classifier** that predicts whether an email message is **Spam** or **Not Spam (Ham)**.  
It uses **Natural Language Processing (NLP)** techniques for text preprocessing and a **Naive Bayes classifier** for prediction.  
A simple and interactive **Streamlit web application** is provided for real-time testing.

---

## 🎯Objectives
- Preprocess email text data using NLP techniques
- Extract important features using **TF-IDF**
- Train a **Naive Bayes classification model**
- Classify emails as Spam or Not Spam
- Build a user-friendly **Streamlit UI**

---

##  Dataset
- Source: **Kaggle – SMS Spam Collection Dataset**
- Labels:
  - `1` → Spam  
  - `0` → Not Spam  

 Dataset Link:  
https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset

---

## Technologies & Libraries Used
- Python  
- Pandas & NumPy  
- NLTK (Natural Language Processing)  
- Scikit-learn  
- Streamlit  
- Pickle  

---

##  Project Workflow
1. Load and clean the dataset  
2. Text preprocessing (lowercasing, stopword removal, stemming)  
3. Feature extraction using **TF-IDF Vectorizer**  
4. Model training using **Naive Bayes**  
5. Save trained model and vectorizer using Pickle  
6. Build Streamlit-based UI for predictions  

---

## Project Structure
```

Email_Spam_Classifier/
│
├── app.py              # Streamlit UI code
├── train_model.py      # Model training script
├── spam_model.pkl      # Trained Naive Bayes model
├── tfidf.pkl           # TF-IDF vectorizer
├── dataset/
│   └── spam.csv        # Dataset file
├── requirements.txt    # Required libraries
└── README.md           # Project documentation

````

---

##  How to Run the Project

###  Activate Conda Environment
```bash
conda activate Ai_ml_engineer
````

### 2️Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️ Run Streamlit Application

```bash
streamlit run app.py
```

Open your browser and go to:
👉 [http://localhost:8501](http://localhost:8501)

---

## Sample Test Emails

###  Spam Example

```
Congratulations! You have won a free prize. Click the link to claim now.
```

### Not Spam Example

```
Please find the attached report for today’s meeting.
```

---

## ✅ Output

* 🚨 SPAM
* ✅ NOT SPAM

---

##  Conclusion

This project demonstrates how **Machine Learning and NLP** can be applied to detect spam emails effectively.
The system provides accurate predictions along with a simple web interface for easy use.

---

## Author

**Asiya Parveen**
AI / ML Engineer

```

