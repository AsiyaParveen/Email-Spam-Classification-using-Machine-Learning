import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download stopwords (only first time)
nltk.download('stopwords')

# Load model and vectorizer
# Note: Ensure these files are in the same directory
try:
    model = pickle.load(open("spam_model.pkl", "rb"))
    vectorizer = pickle.load(open("tfidf.pkl", "rb"))
except FileNotFoundError:
    st.error("Model files not found. Please ensure 'spam_model.pkl' and 'tfidf.pkl' exist.")

# Text preprocessing function
ps = PorterStemmer()
stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower().split()
    text = [ps.stem(word) for word in text if word not in stop_words]
    return " ".join(text)

# --- UI SETTINGS ---
st.set_page_config(page_title="Spam Shield AI", page_icon="🛡️", layout="centered")

# Custom CSS for a cleaner look
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stTextArea textarea {
        border-radius: 10px;
    }
    .stButton button {
        width: 100%;
        border-radius: 20px;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# --- HEADER ---
st.title("🛡️ Email Spam Detector")
st.markdown("Use our AI-powered tool to identify potential phishing or spam messages.")
st.divider()

# --- INPUT SECTION ---
email_text = st.text_area("Paste the email content below:", height=200, placeholder="e.g., 'Congratulations! You've won a $1,000 gift card...'")

col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    predict_button = st.button("Analyze Email")

# --- PREDICTION LOGIC ---
if predict_button:
    if not email_text.strip():
        st.warning("⚠️ Please provide some text to analyze.")
    else:
        with st.spinner('Analyzing patterns...'):
            # Preprocessing
            cleaned_email = clean_text(email_text)
            vectorized_email = vectorizer.transform([cleaned_email])
            
            # Prediction
            prediction = model.predict(vectorized_email)
            # Optional: Get probability if your model supports it
            # prob = model.predict_proba(vectorized_email)[0][1] 

        # --- RESULTS ---
        st.subheader("Analysis Result:")
        if prediction[0] == 1:
            st.error("### 🚨 High Risk: This looks like SPAM")
            st.info("Common spam indicators found: suspicious keywords, urgent tone, or unusual formatting.")
        else:
            st.success("### ✅ Low Risk: This appears to be HAM (Safe)")
            st.balloons()

# --- FOOTER ---
st.divider()
st.caption("Powered by Scikit-Learn and Streamlit • Privacy First: Text is not stored.")