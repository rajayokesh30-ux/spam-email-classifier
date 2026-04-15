# =========================
# STEP 1: IMPORT LIBRARIES
# =========================
import pandas as pd
import string
import nltk

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

nltk.download('stopwords')
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

# =========================
# STEP 2: LOAD DATASETS
# =========================
print("Loading datasets...")

# Dataset 1
data1 = pd.read_csv("spam1.csv", encoding='latin-1')
data1 = data1[['v1', 'v2']]
data1.columns = ['label', 'message']

# Dataset 2
data2 = pd.read_csv("spam2.csv", encoding='latin-1')
data2 = data2[['v1', 'v2']]
data2.columns = ['label', 'message']

# Combine datasets
data = pd.concat([data1, data2], ignore_index=True)

print("Total rows before cleaning:", len(data))

# =========================
# STEP 3: CLEAN DATA
# =========================
# =========================
# STEP 3: CLEAN DATA (FIXED)
# =========================

# clean label column properly
# =========================
# STEP 3: CLEAN DATA (FIXED)
# =========================

# clean label properly
data['label'] = data['label'].astype(str).str.lower().str.strip()

# convert all possible formats
data['label'] = data['label'].replace({
    'ham': 0,
    'not spam': 0,
    '0': 0,
    'spam': 1,
    '1': 1
})

# keep only valid labels
data = data[data['label'].isin([0,1])]
data['label'] = data['label'].astype(int)

# remove duplicates
data = data.drop_duplicates()



# debug check (IMPORTANT)
print("\nUnique labels:", data['label'].unique())
print("\nLabel counts:\n", data['label'].value_counts())

print("Total rows after cleaning:", len(data))
print("Label distribution:\n", data['label'].value_counts())

# =========================
# STEP 4: PREPROCESS TEXT
# =========================
def preprocess_text(text):
    text = str(text).lower()
    
    # remove punctuation
    text = ''.join([c for c in text if c not in string.punctuation])
    
    # remove numbers
    text = ''.join([c for c in text if not c.isdigit()])
    
    # remove stopwords
    words = text.split()
    words = [word for word in words if word not in stop_words]
    
    return ' '.join(words)

print("Preprocessing...")
data['message'] = data['message'].apply(preprocess_text)
print("Preprocessing done ✅")

# =========================
# STEP 5: TRAIN TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    data['message'], data['label'], test_size=0.2, random_state=42
)

# =========================
# STEP 6: TF-IDF VECTORIZER
# =========================
vectorizer = TfidfVectorizer(
    ngram_range=(1,2),
    max_df=0.9,
    min_df=2
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# =========================
# STEP 7: MODEL TRAINING
# =========================
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

print("Model trained ✅")

# =========================
# STEP 8: EVALUATION
# =========================
y_pred = model.predict(X_test_vec)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nReport:\n", classification_report(y_test, y_pred))

# =========================
# STEP 9: PREDICTION FUNCTION
# =========================
def predict_spam(text):
    text = preprocess_text(text)
    vector = vectorizer.transform([text])
    
    prob = model.predict_proba(vector)[0]
    result = model.predict(vector)[0]
    
    if result == 1:
        return f"SPAM 🚨 ({prob[1]*100:.2f}%)"
    else:
        return f"NOT SPAM ✅ ({prob[0]*100:.2f}%)"

# =========================
# STEP 10: LIVE TESTING
# =========================
print("\n===== LIVE TESTING MODE =====")

while True:
    msg = input("\nEnter a message (type 'exit' to quit): ")
    
    if msg.lower() == 'exit':
        print("Exiting... 👋")
        break
    
    print("Result:", predict_spam(msg))