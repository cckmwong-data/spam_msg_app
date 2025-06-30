import streamlit as st
import joblib
import nltk
import os
import string
import re
import streamlit.components.v1 as components

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Ensure NLTK data is stored and accessed from a local folder (for Streamlit Cloud)
nltk_data_dir = os.path.join(os.path.dirname(__file__), 'nltk_data')
os.makedirs(nltk_data_dir, exist_ok=True)
nltk.data.path.append(nltk_data_dir)

# Download NLTK assets to that folder
#nltk.download('punkt', download_dir=nltk_data_dir, quiet=True)
nltk.download('stopwords', download_dir=nltk_data_dir, quiet=True)
nltk.download('wordnet', download_dir=nltk_data_dir, quiet=True)

# Load model and vectorizer
model = joblib.load('svm_spam_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Preprocessing functions
#Function for converting the message into tokens or individual words
def text_processing(text):
    textList = []
    #text = text.lower() #change the text to lower case
    text = re.sub(r"@\S+|https?:\S+|http?:\S+|\S+@\S+", ' ', str(text).lower())
    text = re.sub(r"[^a-z\s]", ' ', text)
    text = text.split() #divide the whole message into bite-sized words

    for i in text:
        #check if each of the token is either alphabet or numbers. If yes, store the useful bits into the list
        if i.isalnum():
            textList.append(i)
    return textList


#Function to remove stopwords, words that are commonly used in English 
def remove_stopwords(text):
    textList = [] #create a new list
    for i in text:
        #Check if the token is not a stop word AND is not punctuation
        if i not in nltk.corpus.stopwords.words('english') and i not in string.punctuation:
            textList.append(i) #append individual tokens into a list
    return textList

#Function to reduce words to their root form
def lemmatize_text(text):
    lemmatizer = WordNetLemmatizer()
    textList = []
    for i in text:
        textList.append(lemmatizer.lemmatize(i))
    return textList

#A function for predicting whether the input message is SPAM or NOT SPAM
def predict_message(model, vectorizer, message):
    #Preprocess the input message
    message = text_processing(message) #change the message into tokens
    message = remove_stopwords(message) #remove stopwords
    message = lemmatize_text(message) #transform words into their base-words
    
    #Convert to one whole string
    message = ' '.join(message)
    
    #Transform the message using the same TF-IDF vectorizer as that of training the SVM model
    message_transformed = vectorizer.transform([message])
    
    #Predict using trained model
    prediction = model.predict(message_transformed)[0]
    
    #Decode the label to SPAM or NOT SPAM
    result = "SPAM" if prediction == 1 else "NOT SPAM"
    return result


def speak_directly_in_browser(text):
    escaped_text = text.replace('"', "'")
    components.html(f"""
        <button onclick="window.speechSynthesis.speak(new SpeechSynthesisUtterance('{escaped_text}'))"
                style="font-size:18px; padding:10px 20px; border-radius:8px; margin-top: 10px;">
            🔊 Speak Result
        </button>
    """, height=80)

# Define a clear function
def clear_text():
    st.session_state["user_input"] = ""

# Custom title with larger font size
st.markdown("<h1 style='font-size: 54px;'>Spam Checker</h1>", unsafe_allow_html=True)

# Custom subtitle
st.markdown("<p style='font-size: 24px;'>This tool is for classifying if a text message is <strong>SPAM</strong> or <strong>NOT SPAM</strong>.</p>", unsafe_allow_html=True)

col1, col2 = st.columns([8, 2])

with col1:  
    # Custom text area label
    st.markdown("<label style='font-size: 24px;'>Enter your message:</label>", unsafe_allow_html=True)
with col2:
    # Clear Text button — calls the function BEFORE text_area rerenders
    st.button("🧹 Clear Text", on_click=clear_text)

# Make sure it's initialized
if "user_input" not in st.session_state:
    st.session_state["user_input"] = ""
    st.session_state["result_to_speak"] = f"Please enter a message and press the Check button."

# Text input area
user_input = st.text_area("", key="user_input")

# After classification result is determined
if st.button("🔍Check", use_container_width=True):
    if user_input.strip():
        result = predict_message(model, vectorizer, user_input)

        # Display result
        if result == "NOT SPAM":
            st.markdown(
                "<div style='background-color:#007acc; color:white; padding:10px; border-radius:5px;'>"
                "<strong>✅ Prediction: NOT SPAM</strong>"
                "</div>", unsafe_allow_html=True
            )
        else:
            st.markdown(
                "<div style='background-color:#f39c12; color:white; padding:10px; border-radius:5px;'>"
                "<strong>❗ Prediction: SPAM</strong>"
                "</div>", unsafe_allow_html=True
            )

        # ✅ Save result in session state so it can be spoken later
        clean_input = str(user_input).replace("\n", "").replace("\r", "")
        st.session_state["result_to_speak"] = f"The message is {clean_input}. It is classified as {result}"
    else:
        st.session_state["result_to_speak"] = f"lease enter a message and press the Check button."
        st.warning("Please enter a message to classify.")

speak_directly_in_browser(st.session_state["result_to_speak"])

st.markdown(
    'Suspect a scam text message? <a href="https://www.ncsc.gov.uk/collection/phishing-scams/report-scam-text-message" target="_blank">Learn what to do here</a>',
    unsafe_allow_html=True
)
