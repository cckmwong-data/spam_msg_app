import streamlit as st
import joblib
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK assets (only the first time)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load model and vectorizer
model = joblib.load('svm_spam_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Preprocessing functions
#Function for converting the message into tokens or individual words
def text_processing(text):
    text = text.lower() #change the text to lower case
    
    textList = []
    text = nltk.word_tokenize(text) #divide the whole message into bite-sized tokens or individual words
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

# Streamlit UI
st.title("📩 Spam Message Classifier")
st.write("This tool uses machine learning to classify whether a message is SPAM or NOT SPAM.")

user_input = st.text_area("Enter your message:")

if st.button("Classify"):
    if user_input.strip():
        #result = predict_message(user_input)
        result = predict_message(model, vectorizer, user_input)
        st.success(f"✅ Prediction: {result}")
    else:
        st.warning("Please enter a message to classify.")
