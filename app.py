import nltk
import streamlit as st
import pickle
import re
from nltk.corpus import stopwords

nltk.download('stopwords')
import string
from nltk.stem import PorterStemmer

tfidf = pickle.load(open("./vect_1.pickle", "rb"))
model = pickle.load(open("./model.pickle", "rb"))

st.title('Spam/Ham classifier')

input_sms = st.text_area('Enter the sms:')

if st.button('Predict'):

    # preprocess

    def processing(text):
        text = text.lower()
        text = nltk.word_tokenize(text)
        filtered_words = [re.sub('[^A-Za-z0-9]+', '', word) for word in text if
                          word not in stopwords.words('english') and word not in string.punctuation]
        ps = PorterStemmer()
        words = [ps.stem(w) for w in filtered_words]
        return ' '.join(words)


    transformed_text = processing(input_sms)

    # Vectorize
    vect_text = tfidf.transform([transformed_text])

    # Predict
    pred = model.predict(vect_text)[0]
    print(pred)

    # Result
    if pred == 0:
        st.header('The message you entered seems to be: NOT SPAM')
    else:
        st.header('The message you entered seems to be: SPAM')
