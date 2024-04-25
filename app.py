from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string
import pickle
import nltk
nltk.download('stopwords')
vec = pickle.load(open('tfidf_vectorizer.sav', 'rb'))


stopwords_list = set(stopwords.words('english'))
negation_words = ['not', 'never', 'nor', 'no']
custom_stopwords = stopwords_list - set(negation_words)
custom_stopwords_list = list(custom_stopwords)
port_stem = PorterStemmer()

def stemming(content):
    content = content.translate(str.maketrans('', '', string.punctuation)) 
    content = re.sub(r'@[\w]+', '', content)
    content = re.sub(r'http\S+', '', content)
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)  
    stemmed_content = re.sub(r'\bnot\s+(good|bad)\b', r'not_\1', content) 
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in custom_stopwords_list]
    stemmed_content = ' '.join(stemmed_content)
    
    return stemmed_content

loaded_model = pickle.load(open('sentiment_model.sav', 'rb'))

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('main.html')

@app.route('/analyze', methods=["POST"])
# def take_action():
#     text = request.form['text']
    
       
#     stemmed_text = stemming(text)
#     custom_test = vec.transform([stemmed_text])
#     prediction = loaded_model.predict(custom_test)
#     if prediction == 1:
#         result = "✅😍: Your given text have Positive sentiment"
#     else:
#         result = "❌😒: Your given text have Negative sentiment"
#     return render_template('result.html', result=result)
def take_action():
    try:
        text = request.form['text']
        if text.strip() == "":
            result = "Seem like u haven't enter the text 😅"
        elif any(char.isdigit() for char in text) and any(char.isalpha() for char in text):
            result = "Oops! Your text contains both words and numbers. Please enter text without numbers or remove the letters."
        elif text.isdigit() or (text[0] in ['+', '-'] and text[1:].isdigit()):
            number = int(text)
            if number > 0:
                result = "Oops! Looks like your text contains a positive integer. Please enter some text.🤣"
            elif number < 0:
                result = "Oops! Looks like your text contains a negative integer. Please enter some text.🤣"
            else:
                result = "Oops! Looks like your text contains only zero. Please enter some text.🤣"
        else:
            stemmed_text = stemming(text)
            custom_test = vec.transform([stemmed_text])
            prediction = loaded_model.predict(custom_test)
            if prediction == 1:
                result = "😅😇: Looks like your text is wearing a big smiley face!"
            else:
                result = "🥲🥺: Uh-oh! Your text seems to have caught a case of the grumpy bugs!"
        
        return render_template('result.html', result=result)
    except Exception as e:
        # Log the error or handle it appropriately
        error_message = str(e)
        return render_template('error.html', error_message=error_message)
@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

if __name__ == "__main__":
    app.run(debug=True)
