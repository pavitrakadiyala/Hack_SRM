"""
the body was lying on the ground, dismembered and blooding coming from everywhere.
The eyes had been gouged out of the sockets. The body mutilated all over. It was clear that the murdered was a psycho.


today's weather is beautiful. Birds chirping on the branches. Looking into my garden 
I see my dog playing around with butterflies. I put on my favourite music and start the day on an ebullient note.
"""


# importing necessary libraries and functions
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
#Pre 
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer,PorterStemmer
from nltk.corpus import stopwords
from string import punctuation
import re
import string
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer() 
df=pd.read_csv('NLPLR.csv')
def preprocess(sentence):
    sentence=str(sentence) #converting to string
    sentence = sentence.lower()#lower the sentence    
    cleanr = re.compile('<.*?>') #defining html tags
    cleantext = re.sub(cleanr, '', sentence)#remove html tags
    rem_url=re.sub(r'http\S+', '',cleantext)# remove url links
    rem_num = re.sub('[0-9]+', '', rem_url) #remove numbers
    tokens = word_tokenize(rem_num) #tokenization
    tokens = ([char.lower() for char in tokens  if char not in string.punctuation]) #removing punctuations 
    filtered_words = [w for w in tokens if len(w) > 2 if not w in stopwords.words('english')]#stopword removal
    stem_words=[stemmer.stem(w) for w in filtered_words]#stemming    
    lemma_words=[lemmatizer.lemmatize(w) for w in stem_words]#lemmatizing
    return " ".join(filtered_words)

df['cleanText']=df['Social Media Message'].apply(preprocess)#storing the preprocessed data
#feature extraction
#tfidf
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(strip_accents=None,
                       use_idf=True,
                       preprocessor=None,
                       norm='l2',
                       smooth_idf=True)
Y=df.Type
X=tfidf.fit_transform(df.cleanText)
from sklearn.model_selection import train_test_split
X_train, x_test, Y_train, y_test= train_test_split(X,Y, test_size = 0.3, random_state =0)

app = Flask(__name__) #Initialize the flask App
model = pickle.load(open('model.pkl', 'rb')) # loading the trained model


@app.route('/') # Homepage
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''

    
    # retrieving values from form
    #init_features = [float(x) for x in request.form.values()]
    #final_features = [np.array(init_features)]
    user_text = [str(x) for x in request.form.values()]

    #prediction = model.predict(final_features) # making prediction
    prediction = user_text
    predict_ndarray = model.predict(tfidf.transform(prediction))
    predict_list = predict_ndarray.tolist()
    #print(type(predict_list[0]))
    if(predict_list[0]==1):
    	predict_text = "Discretion Needed"
    else:
    	predict_text = "Text is safe for being viewed by kid"
    
    return render_template('index.html', prediction_text=predict_text) # rendering the predicted result

if __name__ == "__main__":
    app.run(debug=True)
