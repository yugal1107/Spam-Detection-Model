from sklearn.externals import joblib
import re
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Load the trained model from the file
classifier_2 = joblib.load('spam_classifier_model.pkl')

# Load the LabelEncoder object from the file
le = joblib.load('label_encoder.pkl')

cv= joblib.load('countvectorizer.pkl')
y="lets meet tomorrow "
sp= re.sub('[^a-zA-Z]',' ',y)
sp= sp.lower()
sp=sp.split()
all_stopwords= stopwords.words('english')
ps=PorterStemmer()
sp=[ps.stem(word) for word in sp if not word in all_stopwords]
sp=' '.join(sp)
new= [sp]
new= cv.transform(new).toarray()
y_pred=classifier_2.predict(new)
y_pred= le.inverse_transform(y_pred)
print(y_pred)