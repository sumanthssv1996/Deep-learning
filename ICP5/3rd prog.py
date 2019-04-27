import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.wrappers.scikit_learn import KerasClassifier
import re

from sklearn.preprocessing import LabelEncoder

data = pd.read_csv('spam.csv',encoding='latin-1')
# Keeping only the neccessary columns
data = data[['v2','v1']]



data['v2'] = data['v2'].apply(lambda x: x.lower())
data['v2'] = data['v2'].apply((lambda x: re.sub('[^a-zA-z0-9\s]', '', x)))

print(data[data['v1'] == 'ham'].size)
print(data[data['v1'] == 'spam'].size)



max_fatures = 2000
tokenizer = Tokenizer(num_words=max_fatures, split=' ')
tokenizer.fit_on_texts(data['v2'].values)
X = tokenizer.texts_to_sequences(data['v2'].values)
print(X)
X = pad_sequences(X)
print(X)
embed_dim = 128
lstm_out = 196
def createmodel():
    model = Sequential()
    model.add(Embedding(max_fatures, embed_dim,input_length = X.shape[1]))
    model.add(SpatialDropout1D(0.4))
    model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(2,activation='softmax'))
    model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
    return model
# print(model.summary())

labelencoder = LabelEncoder()
integer_encoded = labelencoder.fit_transform(data['v1'])
y = to_categorical(integer_encoded)
X_train, X_test, Y_train, Y_test = train_test_split(X,y, test_size = 0.33, random_state = 42)
print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)






'''model = KerasClassifier(build_fn=createmodel, verbose=0)
batch_size= [20, 30, 40]
epochs = [3, 4, 5]
param_grid= dict(batch_size=batch_size, epochs=epochs)
from sklearn.model_selection import GridSearchCV
grid  = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
grid_result= grid.fit(X_train, Y_train)


# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))'''

model = createmodel()
model.fit(X_train, Y_train, epochs = 4, batch_size=40, verbose = 2)
twt = ['Go until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there got amore wat...']


score,acc = model.evaluate(X_test,Y_test,verbose=2,batch_size=40)
print(score)
print(acc)