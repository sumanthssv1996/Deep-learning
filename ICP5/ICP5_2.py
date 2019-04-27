import numpy as np
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
model = load_model('model.h5')
text1 = ('A lot of good things are happening. We are respected again throughout the world, and thats a great thing.@realDonaldTrump')
tokenizer = Tokenizer(num_words=max_fatures,  split=' ')
tokenizer.fit_on_texts(text1)
X =tokenizer.texts_to_sequences(text1)
X =pad_sequences(X,maxlen=28)
sent=model.predict(X, batch_size=1, verbose = 2)[0]
print(np.argmax(sent))