import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
import csv
from pyspark.sql import SparkSession
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.preprocessing import text
from keras import utils


def train(batch_size):
    data = pd.read_csv("cleantextlabels7.csv")
    train_size = int(len(data) * .8)
    train_posts = data['tweet'][:train_size]
    train_tags = data['label'][:train_size]
    test_posts = data['tweet'][train_size:]
    test_tags = data['label'][train_size:]


    vocab_size = 6000
    tokenize = text.Tokenizer(num_words=vocab_size)
    tokenize.fit_on_texts(train_posts)


    x_train = tokenize.texts_to_matrix(train_posts)
    x_test = tokenize.texts_to_matrix(test_posts)
    encoder = LabelEncoder()
    encoder.fit(train_tags)
    y_train = encoder.transform(train_tags)
    y_test = encoder.transform(test_tags)
    num_classes = np.max(y_train) + 1
    y_train = utils.to_categorical(y_train, num_classes)
    y_test = utils.to_categorical(y_test, num_classes)

    model = Sequential()
    model.add(Dense(512, input_shape=(vocab_size,)))
    model.add(Activation('relu'))

    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    history = model.fit(x_train, y_train,batch_size=batch_size,epochs=2,verbose=1,validation_split=0.1)

    score = model.evaluate(x_test, y_test,batch_size=batch_size, verbose=1)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])
    return model, tokenize, encoder

def predict(model, tokenize, encoder):
    testing = pd.read_csv("result.csv")
    test = tokenize.texts_to_matrix(testing['tweet'])

    text_labels = encoder.classes_
    # Start File
    with open('keywords.csv', 'w') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
        filewriter.writerow(['Keyword', 'Label'])

    for i in range(len(test)):
        prediction = model.predict(np.array([test[i]]))
        predicted_label = text_labels[np.argmax(prediction)]
        words = ["flu", "zika", "diarrhea", "headache", "measles", "ebola"]
        hasKeyword = False
        wordA = []
        for word in words:
            if word in testing['tweet'][i].lower():
                hasKeyword = True
                wordA.append(word)
        if hasKeyword:
            for word in wordA:
                with open('keywords.csv', 'a') as csvfile:
                    filewriter = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
                    filewriter.writerow([word, predicted_label])
    spark = SparkSession.builder.getOrCreate()
    df = spark.read.csv("keywords.csv", header=True,
                            inferSchema=True)
    df.createOrReplaceTempView("neural")
    sqlResult = spark.sql(
            "select Keyword, Label as label, count(*) as value from neural group by Keyword, label order by Keyword, label")
    sqlResult.show()
    print(sqlResult.toJSON().collect())

if __name__ == "__main__":
    print("Starting training algorithm")
    batch_size = 30
    model, tokenize, encoder = train(batch_size)
    predict(model,tokenize, encoder)
