import tkinter
import time
from tkinter import *
from tkinter import filedialog
import nltk; nltk.download('stopwords'); nltk.download("punkt")
import pandas as pd
from pprint import pprint
import pymorphy2
# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

# spacy for lemmatization

#Plotting tools
import pyLDAvis
import pyLDAvis.gensim
#Enable logging for gensim - optional
import logging
import warnings
# NLTK Stop words
from nltk.corpus import stopwords

def topic_root(name_of_file):
    #Enable logging for gensim - optional
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',level= logging.ERROR)
    warnings.filterwarnings("ignore",category=DeprecationWarning)
    # NLTK Stop words
    stop_words = stopwords.words('russian')
    stop_words.append("иль")
    stop_words.append("тебе")
    print(stop_words)

    # Import dataset
    def return_corpus(file_name):
        f = open(file_name, "r")

        corpus_array = []
        collecting_string = ""

        for i in f:
            i = i.replace('\n', ' ')
            if i.__contains__('!!!'):
                corpus_array.append(collecting_string)
                collecting_string = ""
            else:
                collecting_string += i
        return corpus_array

    df = pd.DataFrame(return_corpus(name_of_file))
    df.columns = ['text']
    print(df)

    data = df.text.values.tolist()

    pprint(data[:1])

    def sent_to_words(sentences):
        for sentence in sentences:
            yield(gensim.utils.simple_preprocess(str(sentence),deacc=True)) # deacc = True removes punctuations


    data_words = list(sent_to_words(data))

    print(data_words[:1])

    # Build the bigram and trigram models

    bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
    trigram = gensim.models.Phrases(bigram[data_words],threshold=100)

    #Faster way to get a sentence clubbed as a trigram/bigram
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)

    # Set trigram example
    print(trigram_mod[bigram_mod[data_words[0]]])

    #Define functions for stopwords, bigrams, trigrams and lemmatization
    def remove_stopwords(texts):
        return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

    def make_bigrams(texts):
        return [bigram_mod[doc] for doc in texts]

    def make_trigrams(texts):
        return [trigram_mod[bigram_mod[doc]] for doc in texts]

    #Remove Stop Words
    data_words_nostops = remove_stopwords(data_words)
    data_words_bigrams = make_bigrams(data_words_nostops)

    print(data_words_bigrams)


    pymorph = pymorphy2.MorphAnalyzer()
    data_words_bigrams2 = []
    for k in range(len(data_words_bigrams)):
        for m in range(len(data_words_bigrams[k])):
            data_words_bigrams[k][m] = pymorph.parse(data_words_bigrams[k][m])[0].normal_form

    data_lemmatized = data_words_bigrams

    # Create Dictionary
    id2word = corpora.Dictionary(data_lemmatized)

    # Create Corpus
    texts = data_lemmatized

    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in texts]

    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                id2word=id2word,
                                                num_topics=20,
                                                random_state=100,
                                                update_every=1,
                                                chunksize=100,
                                                passes=10,
                                                alpha='auto',
                                                per_word_topics=True)
    pprint(lda_model.print_topics())
    #doc_lda = lda_model[corpus]

    # Compute Perplexity
    #print('\nPerplexity: ',lda_model.log_perplexity(corpus)) # a measure of how good the model is. lower the better

    # Compute Coherence Score
    coherence_model_lda = CoherenceModel(model=lda_model,texts=data_lemmatized,dictionary=id2word,coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    #print('\nCoherence Score: ', coherence_lda)

    vis = pyLDAvis.gensim.prepare(lda_model,corpus,id2word)
    pyLDAvis.save_html(vis,"cartoon.html")


def open_file():
    global filename
    global label
    global root
    filename = filedialog.askopenfilename()

    label.config(text="Загрузка..")
    label.place(x=220, y=200)
    label.update()
    topic_root(filename)

    label.config(text="Корпус обработан, ЛРД модель обучена, \n html-файл создан!")
    label.place(x=90, y=200)
    label.update()

    root.destroy()

def main():
    global filename
    global label
    global root
    root = tkinter.Tk()
    root.geometry("500x400")

    button = tkinter.Button(root,text="Выберите корпус",font=("Tahoma",14),command=open_file)
    button.place(x=175,y=150)
    label = tkinter.Label(root,text="",font=("Tahoma",14))

    label.place(x=90, y=200)
    root.mainloop()


if __name__ == '__main__':
    start = time.time()
    main()
    print((time.time()-start)/60)