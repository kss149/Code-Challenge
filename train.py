import os, json, sys
import gensim

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

#Code is taken from https://radimrehurek.com/gensim/tut1.html

#---------------------------------------------------Load Corpus from JSON files---------------------------------------------------#
all_json_files = []
documentBodies = []
training_data_size = int(sys.argv[1])-1

# this finds the json files
for dirname, dirnames, filenames in os.walk('.'):
    for filename in filenames:
    	if filename.lower().endswith(".json"):
    		all_json_files.append(filename)

# this takes ONLY the body of the articles and makes a list of them(a list of strings)
for js in all_json_files:
    # while(training_data_size!=0):
    with open(os.path.join('json/', js)) as json_file:
        json_text = json.load(json_file)
        documentBodies.append(json_text['m_szDocBody'])
        # training_data_size-=1
documentBodies = documentBodies[21:]

#------------------------------------------------Preprocess Corpus of Document Bodies----------------------------------------------#
# remove common words and tokenize
stoplist = set('for a of the and to in'.split())
training_data = [[word for word in documentBody.lower().split() if word not in stoplist]
         for documentBody in documentBodies]

# remove words that appear only once
from collections import defaultdict
frequency = defaultdict(int)
for training_vector in training_data:
    for token in training_vector:
        frequency[token] += 1
training_data = [[token for token in training_vector if frequency[token] > 1]
         for training_vector in training_data]

dictionary = gensim.corpora.Dictionary(training_data)
dictionary.save('corpus.dict')

corpus = [dictionary.doc2bow(training_vector) for training_vector in training_data]
gensim.corpora.MmCorpus.serialize('corpus.mm', corpus)  # store to disk, for later use

#------------------------------------------------------Train and Save model------------------------------------------------------#
lda = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=100, update_every=1, chunksize=10000, passes=10)
lda.save("lda")