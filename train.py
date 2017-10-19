import os, json, sys
import gensim

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

#---------------------------------------------------Load Corpus from JSON files---------------------------------------------------#
all_json_files = []
training_data = []
training_data_size = int(sys.argv[1])-1

# this finds the json files
for dirname, dirnames, filenames in os.walk('.'):
    for filename in filenames:
    	if filename.lower().endswith(".json"):
    		all_json_files.append(filename)

# this takes ONLY the body of the articles and makes a list of them(a list of strings)
for js in all_json_files:
    while(training_data_size!=0):
        with open(os.path.join('json/', js)) as json_file:
            json_text = json.load(json_file)
            training_data.append(json_text['m_szDocBody'])
        training_data_size-=1
training_data = training_data[21:]

#------------------------------------------------Preprocess Corpus of Document Bodies----------------------------------------------#
# remove common words and tokenize
stoplist = set('for a of the and to in'.split())
documentBodies = [[word for word in training_data_one_document.lower().split() if word not in stoplist]
         for training_data_one_document in training_data]

# remove words that appear only once
from collections import defaultdict
frequency = defaultdict(int)
for documentBody in documentBodies:
    for token in documentBody:
        frequency[token] += 1
documentBodies = [[token for token in documentBody if frequency[token] > 1]
         for documentBody in documentBodies]

dictionary = gensim.corpora.Dictionary(documentBodies)
dictionary.save('corpus.dict')

corpus = [dictionary.doc2bow(documentBody) for documentBody in documentBodies]
gensim.corpora.MmCorpus.serialize('corpus.mm', corpus)  # store to disk, for later use

#------------------------------------------------------Train and Save model------------------------------------------------------#
lda = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=100, update_every=1, chunksize=10000, passes=10)
lda.save("lda")