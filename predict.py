import sys,json,os
from gensim.corpora.dictionary import Dictionary
from gensim.models.ldamodel import LdaModel
from operator import itemgetter

json_data = ""
jsonFileName = input('Enter name of JSON document to find topic distribution for:\n\n')
jsonFileName+='.JSON'
with open(os.path.join('json/', jsonFileName)) as json_file:    
    json_text = json.load(json_file)
    json_data = json_text['m_szDocBody']

dictionary = Dictionary.load("corpusEntireDataset.dict")
lda = LdaModel.load("ldaEntireDataset")

new_doc = dictionary.doc2bow(json_data.lower().split())
topics = lda[new_doc]
print '\nTopic distribution:'
for topic,weight in sorted(topics,key=itemgetter(1),reverse=True):
	print '\nTopic ID:', topic
	print 'Weight:', weight
	print lda.show_topic(topic,topn=10)