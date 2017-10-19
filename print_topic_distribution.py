from gensim.corpora.dictionary import Dictionary
from gensim.models.ldamodel import LdaModel

def get_topic_words(ldaModel, topicID, dictionary, n=10):
    word_dist=ldaModel.get_topic_terms(topicid=topicID,topn=n)
    res={}
    for t,p in word_dist:
        res[dictionary[t]]=p
    return res

lda = LdaModel.load("ldaEntireDataset")
dictionary = Dictionary.load("corpusEntireDataset.dict")
topics=lda.print_topics(num_topics=-1,num_words=10)
text_file = open("Topic Distribution.txt", "w")
for topic in topics:
    text_file.write('\nTopic %s: '%(topic[0]))
    y = get_topic_words(lda,topic[0],dictionary)
    for k, v in y.iteritems():
        text_file.write(' %s:%.3f '%(k,v))
text_file.close()
