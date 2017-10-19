# Code-Challenge
This project used topic modelling to find a distribution of topics for the texts in the webpages provided.

## Getting Started

### Prerequisites

* Anaconda (if running on Windows)
* gensim

To run on Windows I used Anaconda (can be downloaded from [here](https://www.anaconda.com/download/ "Download Anaconda")).
Then run the scripts from the Anaconda prompt instead of the Windows one (including the gensim installation).

To install gensim use:
```
pip install gensim
```
### Installation
Download repository.
## Running
### Training the Model
There is already a trained model saved as "ldaEntireDataset" in the folder. It was trained for about 6 hours on the entire dataset (the first 20 objects were not used in the training to serve as testing set).

To train the model on another dataset, run in the command prompt (or Anaconda prompt):

``` python train.py x ```

where x is the number of JSON objects to include in the training set.

The model will be saved as "lda" to be distinct from the model trained on the entire dataset.
### Getting a list of the topics
To get a list of the distributions run:

``` python print_topic_distribution.py x```

where x is the size of the training set for the model. Type "entire" (no quotations) to get the list for the model using the entire dataset.

The topics are printed in a text file in the folder "topic distributions" with the words contained in each topic.

The script uses the model trained on the entire set. To predict based on other models manually change the model and corpus loaded in the script. Same goes for predicting the topic distribution.
### Predicting a topic distribution for unseen document
Run in the command prompt (or Anaconda prompt):

``` python predict.py x```

where x is the size of the training set for the model. Type "entire" (no quotations) to get a prediction based on the model using the entire dataset.

Once you run predict.py, you'll be prompted to enter the name/id of the unseen document. Please enter the number of the JSON file **surrounded by quotation marks**(eg for the file 232.JSON, write "232").

The topic distribution for the new file will be printed in the command window with the words contained in each topic.
## Motivation
Looking through the files I noticed the field "m_Topics" is empty in a lot of the JSON objects.

I wanted to see if I can find any common topics across the different documents. I did that by training an LDA model (a model that finds topic distributions over documents) on the bodies of the documents (webpages) and manually check the topics to see if there are any meaningful ones.

The model returns a list of topics and each topic contains words/tokens with weights. For example,

` Topic 3:  assembly:0.009  vehicle:0.035  system:0.019  device:0.006  video:0.007  fig:0.011  mirror:0.011  interior:0.007  include:0.006  display:0.011 `

This may be interpreted as in some of the documents, the content is about the device system in cars.

After training the model can be used to find a topic distribution for a single unseen document. A document may contain a few different topics and each topic may contain different words.

### LDA
LDA(Latent Dirichlet allocation) is a generative statistical model that makes a few assumptions:
1. A topic is a distribution over words
2. A document is a mixture of corpus-wide topics
3. Each word is drawn from one of those topics

More information can be found [here](http://psiexp.ss.uci.edu/research/papers/Griffiths_Steyvers_Tenenbaum_2007.pdf "Topics in Semantic Representation").
