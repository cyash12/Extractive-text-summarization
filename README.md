Extractive Text Summary Method:
The main algorithm used is Term Frequency-Inverse Document Frequency
Every word is first ranked as per their frequency within the originating document and then compared with its frequency throughout the corpus
Once the words are assigned weights, sentences are given a score depending upon the weight of the words contained in the sentence
Additionally, sentences are given additional bias depending upon their position in the article, further down higher the priority.
Data is first cleaned to remove abbreviations, special characters and certain punctuations.
Next the stopwords are removed so as they do not disturb the weights assigned to terms.
Sentences are vectorised and a vocab is generated for training TF-IDF.
The document to be summarized is then vectorised and converted into a document-term matrix
These terms are assigned weights based on the TF-IDF fitting.
Sentences are ranked as per the weights of their terms contained
The top 3 sentences are taken as the summary
