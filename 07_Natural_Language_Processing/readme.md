## Ch 7: Natural Language Processing

 1. [Introduction](01_Introduction#natural-language-processing-introduction)
  * We introduce methods for turning text into numerical vectors. We introduce the TensorFlow 'embedding' feature as well.
 2. [Working with Bag-of-Words](02_Working_with_Bag_of_Words#working-with-bag-of-words)
  * Here we use TensorFlow to do a one-hot-encoding of words called bag-of-words.  We use this method and logistic regression to predict if a text message is spam or ham.
 3. [Implementing TF-IDF](03_Implementing_tf_idf#implementing-tf-idf)
  * We implement Text Frequency - Inverse Document Frequency (TFIDF) with a combination of Sci-kit Learn and TensorFlow. We perform logistic regression on TFIDF vectors to improve on our spam/ham text-message predictions.
 4. [Working with Skip-Gram](04_Working_With_Skip_Gram_Embeddings#working-with-skip-gram-embeddings)
  * Our first implementation of Word2Vec called, "skip-gram" on a movie review database.
 5. [Working with CBOW](05_Working_With_CBOW_Embeddings#working-with-cbow-embeddings)
  * Next, we implement a form of Word2Vec called, "CBOW" (Continuous Bag of Words) on a movie review database.  We also introduce method to saving and loading word embeddings.
 6. [Implementing Word2Vec Example](06_Using_Word2Vec_Embeddings#using-word2vec-embeddings)
  * In this example, we use the prior saved CBOW word embeddings to improve on our TF-IDF logistic regression of movie review sentiment.
 7. [Performing Sentiment Analysis with Doc2Vec](07_Sentiment_Analysis_With_Doc2Vec#sentiment-analysis-with-doc2vec)
  * Here, we introduce a Doc2Vec method (concatenation of doc and word embeddings) to improve out logistic model of movie review sentiment.
