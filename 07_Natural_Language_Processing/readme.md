## [Ch 7: Natural Language Processing](https://github.com/nfmcclure/tensorflow_cookbook/tree/master/07_Natural_Language_Processing)

 1. [Introduction](https://github.com/nfmcclure/tensorflow_cookbook/tree/master/07_Natural_Language_Processing/01_Introduction)
  * We introduce methods for turning text into numerical vectors. We introduce the Tensorflow 'embedding' feature as well.
 2. [Working with Bag-of-Words](https://github.com/nfmcclure/tensorflow_cookbook/tree/master/07_Natural_Language_Processing/02_Working_with_Bag_of_Words)
  * Here we use Tensorflow to do a one-hot-encoding of words called bag-of-words.  We use this method and logistic regression to predict if a text message is spam or ham.
 3. [Implementing TF-IDF](https://github.com/nfmcclure/tensorflow_cookbook/tree/master/07_Natural_Language_Processing/03_Implementing_tf_idf)
  * We implement Text Frequency - Inverse Document Frequency (TFIDF) with a combination of Sci-kit Learn and Tensorflow. We perform logistic regression on TFIDF vectors to improve on our spam/ham text-message predictions.
 4. [Working with CBOW](https://github.com/nfmcclure/tensorflow_cookbook/tree/master/07_Natural_Language_Processing/04_Working_With_Skip_Gram_Embeddings)
  * Our first implementation of Word2Vec called, "skip-gram" on a movie review database.
 5. [Working with Skip-Gram](https://github.com/nfmcclure/tensorflow_cookbook/tree/master/07_Natural_Language_Processing/05_Working_With_CBOW_Embeddings)
  * Next, we implement a form of Word2Vec called, "CBOW" (Continuous Bag of Words) on a movie review database.  We also introduce method to saving and loading word embeddings.
 6. [Implementing Word2Vec Example](https://github.com/nfmcclure/tensorflow_cookbook/tree/master/07_Natural_Language_Processing/06_Using_Word2Vec_Embeddings)
  * In this example, we use the prior saved CBOW word embeddings to improve on our TF-IDF logistic regression of movie review sentiment.
 7. [Performing Sentiment Analysis with Doc2Vec](https://github.com/nfmcclure/tensorflow_cookbook/tree/master/07_Natural_Language_Processing/07_Sentiment_Analysis_With_Doc2Vec)
  * Here, we introduce a Doc2Vec method (concatenation of doc and word emebeddings) to improve out logistic model of movie review sentiment.
