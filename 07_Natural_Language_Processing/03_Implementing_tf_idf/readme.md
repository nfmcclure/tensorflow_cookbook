# Implementing TF-IDF
-------------------------

TF-IDF is an acronym that stands for Text Frequency – Inverse Document Frequency.  This term is essentially the product of text frequency and inverse document frequency for each word.


In the prior recipe, we introduced the bag of words methodology, which assigned a value of one for every occurrence of a word in a sentence. This is probably not ideal as each category of sentence (spam and ham for the prior recipe example) most likely has the same frequency of “the”, “and” and other words, whereas words like “viagra" and “sale” probably should have increased importance in figuring out whether or not the text is spam.


We first want to take into consideration the word frequency.  Here we consider the frequency that a word occurs in an individual entry. The purpose of this part (TF), is to find terms that appear to be important in each entry.


But words like “the” and “and” may appear very frequently in every entry. We want to down weight the importance of these words, so we can imagine that multiplying the above text frequency (TF) by the inverse of the whole document frequency might help find important words.  But since a collection of texts (a corpus) may be quite large, it is common to take the logarithm of the inverse document frequency.  This leaves us with the following formula for TF-IDF for each word in each document entry.

$$
w_{tf-idf}=w_{tf} \cdot \frac{1}{log(w_{df})}
$$


Where $w_{tf}$ is the word frequency by document, and $w_{df}$ is the total frequency of such word across all documents.  We can imagine that high values of TF-IDF might indicate words that are very important to determining what a document is about.


Creating the TF-IDF vectors requires us to load all the text into memory and count the occurrences of each word before we can start training our model.  Because of this, it is not implemented fully in Tensorflow, so we will use Scikit-learn for creating our TF-IDF embedding, but use Tensorflow to fit the logistic model.
