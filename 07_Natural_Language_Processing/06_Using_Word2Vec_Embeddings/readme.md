# Using Word2Vec Embeddings

## Summary

Now that we have created and saved CBOW word embeddings, we need to use them to make sentiment prediction on the movie data set.  In this recipe, we will learn how to load and use prior trained embeddings and use these embeddings to perform sentiment analysis by training a logistic linear model to predict a good or bad review.

Sentiment analysis is a really hard task to do because human language makes it very hard to grasp the subtleties and nuances of the true meaning.  Sarcasm, jokes, ambiguous references all make the task exponentially harder.  We will create a simple logistic regression on the movie review data set to see if we can get any information out of the CBOW embeddings we created and saved in the prior recipe.  Since the focus of this recipe is in the loading and usage of saved embeddings, we will not pursue more complicated models.

## Pre-requisites

In order to load the CBOW embeddings, we must run the [CBOW tutorial](../05_Working_With_CBOW_Embeddings) before running this one.  Navigate to the [CBOW tutorial](../05_Working_With_CBOW_Embeddings) folder and run:

    python3 05_Working_With_CBOW.py

This will train and save the CBOW embeddings we need to perform predictions for this recipe.
