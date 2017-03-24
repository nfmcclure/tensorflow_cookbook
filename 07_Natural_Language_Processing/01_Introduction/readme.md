# Natural Language Processing Introduction

Placeholder for future purposes.

Up to this point, we have only considered machine learning algorithms that mostly operate on numerical inputs.  If we want to use text, we must find a way to convert the text into numbers.  There are many ways to do this and we will explore a few common ways this is achieved.\n",


If we consider the sentence **“tensorflow makes machine learning easy”**, we could convert the words to numbers in the order that we observe them.  This would make the sentence become “1 2 3 4 5”.  Then when we see a new sentence, **“machine learning is easy”**, we can translate this as “3 4 0 5”. Denoting words we haven’t seen bore with an index of zero.  With these two examples, we have limited our vocabulary to 6 numbers.  With large texts we can choose how many words we want to keep, and usually keep the most frequent words, labeling everything else with the index of zero.


If the word “learning” has a numerical value of 4, and the word “makes” has a numerical value of 2, then it would be natural to assume that “learning” is twice “makes”.  Since we do not want this type of numerical relationship between words, we assume these numbers represent categories and not relational numbers.


Another problem is that these two sentences are of different size. Each observation we make (sentences in this case) need to have the same size input to a model we wish to create.  To get around this, we create each sentence into a sparse vector that has that value of one in a specific index if that word occurs in that index.
