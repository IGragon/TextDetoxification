# Solution Building Report

## Evaluating results
First step towards text detoxification is to learn to separate toxic texts from non-toxic. Therefore, we need to
train not that complex toxicity classifier.

Using such classifier we can further evaluate the performance of different approaches.

The second step is to learn to evaluate similarity between texts, as we want detoxified sentence to convey the meaning of
the original one as much as possible.

As a result, we want to have two decent classifiers that will evaluate the performance of our approaches.


## Basic approach
The first idea that comes to mind is to have a list of bad words and a list of neutral synonyms. The algorithm will
simply substitute the bad word with its neutral equivalent.

## Advanced approach
First thing that we want to understand is that the formal definition of Detoxification task can be reduced to Seq2Seq
generation. Moreover, when we want resulting text to be similar to the original one then the task is called paraphrasing.

So, more advanced approach is to fine-tune some LLM that was firstly trained to perform text paraphrasing.