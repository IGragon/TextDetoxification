# Introduction
Text detoxification refers to the task of removing toxic or offensive 
language from text while preserving the original meaning as much as possible.
As discussed in the [article](../references/Text%20Detoxification%20using%20Large%20Pre-trained%20Neural%20Models.pdf), this can be framed as a style
transfer task where the "style" is the level of toxicity. 
The article provides a formal definition of a text detoxification 
model as a function that takes as input a source text containing toxic 
language, and outputs a new version with reduced toxicity, high semantic 
similarity to the original, and fluency.

This is an important task as the growing amount of user-generated content
online inevitably contains some level of toxic language that could be
harmful. Automatically cleaning up text rather than simply removing it 
could enable better online conversations and reduce harm. Potential 
applications include suggesting less toxic alternatives to users before 
they post angry comments, or making chatbots safer by detoxifying any 
inappropriate responses they generate. However, text detoxification is 
challenging as it requires changing the style while minimally distorting 
the original meaning.

# Data analysis
The provided filtered ParaNMT-detox corpus was a good starting point for 
building a solution. 

First, I mentioned that reference and translation toxicity
is chaotic. Therefore, I refactored the dataset to the format of
"high toxic" and "less toxic" texts is pair. 

Second, I selected only pairs which toxicity difference was greater than 0.9,
leaving over 400k samples.

Finally, I separated what was left in training set with 325k samples and 
evaluation set with 80k samples. Where evaluation set was not seen by model
during training at all.

Another dataset, I stumbled upon was [The Toxicity Dataset](https://github.com/surge-ai/toxicity/tree/main) that 
contains 500 toxic and 500 non-toxic tweets. Initially, I wanted to train toxicity classifier on it, but decided
to use 500 toxic samples as a benchmark for the model.

# Model Specification

As suggested in the given article, I leverage the power of pretrained LLMs
to create my own solution.

As style transfer is basically a paraphrasing task, I took [T5 LLM pretrained on Paws dataset](https://huggingface.co/Vamsi/T5_Paraphrase_Paws)
as my base model and fine-tuned it with HuggingFace convenient interfaces.

Resulting model occupies 892 Mb of storage space.

## More on T5 model

**Specification:**

1. **Model Size**: T5 comes in various sizes, with "T5 Large" being one of the larger variants, typically containing 
355 million parameters. Smaller and larger versions also exist.

2. **Architecture**: T5 is based on the Transformer architecture, which uses self-attention mechanisms to process input 
data in parallel. It employs a stack of encoder and decoder layers.

3. **Training Data**: T5 is pre-trained on a massive corpus of text from the internet, making it capable of understanding 
and generating human language in a wide variety of contexts.

4. **Fine-Tuning**: T5 can be fine-tuned for specific natural language understanding and generation tasks, such as 
translation, text summarization, question answering, and more.

**Description:**

T5 is a versatile language model that approaches various NLP tasks through
a consistent framework. Unlike earlier models, which were fine-tuned separately
for each task, T5 adopts a "text-to-text" approach. In this approach,
both input and output are treated as text, which simplifies the 
understanding of many tasks. T5 is pre-trained using a denoising
autoencoder objective, where it learns to reconstruct text that
is corrupted. During fine-tuning, specific task instructions are 
added to the input text, guiding T5 to generate the desired output.

# Training Process

Some details of training can be found in the following supplementary materials:
- [Training Seq2Seq models with HuggingFace](https://huggingface.co/docs/evaluate/main/en/transformers_integrations#seq2seqtrainer)
- [Loading Tabular data as datasets' Dataset](https://huggingface.co/docs/datasets/tabular_load#csv-files)
- [Saving HuggingFace model for later use](https://discuss.huggingface.co/t/how-to-save-my-model-to-use-it-later/20568/2)


For training, I used Kaggle environment with P100 GPU. I fine-tuned the model on 80% of 
training set and validated it on the remaining 20% of training set.

Each training and validation epoch took ~1h. Resulting in 4h of training for 2 epochs.
Turns out one epoch was enough, as the validation after the second epoch did no show \
significant improvements over the first validation step.

Training parameters were quite default:
- learning_rate: 2e-5
- train_batch_size: 16
- eval_batch_size: 8
- weight_decay: 0.01
- num_train_epochs: 2


# Evaluation

Evaluating my solution required three more language models:
- [ToxicityClassifier](https://huggingface.co/s-nlp/roberta_toxicity_classifier)
- [SimilarityClassifier](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- [FluencyClassifier](https://huggingface.co/textattack/roberta-base-CoLA)

Their interfaces can be found in [helper_models.py](../src/models/helper_models.py)

## Prediction process

Using generative capabilities of chosen model we generate multiple candidates (7 to be exact) using:
```python
outputs = self.model.generate(
                input_ids=input_ids, attention_mask=attention_masks,
                max_length=256,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                early_stopping=True,
                num_return_sequences=7
            )
```

Then we choose the best candidate with the following procedure:
1. Calculate (non)toxicity, similarity, and fluency score using models mentioned above;
2. Obtain overall score as multiplication of these three scores;
3. Select the best candidate as argmax of the overall score.

Implementation of these process can be found in the [detoxiciation_model.py](../src/models/detoxificaton_model.py).

## Evaluation implementation and options

Due to high computational complexity of the procedure mentioned above, evaluating model on whole ~80k would
take ~70h, which is too much. Therefore, I decided to perform this procedure on random subsample of testing set with 
the size of 1000. 

Additionally, I implemented evaluating model on Toxic examples from The Toxicity Dataset.

The output of the evaluation script is the average non-toxicity, similarity, and fluency scores
for the given evaluation samples.


# Results

## Comparative evaluation

|Mean metrics on 1000 random samples| T5 finetuned with 2k samples | T5 final finetune 260k samples, 2 epochs |
|------------|-------------------------------|------------------------------------------|
|non-toxicity| 0.8877 | **0.9838** |
|fluency| 0.8677 | **0.9089** |
|similarity| **0.7986** | 0.7761 |

## Quantitative evaluation

|Mean metrics| T5 final finetune (The Toxicity Dataset) |  T5 final finetune (1000 random samples) |
|------------|-------------------------------|------------------------------------------|
|non-toxicity| 0.9697 | **0.9838** |
|fluency| 0.7799 | **0.9089** |
|similarity| 0.7283 | **0.7761** |


Examples with real text can be found in [the final notebook](../notebooks/3.2_Training_and_evaluating_paraphraser_within_module.ipynb)

## Final thoughts

The results demonstrate the potential of text detoxification models to mitigate online toxicity. The high non-toxicity and fluency scores indicate the model's ability to rewrite offensive text in a benign and coherent manner.

However, there is still room for improvement, particularly in preserving semantic similarity. As seen from the scores, the model struggles to retain the full meaning of longer and more complex texts. Expanding the training data with more varied examples could help address this limitation.

In the future, combining retrieval-based methods with generative paraphrasing may better retain meaning for difficult cases. For instance, retrieving similar non-toxic texts as a starting point for rewriting could augment the paraphraser. Interactive interfaces could also allow human input to guide the model and correct meaning distortions.

Overall, this project shows promise in automatic text detoxification. With further research into semantic preservation, such models could become practically useful for moderating online content. This could lead to genuine benefits in fostering healthier online interactions.
