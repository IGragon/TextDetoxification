# Solution Building Report

## Introduction

The goal of this assignment is to develop a solution for text detoxification, which involves removing toxicity and
offensive language from text while preserving semantic meaning as much as possible. As defined formally in the given
research paper, this can be framed as a text style transfer task between a toxic source style and non-toxic target
style.

When approaching this task, I considered the key challenges involved. A major difficulty is altering the style from
toxic to neutral without distorting the original content and meaning. Typical style transfer methods often inadvertently
modify or lose semantic information. Another challenge is evaluating detoxification quality without human references,
requiring automated metrics for toxicity, fluency, and semantic similarity.

Given these challenges, I hypothesized that leveraging large pretrained language models fine-tuned for paraphrasing
could allow generating fluent and benign rewrites while better retaining meaning. In addition, mining more training data
in the form of toxic/non-toxic paraphrase pairs could further improve meaning preservation. My solution exploration,
described next, focused on testing these hypotheses.

## Hypothesis 1: Leverage Pretrained Language Models

To improve on the baseline, I decided to leverage the power of pretrained language models like T5, which have been
trained on massive amounts of text data. Fine-tuning them on this task could allow knowledge transfer and boost
performance.

I experimented with T5 models of different sizes. The T5 trained on paraphrasing dataset Paws 
achieved the best
results. Fine-tuning it on the filtered ParaNMT corpus significantly improved fluency and style transfer accuracy over
the baseline.

## Hypothesis 2: Use a Paraphrasing Objective

While T5 helped, similarity scores were still low, indicating meaning was not preserved well. I hypothesized that
framing this as a paraphrasing task could better retain semantic content during style transfer.

I adopted a T5 model already fine-tuned for paraphrasing as my base model. Additional fine-tuning on style-labeled data
improved style transfer without hurting fluency or similarity as much. This formaulation was ultimately more effective.

## Hypothesis 3: Beam search and output selection

Using convenient HuggingFace interfaces for generating outputs, I generated multiple
outputs for the given text and the selected the best, combining non-toxicity, similarity, and fluency scores.

It helped in achieving more stable and reliable results.

## Final Results

The T5 paraphrasing model fine-tuned on the augmented filtered ParaNMT corpus achieved the best performance, with high
style accuracy, fluency, and improved semantic similarity. There is still room for improvement, but this approach shows
promise for text detoxification.