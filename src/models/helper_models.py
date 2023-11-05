from transformers import RobertaTokenizer, RobertaForSequenceClassification, pipeline
from sentence_transformers import SentenceTransformer, util
import numpy as np


def softmax(arr):
    """
    Calculate the softmax function for an array of values.

    Args:
        arr (numpy.ndarray): Input array of values.

    Returns:
        numpy.ndarray: Output array with softmax values.
    """
    exp_arr = np.exp(arr - arr.max())
    return exp_arr / exp_arr.sum()


class BaseModel:
    def __init__(self):
        pass

    def predict(self, texts_list):
        pass


class ToxicityClassifier(BaseModel):
    """
    Toxicity classifier using the Hugging Face Roberta model.
    Model: https://huggingface.co/s-nlp/roberta_toxicity_classifier
    """

    def __init__(self):
        super(ToxicityClassifier, self).__init__()
        self.tokenizer = RobertaTokenizer.from_pretrained('SkolkovoInstitute/roberta_toxicity_classifier')
        self.model = RobertaForSequenceClassification.from_pretrained('SkolkovoInstitute/roberta_toxicity_classifier')

    def predict(self, texts_list):
        """
        Predict toxicity scores for a list of texts.

        Args:
            texts_list (list of str): List of input texts.

        Returns:
            list of float: List of toxicity scores for each text.
        """
        results = []
        for text in texts_list:
            batch = self.tokenizer(text, return_tensors='pt')
            outputs = self.model(**batch)[0].cpu().detach().numpy()[0]
            is_toxic_prob = softmax(outputs)[1]
            results.append(is_toxic_prob)
        return results


class SimilarityClassifier(BaseModel):
    """
    Sentence similarity classifier using the Sentence Transformers model.
    Model: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
    """

    def __init__(self):
        super(SimilarityClassifier, self).__init__()
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    def predict(self, sentence_pairs):
        """
        Calculate similarity scores for a list of sentence pairs.

        Args:
            sentence_pairs (list of list): List of pairs of sentences to compare.

        Returns:
            list of float: List of similarity scores for each sentence pair.
        """
        similarity_scores = []
        for sentence_pair in sentence_pairs:
            embeddings = self.model.encode(sentence_pair)
            similarity = util.cos_sim(embeddings[0], embeddings[1]).cpu().detach().numpy()[0][0]
            similarity_scores.append(similarity)
        return similarity_scores


class FluencyClassifier(BaseModel):
    """
    Fluency classifier using the Hugging Face Roberta model.
    Model: https://huggingface.co/textattack/roberta-base-CoLA
    """

    def __init__(self):
        super(FluencyClassifier, self).__init__()
        self.pipe = pipeline("text-classification", model="textattack/roberta-base-CoLA")

    def predict(self, texts_list):
        """
        Predict fluency scores for a list of texts.

        Args:
            texts_list (list of str): List of input texts.

        Returns:
            list of float: List of fluency scores for each text.
        """
        results = []
        for result in self.pipe(texts_list):
            score = result['score']
            if result['label'] == 'LABEL_0':
                score = 1 - score
            results.append(score)
        return results
