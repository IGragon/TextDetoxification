from transformers import RobertaTokenizer, RobertaForSequenceClassification, pipeline
from sentence_transformers import SentenceTransformer, util
import numpy as np


def softmax(arr):
    exp_arr = np.exp(arr - arr.max())
    return exp_arr / exp_arr.sum()


class BaseModel:
    def __init__(self):
        pass

    def predict(self, texts_list):
        pass


class ToxicityClassifier(BaseModel):
    """
    https://huggingface.co/s-nlp/roberta_toxicity_classifier
    """

    def __init__(self):
        super(ToxicityClassifier, self).__init__()
        self.tokenizer = RobertaTokenizer.from_pretrained('SkolkovoInstitute/roberta_toxicity_classifier')
        self.model = RobertaForSequenceClassification.from_pretrained('SkolkovoInstitute/roberta_toxicity_classifier')

    def predict(self, texts_list):
        results = []
        for text in texts_list:
            batch = self.tokenizer(text, return_tensors='pt')
            outputs = self.model(**batch)[0].cpu().detach().numpy()[0]
            is_toxic_prob = softmax(outputs)[1]
            results.append(is_toxic_prob)
        return results


class SimilarityClassifier(BaseModel):
    """
    https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
    """

    def __init__(self):
        super(SimilarityClassifier, self).__init__()
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    def predict(self, sentence_pairs):
        similarity_scores = []
        for sentence_pair in sentence_pairs:
            embeddings = self.model.encode(sentence_pair)
            similarity = util.cos_sim(embeddings[0], embeddings[1]).cpu().detach().numpy()[0][0]
            similarity_scores.append(similarity)
        return similarity_scores


class FluencyClassifier(BaseModel):
    """
    https://huggingface.co/textattack/roberta-base-CoLA
    """

    def __init__(self):
        super(FluencyClassifier, self).__init__()
        self.pipe = pipeline("text-classification", model="textattack/roberta-base-CoLA")

    def predict(self, texts_list):
        results = []
        for result in self.pipe(texts_list):
            score = result['score']
            if result['label'] == 'LABEL_0':
                score = 1 - score
            results.append(score)
        return results
