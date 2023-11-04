from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
from datasets import Dataset
from huggingface_hub import interpreter_login

import numpy as np
import evaluate
import nltk

from helper_models import ToxicityClassifier, SimilarityClassifier, FluencyClassifier


class DetoxificationModel:
    """
    Model to fine-tune: https://huggingface.co/Vamsi/T5_Paraphrase_Paws
    """
    def __init__(self, model_name: str = "Vamsi/T5_Paraphrase_Paws", tokenizer_name: str = "Vamsi/T5_Paraphrase_Paws"):
        self.toxicity_classifier = ToxicityClassifier()
        self.similarity_classifier = SimilarityClassifier()
        self.fluency_classifier = FluencyClassifier()

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to("cuda")

    def choose_output(self, original_text, generated_texts):
        toxicity_scores = self.toxicity_classifier.predict(generated_texts)
        toxicity_scores = [1 - score for score in toxicity_scores]

        fluency_scores = self.fluency_classifier.predict(generated_texts)

        paired_texts = [[original_text, text] for text in generated_texts]
        similarity_scores = self.similarity_classifier.predict(paired_texts)

        scores = [toxicity_score * fluency_score * similarity_score for toxicity_score, fluency_score, similarity_score in zip(toxicity_scores, fluency_scores, similarity_scores)]
        best_index = np.argmax(scores)
        return generated_texts[best_index], {"non-toxicity": toxicity_scores[best_index],
                                             "fluency": fluency_scores[best_index],
                                             "similarity": similarity_scores[best_index]}

    def predict(self, texts_list: list[str]) -> (list[str], dict):
        results = []
        for text in tqdm(texts_list, desc="Detoxification model / predict"):
            original_text = text
            text = "paraphrase: " + text

            encoding = self.tokenizer.encode_plus(text, pad_to_max_length=True, return_tensors="pt")
            input_ids, attention_masks = encoding["input_ids"].to("cuda"), encoding["attention_mask"].to("cuda")

            outputs = self.model.generate(
                input_ids=input_ids, attention_mask=attention_masks,
                max_length=256,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                early_stopping=True,
                num_return_sequences=7
            )

            outputs = [self.tokenizer.decode(output,
                                             skip_special_tokens=True,
                                             clean_up_tokenization_spaces=True) for output in outputs]
            output_text, output_scores = self.choose_output(original_text, outputs)
            results.append({"output": output_text, "scores": output_scores})
        return results

    def tokenize_dataset(self, dataset: Dataset) -> Dataset:
        inputs = ["paraphrase: " + text for text in dataset["tox_high"]]
        model_inputs = self.tokenizer(inputs, max_length=256, truncation=True)
        labels = self.tokenizer(text_target=dataset["tox_low"], max_length=256, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def train(self, dataset: Dataset, args):
        nltk.download("punkt", quiet=True)
        metric = evaluate.load("rouge")

        def compute_metrics(eval_pred):
            preds, labels = eval_pred

            # decode preds and labels
            labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
            decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
            decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

            # rougeLSum expects newline after each sentence
            decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
            decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

            result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
            return result

        # if not args.store_locally:
        #     interpreter_login()

        dataset = dataset.train_test_split(test_size=0.2, seed=42)
        tokenized_dataset = dataset.map(self.tokenize_dataset, batched=True)
        data_collator = DataCollatorForSeq2Seq(tokenizer=self.tokenizer, model=self.model)

        training_args = Seq2SeqTrainingArguments(
            output_dir=args.output_dir,
            evaluation_strategy=args.evaluation_strategy,
            learning_rate=args.learning_rate,
            per_device_train_batch_size=args.train_batch_size,
            per_device_eval_batch_size=args.eval_batch_size,
            weight_decay=args.weight_decay,
            save_total_limit=args.save_total_limit,
            num_train_epochs=args.num_train_epochs,
            fp16=True,
            predict_with_generate=True,
            report_to="none"
        )

        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["test"],
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics
        )

        print(trainer.train())

        if args.store_locally:
            self.model.save_pretrained(args.save_name)
        else:
            self.model.push_to_hub(args.save_name)

