
# BBC News Summarization

The task aims at summarizing BBC news articles by finetuning Large Language Models.


## [Dataset](https://www.kaggle.com/datasets/pariza/bbc-news-summary?datasetId=24984&sortBy=dateRun&tab=profile)

This dataset for extractive text summarization has four hundred and seventeen political news articles of BBC from 2004 to 2005 in the News Articles folder. For each articles, five summaries are provided in the Summaries folder. The first clause of the text of articles is the respective title.
## Evaluation Metrics

The text summarization evaluation metrics we used to assess the quality of generated summaries compared to reference summaries are:

* ROUGE-1 (Recall-Oriented Understudy for Gisting Evaluation - 1): ROUGE-1 measures the overlap of unigram (individual words) between the generated summary and the reference summary. It computes the precision, recall, and F1 score based on the count of overlapping unigrams. ROUGE-1 focuses on evaluating content overlap between the generated summary and the reference summary.

* ROUGE-2: ROUGE-2 extends the evaluation to bigrams (pairs of consecutive words) instead of unigrams. It measures the precision, recall, and F1 score of the overlapping bigrams between the generated summary and the reference summary. ROUGE-2 captures the agreement in word sequences of length two.

* ROUGE-L: ROUGE-L is based on Longest Common Subsequence (LCS) and measures the longest common subsequence of words between the generated summary and the reference summary. It accounts for word ordering and captures the longest matching sequence. ROUGE-L considers the recall of the LCS and normalizes it by the length of the reference summary.

* ROUGE-Lsum: ROUGE-Lsum is an extension of ROUGE-L that takes into account multiple reference summaries. It computes the longest common subsequence of words between the generated summary and all the reference summaries, considering each reference summary separately. It aims to reward summaries that contain information present in any of the references.

* [BERTScore](https://github.com/shreyasajal/BBC-News-Summary/blob/main/bertscore.ipynb): BERTScore is a metric based on contextual embeddings and similarity estimation using BERT (Bidirectional Encoder Representations from Transformers). It calculates the similarity score between the generated summary and the reference summary at the token level. BERTScore takes into account both precision and recall of the token-level embeddings and provides a continuous score that reflects the quality of the generated summary.

These metrics evaluate different aspects of the generated summary, including word overlap, sequence similarity, content coverage, and context-awareness. They were used in combination to obtain a comprehensive evaluation of our models.

## Models Experimented
The models that we chose for our experiments are:
### [BART](https://arxiv.org/abs/1910.13461v1)
![image](https://github.com/shreyasajal/BBC-News-Summary/assets/58565264/9bce0645-bcf9-455b-96a5-a12d794fd60a)

BART is a denoising autoencoder for pretraining sequence-to-sequence models. It is trained by

 * corrupting text with an arbitrary noising function, and 
 
 * learning a model to reconstruct the original text.
 
  It uses a standard Transformer-based neural machine translation architecture. It uses a standard seq2seq/NMT architecture with a bidirectional encoder (like BERT) and a left-to-right decoder (like GPT). This means the encoder's attention mask is fully visible, like BERT, and the decoder's attention mask is causal, like GPT2.
  

### [T5](https://arxiv.org/abs/1910.10683v3)
![image](https://github.com/shreyasajal/BBC-News-Summary/assets/58565264/887ff67d-cb8b-4ca3-88d8-a6b32010744a)

T5, or Text-to-Text Transfer Transformer, is a Transformer based architecture that uses a text-to-text approach. Every task – including translation, question answering, and classification – is cast as feeding the model text as input and training it to generate some target text. This allows for the use of the same model, loss function, hyperparameters, etc. across our diverse set of tasks. The changes compared to BERT include:

* adding a causal decoder to the bidirectional architecture.
* replacing the fill-in-the-blank cloze task with a mix of alternative pre-training tasks.
### [FlanT5](https://arxiv.org/abs/2210.11416v5)
Flan-T5 is the instruction fine-tuned version of T5 or Text-to-Text Transfer Transformer Language Model.
![image](https://github.com/shreyasajal/BBC-News-Summary/assets/58565264/03f3aff7-c30e-43a6-aff5-4a4f76161c29)


Google created Flan T5, a transformer-based language model. It is based on the T5 architecture and has 12 transformer layers and a feed-forward neural network to process text in parallel. The model is one of Google's largest, with over 20 billion parameters and pre-trained on massive data sets such as web pages, books, and articles. Flan T5 comes in various sizes and is used for various NLP tasks such as text classification, summarization, and question-answering. The model is pre-trained with the BERT-style objective, where it learns to predict masked tokens, and is trained with a denoising autoencoder to capture the text's semantics.

The training procedure for Flan T5 involves two stages: pre-training and instruction finetuning. The pre-training stage is done using the T5 architecture, and it involves training the model to predict the next token in a sequence given the previous tokens. Finetuning instruction involves training the model on a collection of instruction datasets to improve its performance and generalization to unseen tasks.

## Performance Comparison
Below are the validation Rouge 1 scores, as calculated by trainer.metrics function. Note that there might be some difference in the scores calculated via model.generate generated outputs and trainer.predict generated outputs due to the difference in default parameters .Also the training logs show the metrics as calculated via a different method than trainer.metrics that is why the eval rouge in the logs are different from the final validation logs. The log metrics can be used to track the improvement as the training progresses. Whereas the validation and test rouge scores of the best checkpoint can be used to evaluate our model's performances.

| Model       | Validation Rouge1 | Test Rouge1 | Test BERTScore(f1) | No. of epochs | Model size             |
|-------------|-------------------|-------------|--------------------|---------------|------------------------|
| BART-large  | 64.9158           | 63.3473     | 0.911              | 7             | 400 million parameters |
| BART-base   | 63.9606           | 63.1959     | 0.907              | 20            | 140 million parameters |
| T5-base     | 62.8756           | 62.6224     | 0.88               | 20            | 220 million parameters |
| FlanT5-base | 59.9384           | 59.6714     | 0.861              | 20            | 250 million parameters |

We can see that BART-Large is our best performing model ( in terms of both Rouge and BERTscore) and it beats the other model's performance in just 7 epochs. To fit BART-large and FlanT5-base, a batch size of 4 was used and gradient accumulation of 4 was used. Hence, the effective batch size was 16. For all the other models a batch size of 8 with gradient accumulation of 2 was used.

![image](https://github.com/shreyasajal/BBC-News-Summary/assets/58565264/d607aa13-6d3e-4d1c-bea7-a9baf6c9e189)
![image](https://github.com/shreyasajal/BBC-News-Summary/assets/58565264/30f610b1-d388-43d0-85fc-714b9b1f67e0)

## Experiment Tracking

Weights and Biases (W&B) is a platform that provides tools for experiment tracking, model visualization, and collaboration in machine learning projects. While W&B is primarily known for its experiment tracking capabilities, it can also be used for information tracking in various contexts.

![image](https://github.com/shreyasajal/BBC-News-Summary/assets/58565264/90e95919-ebc6-46c8-9cb6-0aae99f23b3e)

[This](https://wandb.ai/shreyasajal/huggingface?workspace=user-shreyasajal) link can be followed to track the experiments that were conducted for our task.

## Challenges
The major challenges while performing the experiments was:
1. Fitting large variants of the LLMs on the available P100 16GB GPU. Due to their immense size we often ran out of GPU memory and training took longer time than a kaggle session permits. 
The large variants of models like T5 and FlanT5 couldn't be fitted even by applying the following methods to consume less memory:
* Reducing Batch Size till 2
* Reducing the maximum input tokens to 384
* Using Gradient Accumulation
* Using Gradient Checkpointing
* FP16 training
* Changing Optimizers (AdaFactor optimizer)

These are inspired by [this](https://huggingface.co/docs/transformers/v4.18.0/en/performance) hugging face blog and more details can be found there.

2. Even though the models we used supported max_length>512 but we chose 512 as the max number of tokens, and truncated the texts that generated tokens> 512. This was done taking in mind the higher memory requirements that come with higher token lengths( 1024, 2048, etc). This truncation resulted in loss of information while generating the summaries.

## Long Text Summarization 
To deal with the problem of loss of information in long text summarization, we experimented three methods highly inspired by the CombineDocumentChain methods used the LangChain Model for long text summarization. 
* Map Reduce
* Modified Map Reduce
* Refine

The method descriptions and the code can be found in the long text summarization notebook.

We saw that Map Reduce and Refine method didn't improve the scores. Refine method was expected to increase the score specially on FlanT5( since it is instruction tuned, it was hypothesised that it could process the prompts) but we saw that it didn't help much there also. One of the possible reasons is that the model was finetuned by us for summary generation task so it doesn't understand the prompts to refine the summary based on the new context.
The Modified Map reduce method gave a significant improvement in score. The results were similar for other trained models also.


