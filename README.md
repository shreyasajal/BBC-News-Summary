
# BBC News Summarization

The task aims at summarizing BBC news articles by finetuning Large Language Models.


## Dataset

## Evaluation Metrics

The text summarization evaluation metrics we used to assess the quality of generated summaries compared to reference summaries are:

* ROUGE-1 (Recall-Oriented Understudy for Gisting Evaluation - 1): ROUGE-1 measures the overlap of unigram (individual words) between the generated summary and the reference summary. It computes the precision, recall, and F1 score based on the count of overlapping unigrams. ROUGE-1 focuses on evaluating content overlap between the generated summary and the reference summary.

* ROUGE-2: ROUGE-2 extends the evaluation to bigrams (pairs of consecutive words) instead of unigrams. It measures the precision, recall, and F1 score of the overlapping bigrams between the generated summary and the reference summary. ROUGE-2 captures the agreement in word sequences of length two.

* ROUGE-L: ROUGE-L is based on Longest Common Subsequence (LCS) and measures the longest common subsequence of words between the generated summary and the reference summary. It accounts for word ordering and captures the longest matching sequence. ROUGE-L considers the recall of the LCS and normalizes it by the length of the reference summary.

* ROUGE-Lsum: ROUGE-Lsum is an extension of ROUGE-L that takes into account multiple reference summaries. It computes the longest common subsequence of words between the generated summary and all the reference summaries, considering each reference summary separately. It aims to reward summaries that contain information present in any of the references.

* BERTScore: BERTScore is a metric based on contextual embeddings and similarity estimation using BERT (Bidirectional Encoder Representations from Transformers). It calculates the similarity score between the generated summary and the reference summary at the token level. BERTScore takes into account both precision and recall of the token-level embeddings and provides a continuous score that reflects the quality of the generated summary.

These metrics evaluate different aspects of the generated summary, including word overlap, sequence similarity, content coverage, and context-awareness. They were used in combination to obtain a comprehensive evaluation of our models.

## Models Experimented
The models that we chose for our experiments are:
### [BART]("https://arxiv.org/abs/1910.13461v1")
BART is a denoising autoencoder for pretraining sequence-to-sequence models. It is trained by

 * corrupting text with an arbitrary noising function, and 
 
 * learning a model to reconstruct the original text.
 
  It uses a standard Transformer-based neural machine translation architecture. It uses a standard seq2seq/NMT architecture with a bidirectional encoder (like BERT) and a left-to-right decoder (like GPT). This means the encoder's attention mask is fully visible, like BERT, and the decoder's attention mask is causal, like GPT2.
  

### [T5]("https://arxiv.org/abs/1910.10683v3")

T5, or Text-to-Text Transfer Transformer, is a Transformer based architecture that uses a text-to-text approach. Every task – including translation, question answering, and classification – is cast as feeding the model text as input and training it to generate some target text. This allows for the use of the same model, loss function, hyperparameters, etc. across our diverse set of tasks. The changes compared to BERT include:

* adding a causal decoder to the bidirectional architecture.
* replacing the fill-in-the-blank cloze task with a mix of alternative pre-training tasks.
### [FlanT5]("https://arxiv.org/abs/2210.11416v5")
Flan-T5 is the instruction fine-tuned version of T5 or Text-to-Text Transfer Transformer Language Model.


Google created Flan T5, a transformer-based language model. It is based on the T5 architecture and has 12 transformer layers and a feed-forward neural network to process text in parallel. The model is one of Google's largest, with over 20 billion parameters and pre-trained on massive data sets such as web pages, books, and articles. Flan T5 comes in various sizes and is used for various NLP tasks such as text classification, summarization, and question-answering. The model is pre-trained with the BERT-style objective, where it learns to predict masked tokens, and is trained with a denoising autoencoder to capture the text's semantics.

The training procedure for Flan T5 involves two stages: pre-training and instruction finetuning. The pre-training stage is done using the T5 architecture, and it involves training the model to predict the next token in a sequence given the previous tokens. Finetuning instruction involves training the model on a collection of instruction datasets to improve its performance and generalization to unseen tasks.

## Performance Comparison

## Experiment Tracking

## Challenges

## Long Text Summarization 

## Hardware Information

## Zero Shot Approach to the Task


