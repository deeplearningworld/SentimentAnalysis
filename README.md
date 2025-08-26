# SentimentAnalysis


###Sentiment Analyzer###

This project provides a Python script for classifying the sentiment of text as Positive, Negative, or Neutral. It processes a list of phrases from a text file and prints the analysis for each one.It uses a pre-trained RoBERTa model from the Hugging Face library to achieve high accuracy without needing to train a model from scratch.

Features
High Accuracy: Leverages the cardiffnlp/twitter-roberta-base-sentiment-latest model, which is fine-tuned on millions of tweets for excellent performance on social media-style text.
Batch Processing: Analyzes all phrases listed in a document.txt file in a single run.
Clear Output: Displays the predicted sentiment and detailed probability scores for each phrase.
Lightweight: Simple command-line script with minimal dependencies.

Tech Stack:
-Python
-PyTorch: The backend framework for the model.
-Hugging Face transformers: For accessing the pre-trained RoBERTa model and tokenizer.
-SciPy: Used for the softmax function to calculate probability scores.
