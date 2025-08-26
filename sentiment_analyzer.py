# sentiment_analyzer.py

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from scipy.special import softmax

class SentimentAnalyzer:
    """
    A class to perform sentiment analysis using a pre-trained Transformer model.
    """
    def __init__(self, model_name="cardiffnlp/twitter-roberta-base-sentiment-latest"):
        """
        Initializes the tokenizer and model.
        
        Args:
            model_name (str): The name of the pre-trained model from Hugging Face.
        """
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.labels = ['Negative', 'Neutral', 'Positive']
            print(" Model loaded successfully.")
        except Exception as e:
            print(f" Error loading model: {e}")
            self.tokenizer = None
            self.model = None

    def analyze(self, text):
        """
        Analyzes the sentiment of a given text.

        Args:
            text (str): The input text to analyze.

        Returns:
            dict: A dictionary containing the predicted label and the scores for each sentiment.
                  Returns None if the model failed to load.
        """
        if not self.model or not self.tokenizer:
            return None

        # 1. Tokenize the input text
        encoded_input = self.tokenizer(text, return_tensors='pt')

        # 2. Get model output (logits)
        with torch.no_grad():
            output = self.model(**encoded_input)
        
        scores = output[0][0].detach().numpy()
        
        # 3. Convert logits to probabilities using softmax
        scores = softmax(scores)

        # 4. Rank the scores and get the predicted label
        ranking = scores.argsort()[::-1]
        predicted_label = self.labels[ranking[0]]

        # 5. Create a dictionary of scores, ensuring order is always Neg, Neu, Pos
        result = {
            "predicted_label": predicted_label,
            "scores": {
                'Negative': scores[0],
                'Neutral': scores[1],
                'Positive': scores[2]
            }
        }
        
        return result

# --- MODIFIED SECTION ---
if __name__ == '__main__':
    analyzer = SentimentAnalyzer()
    
    # Ensure the model loaded before proceeding
    if analyzer.model:
        # Define the path to your text file
        file_path = 'document.txt'
        print(f"\nAnalyzing sentiments from '{file_path}'...\n")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                # Read all lines from the file
                phrases = f.readlines()
            
            # Iterate over each phrase in the file
            for phrase in phrases:
                # Clean up the line by removing whitespace and newline characters
                clean_phrase = phrase.strip()
                
                # Skip empty lines or comment lines
                if not clean_phrase or clean_phrase.startswith('#'):
                    continue
                
                # Analyze the cleaned phrase
                result = analyzer.analyze(clean_phrase)
                
                # Print the results in a readable format
                if result:
                    print(f"Text: '{clean_phrase}'")
                    print(f"  -> Predicted Sentiment: {result['predicted_label']}")
                    
                    # Optional: Print the detailed scores formatted to 4 decimal places
                    scores_str = (f"Positive: {result['scores']['Positive']:.4f}, "
                                  f"Neutral: {result['scores']['Neutral']:.4f}, "
                                  f"Negative: {result['scores']['Negative']:.4f}")
                    print(f"     Scores: [{scores_str}]")
                    print("-" * 50)

        except FileNotFoundError:
            print(f" ERROR: The file '{file_path}' was not found. Please make sure it's in the same directory as the script.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
