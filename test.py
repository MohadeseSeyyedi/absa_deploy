from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load Aspect-Based Sentiment Analysis model
absa_tokenizer = AutoTokenizer.from_pretrained("models/pretrained/tokenizers/absa", local_files_only=True)
absa_model = AutoModelForSequenceClassification.from_pretrained("models/pretrained/models/absa", local_files_only=True)

print(type(absa_tokenizer))
print(type(absa_model))