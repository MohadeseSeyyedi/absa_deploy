from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers.models.xlm_roberta.tokenization_xlm_roberta_fast import XLMRobertaTokenizerFast
from transformers.models.xlm_roberta.modeling_xlm_roberta import XLMRobertaForSequenceClassification
from predict import predict

# Load Aspect-Based Sentiment Analysis model
absa_tokenizer = AutoTokenizer.from_pretrained("models/pretrained/tokenizers/absa", local_files_only=True)
absa_model = AutoModelForSequenceClassification.from_pretrained("models/pretrained/models/absa", local_files_only=True)

# Load basic sentiment analysis model
sent_tokenizer = XLMRobertaTokenizerFast.from_pretrained("models/pretrained/tokenizers/sentiment", local_files_only=True)
sent_model = XLMRobertaForSequenceClassification.from_pretrained("models/pretrained/models/sentiment", local_files_only=True)

print(type(sent_tokenizer))
print(type(sent_model))

sentence = "I'm pretty satisfied with the price of the laptop but the screen resolution is just fine"

print(f"Sentence: {sentence}")
print()

# ABSA of "food"
aspect = "screen resolution"

output_dict = predict(sentence, aspect, absa_tokenizer, absa_model)
print(output_dict)

output_dict = predict(sentence, None, sent_tokenizer, sent_model)
print(output_dict)