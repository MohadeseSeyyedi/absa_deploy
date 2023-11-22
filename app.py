
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers.models.xlm_roberta.tokenization_xlm_roberta_fast import XLMRobertaTokenizerFast
from transformers.models.xlm_roberta.modeling_xlm_roberta import XLMRobertaForSequenceClassification
from flask import Flask, request, render_template
from predict import predict

#Create an app object using the Flask class. 
app = Flask(__name__)

absa_tokenizer_path = 'models/pretrained/tokenizers/absa'
absa_model_path = 'models/pretrained/models/absa'
sent_tokenizer_path = 'models/pretrained/tokenizers/sentiment'
sent_model_path = 'models/pretrained/models/sentiment'

# Load Aspect-Based Sentiment Analysis model
absa_tokenizer = AutoTokenizer.from_pretrained(absa_tokenizer_path, local_files_only=True)
absa_model = AutoModelForSequenceClassification.from_pretrained(absa_model_path, local_files_only=True)

# Load basic sentiment analysis model
sent_tokenizer = XLMRobertaTokenizerFast.from_pretrained(sent_tokenizer_path, local_files_only=True)
sent_model = XLMRobertaForSequenceClassification.from_pretrained(sent_model_path, local_files_only=True)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def show_prediction():
    from numpy import round
    comment = request.form['comment']
    aspect = request.form['aspect']
    if aspect:
        prediction_dict = predict(comment, aspect, absa_tokenizer, absa_model)
        positive = round(prediction_dict["positive"], 2)
        negative = round(prediction_dict["negative"], 2)
        neutral = round(prediction_dict["neutral"], 2)
        
        prediction_text = f'positive:{round(100*positive, 2)}\t\t\tnegative:{round(100*negative, 2)}\t\t\tneutral:{round(100*neutral, 2)}'

    else:
        prediction_dict = predict(comment, aspect, sent_tokenizer, sent_model)
        for key, value in prediction_dict.items():
            if value:
                prediction_text = f'{round(value*100, 2)}% {key}'

    
    
    return render_template('index.html', input_comment=comment, aspect_text = prediction_dict['aspect'], prediction_text=prediction_text)


if __name__=='__main__':
    app.run()
