from flask import Flask, request, render_template, jsonify
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

app = Flask(__name__)

# Load the fine-tuned model and tokenizer
model_path = "./sarcasm_detector_model"  # Path where you saved your fine-tuned model
tokenizer_path = "./sarcasm_detector_tokenizer"  # Path where you saved your tokenizer

tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

def detect_sarcasm(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1)
    confidence, predicted_class = torch.max(probabilities, dim=1)
    is_sarcastic = bool(predicted_class.item())
    confidence_score = confidence.item()

    return {
        'is_sarcastic': is_sarcastic,
        'confidence': confidence_score
    }

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route('/check_sarcasm', methods=['POST'])
def check_sarcasm():
    user_input = request.form['text']
    result = detect_sarcasm(user_input)
    if result['is_sarcastic'] and result['confidence'] > 0.4:
        message = "Sarcastic. Keep it real bro!"
    else:
        message = "Not sarcastic. Wow, you really meant that, huh?"
    response = {
        'message': message,
        'confidence': round(result['confidence'], 2),
        'user_input': user_input
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)

