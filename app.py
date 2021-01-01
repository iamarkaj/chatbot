from flask import Flask, render_template, request
import os

from torch import tensor
from transformers import BlenderbotSmallTokenizer, BlenderbotForConditionalGeneration
mname = 'facebook/blenderbot-90M'
model = BlenderbotForConditionalGeneration.from_pretrained(mname)
tokenizer = BlenderbotSmallTokenizer.from_pretrained(mname)

def call(UTTERANCE):
    inputs = tokenizer([UTTERANCE], return_tensors='pt')
    new_inputs={'input_ids': inputs['input_ids'], 'attention_mask': inputs['attention_mask']}
    reply_ids = model.generate(**new_inputs)
    reply=[tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in reply_ids]
    return reply[0]


app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def predict():
    try:
        if request.method == 'POST':
            form = request.form
            result = []
            message = form['message']
            result.append(call(message))
            return render_template("home.html", result=result)
    except:
        pass
    return render_template("home.html")


if __name__ == '__main__':
    app.run()
