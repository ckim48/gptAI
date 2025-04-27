from flask import Flask, render_template, request
from aifeatures import *


app = Flask(__name__)

@app.route('/', methods = ["GET", "POST"])
def index():
    if request.method =="GET":
        age = 11
        subject = "photosynthesis"
        prompt = f"""
            Write a short, fun educational story for a {age}-year old student learning about {subject}.
            Use simple words and make it engaging. Format it with a title and divide the story into clearly labeled pages like this:
            
            Title:[Story Title]
            
            Page 1: [Content]
            Page 2: [Content]
            ...
            Keep the total story around 300-400 words.
            
             """
        result = generate_full_story_info(prompt)
        return render_template('index.html', summary=result['summary'], story=result['story'], sentiment=result['sentiment'])

    return render_template('index.html')


# 1. fintunig GPT
# 2. AI model which can calculates sentiment
# 3. Connect sentiment model with the finuted GPT model

if __name__=="__main__":
    app.run(debug=True)