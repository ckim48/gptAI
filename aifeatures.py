from openai import OpenAI
from transformers import BertTokenizer, pipeline, BertForSequenceClassification, BartForConditionalGeneration, BartTokenizer

tokenizer = BertTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
model = BertForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
tokenizer_summarization = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
model_summarization = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')

def story_generation(prompt):
  response = client.chat.completions.create(
      model = "gpt-4o",
      messages = [
          {'role':'user', 'content':prompt}
      ],
      temperature = 0
  )
  return response.choices[0].message.content

def summarize_story(text):

    inputs = tokenizer_summarization(text, max_length = 1024, return_tensors='pt')
    summary_ids = model_summarization.generate(
        inputs["input_ids"],
        max_length = 100,
        min_length = 30,
        length_penalty = 2.0,
        num_beams = 4,
        early_stopping = True
    )
    summary = tokenizer_summarization.decode(
        summary_ids[0],
        skip_special_tokens=True
    )
    return summary


def analyze_sentiment(text):
    sentiment_classifier = pipeline("sentiment-analysis",model=model,tokenizer=tokenizer)
    result = sentiment_classifier(text)[0]
    if result['label'] in ["4 stars", "5 stars"]:
        return 'POSITIVE'
    elif result['label'] in ["1 star", "2 stars"]:
        return 'NEGATIVE'
    else:
        return 'NEUTRAL'

# print(analyze_sentiment("The first little pig was very lazy. He didn't want to work at all and he built his house out of straw. The second little pig worked a little bit harder but he was somewhat lazy too and he built his house out of sticks. Then, they sang and danced and played together the rest of the day."))


def generate_full_story_info(prompt):
    story = story_generation(prompt)
    sentiment = analyze_sentiment(story)
    summary = summarize_story(story)
    return {
        "story": story,
        "sentiment": sentiment,
        "summary": summary
    }
# 1 stars --> very negative
# 2 stars
# 3 stars
# 4 stars
# 5 stars  --> very positive