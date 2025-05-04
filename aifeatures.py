from openai import OpenAI
from transformers import BertTokenizer, pipeline, BertForSequenceClassification, BartForConditionalGeneration, BartTokenizer
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from wordfreq import zipf_frequency
import textstat

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
    text = text[:1000]
    sentiment_classifier = pipeline("sentiment-analysis",model=model,tokenizer=tokenizer)
    result = sentiment_classifier(text)[0]
    if result['label'] in ["4 stars", "5 stars"]:
        return 'POSITIVE'
    elif result['label'] in ["1 star", "2 stars"]:
        return 'NEGATIVE'
    else:
        return 'NEUTRAL'

# print(analyze_sentiment("The first little pig was very lazy. He didn't want to work at all and he built his house out of straw. The second little pig worked a little bit harder but he was somewhat lazy too and he built his house out of sticks. Then, they sang and danced and played together the rest of the day."))

# Vocabulary Level Analyzer: This feature can highlight difficult words
# in a story and provide simpler alternatives or definitions.
def get_definition(word):
    synsets = wordnet.synsets(word)
    if synsets:
        return synsets[0].definition()
    return "No definition found."

# It analyzes a piece of text to find less common (difficult) words
# 'dog' --> common words, it has zipf score around 6.0
def vocab_level_analysis(text, threshold=4.0):
    tokens = word_tokenize(text)
    seen = set()
    results = []
    for token in tokens:
        word = token.lower()
        if word.isalpha() and word not in seen:
            freq = zipf_frequency(word,'en')
            if freq < threshold: # checks the difficulty of the words
                word_definition = get_definition(word)
                seen.add(word)
                if word_definition != 'No definition found.':
                    results.append({
                        "word": word,
                        "frequency": round(freq,2),
                        "definition": word_definition
                    })
    return results

# 2 metrics for readability and reading ease
#1 . Flesch-kincaid Grade Level: tells us what U.S school grade is needed to understand the given text
# example: score of 3.5 --> 3rd to 4th grade student can understand
#         score = 9 --> 9th grade reading level
# 2. Flesch Reading Ease 0-100 --> higher is easier to read.
#  70 ~ 5 gth grade

def readability_score(text):
    return {
        "grade_level": textstat.flesch_kincaid_grade(text),
        "readability": textstat.flesch_reading_ease(text)
    }

def generate_full_story_info(prompt):
    story = story_generation(prompt)
    sentiment = analyze_sentiment(story)
    summary = summarize_story(story)
    return {
        "story": story,
        "sentiment": sentiment,
        "summary": summary
    }

story = """
Title: The Great Leaf Adventure Page 1: The Secret of the Green Kingdom Once upon a time, in a sunny forest, there was a little leaf named Lila. Lila lived on a big, strong tree called Oakley. Every day, Lila would look up at the bright sun and feel its warm rays. But Lila had a secret job that made her very special. She was part of the Green Kingdom, where all the leaves worked together to make food for their tree. Page 2: The Magic Recipe One morning, Lila's friend, Benny the Bee, buzzed by and asked, "Lila, how do you make food for Oakley?" Lila giggled and said, "It's a magical recipe called photosynthesis!" Benny's eyes widened with curiosity. "Photo-what?" he asked. Lila explained, "Photosynthesis! It's how we leaves turn sunlight, air, and water into food." Page 3: The Sunlight Dance Lila continued, "First, we catch sunlight with our green color, called chlorophyll. It's like a dance party with the sun! The sunlight gives us energy to start the magic." Benny buzzed with excitement. "Wow, a dance party with the sun! That sounds fun!" Page 4: The Airy Hug "Next," Lila said, "we take a big, airy hug from the air. We breathe in a special gas called carbon dioxide. It's like a big breath of fresh air!" Benny nodded, "I love fresh air! So, what's next?" Page 5: The Raindrop Sip "Finally," Lila explained, "we sip water from the ground through Oakley's roots. It's like drinking a cool glass of lemonade on a hot day!" Benny clapped his wings. "Sunlight, air, and water! That's the magic recipe!" Page 6: The Food Factory Lila smiled, "With all these ingredients, we make a special sugar called glucose. It's like a yummy snack for Oakley, and it helps him grow big and strong!" Benny buzzed happily, "That's amazing, Lila! You're like a tiny chef in the Green Kingdom!" Page 7: The Happy Forest Thanks to Lila and her leaf friends, Oakley and all the trees in the forest stayed healthy and happy. Benny flew off to tell all his bee friends about the magic of photosynthesis. And every day, Lila danced with the sun, hugged the air, and sipped the rain, knowing she was part of something truly wonderful. The End.
"""
print(vocab_level_analysis(story))
# print(readability_score(story))