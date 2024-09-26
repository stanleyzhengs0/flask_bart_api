from flask import Flask, request, jsonify 
from transformers import BartTokenizer, BartForConditionalGeneration
from pymongo import MongoClient
import re

app = Flask(__name__)

#Load the Bart Model and Tokenizers
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')

#connect to MongoDB
client = MongoClient("mongodb+srv://stanleyzhengs:wirepint13sT@atlascluster.jjgxbm5.mongodb.net/?retryWrites=true&w=majority&appName=AtlasCluster")
db = client["test"]
collections = db['reviews']


@app.route('/summarize_reviews', methods=['GET'])
def summarize_reviews(): 

    cafe = request.args.get('cafe', '')
    
    # Create a regex pattern that ignores spaces and is case-insensitive
 
    

    reviews = collections.find({"cafeName": {"$regex": cafe, "$options": "i"}})

    #extracts the review descriptions 
    descriptions = " ".join([review['description'] for review in reviews if review['description'] != "No Desc."])

    #tokenize the input for BART
    inputs = tokenizer.encode("summarize: " + descriptions, return_tensors = "pt", max_length = 1024, truncation = True)

    #Generate the summary using BART
    summary_ids = model.generate(inputs, max_length=250, min_length = 80, length_penalty=2.0, num_beams=4, early_stopping = True )

    #decode the summary
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens = True)
    
    
    return jsonify({"summary": summary})





if __name__ == '__main__':
    app.run(debug=True)




# #Test Article to Summerize

# article = "The best known natural language processing tool is GPT-3, from OpenAI, which uses AI and statistics to predict the next word in a sentence based on the preceding words. NLP practitioners call tools like this “language models,” and they can be used for simple analytics tasks, such as classifying documents and analyzing the sentiment in blocks of text, as well as more advanced tasks, such as answering questions and summarizing reports. Language models are already reshaping traditional text analytics, but GPT-3 was an especially pivotal language model because, at 10x larger than any previous model upon release, it was the first large language model, which enabled it to perform even more advanced tasks like programming and solving high school–level math problems. The latest version, called InstructGPT, has been fine-tuned by humans to generate responses that are much better aligned with human values and user intentions, and Google’s latest model shows further impressive breakthroughs on language and reasoning.For businesses, the three areas where GPT-3 has appeared most promising are writing, coding, and discipline-specific reasoning. OpenAI, the Microsoft-funded creator of GPT-3, has developed a GPT-3-based language model intended to act as an assistant for programmers by generating code from natural language input. This tool, Codex, is already powering products like Copilot for Microsoft’s subsidiary GitHub and is capable of creating a basic video game simply by typing instructions. This transformative capability was already expected to change the nature of how programmers do their jobs, but models continue to improve — the latest from Google’s DeepMind AI lab, for example, demonstrates the critical thinking and logic skills necessary to outperform most humans in programming competitions.Models like GPT-3 are considered to be foundation models — an emerging AI research area — which also work for other types of data such as images and video. Foundation models can even be trained on multiple forms of data at the same time, like OpenAI’s DALL·E 2, which is trained on language and images to generate high-resolution renderings of imaginary scenes or objects simply from text prompts. Due to their potential to transform the nature of cognitive work, economists expect that foundation models may affect every part of the economy and could lead to increases in economic growth similar to the industrial revolution."

# inputs = tokenizer(article, return_tensors = "pt")

# sum_id = model.generate(inputs['input_ids'], max_length = 500, early_stopping=False)

# print([tokenizer.decode(g, skip_special_tokens=True,clean_up_tokenization_spaces=True) for g in sum_id])


