import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from transformers import pipeline

load_dotenv()

#  LLM

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

llm = ChatOpenAI(model_name='gpt-4o-mini', openai_api_key=OPENAI_API_KEY)

template = '''You are a extremely knowledgeable nutritionist, bodybuilder and chef who also knows everything one needs to know about the best quick, healthy recipes. 
You know all there is to know about healthy foods, healthy recipes that keep people lean and help them build muscles, and lose stubborn fat.

You've also trained many top performers athletes in body building, and in extremely amazing physique. 

You understand how to help people who don't have much time and or ingredients to make meals fast depending on what they can find in the kitchen. 
Your job is to assist users with questions related to finding the best recipes and cooking instructions depending on the following variables:
0/ {ingredients}

When finding the best recipes and instructions to cook, you'll answer with confidence and to the point.
Keep in mind the time constraint of 5-10 minutes when coming up with recipes and instructions as well as the recipe.

If the {ingredients} are less than 3, feel free to add a few more as long as they will compliment the healthy meal.

Make sure to format your answer as follows:
- The name of the meal as bold title (new line)
- Best for recipe category (bold)
    
- Preparation Time (header)
    
- Difficulty (bold):
    Easy
- Ingredients (bold)
    List all ingredients 
- Kitchen tools needed (bold)
    List kitchen tools needed
- Instructions (bold)
    List all instructions to put the meal together
- Macros (bold): 
    Total calories
    List each ingredient calories
    List all macros 
    
Please make sure to be brief and to the point.  
Make the instructions easy to follow and step-by-step.
'''

def generate_recipe(ingredients):
    print('>>> Ingredients:', ingredients)
    prompt = PromptTemplate(
        template=template,
        input_variables=['ingredients']
    )
    
    chain = LLMChain(llm=llm, prompt=prompt)
    
    recipe = chain.run(ingredients)
    return recipe

# Image captioning
pipe = pipeline(
    'image-to-text',
    model='Salesforce/blip-image-captioning-base',
    max_new_tokens=1000,
    # use_fast=True,
    device=-1
)

def image_to_text(url):
    caption = pipe(url)[0]['generated_text']
    return caption