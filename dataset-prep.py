import fitz
import os
from openai import OpenAI
import instructor
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import time
from dotenv import load_dotenv

load_dotenv()

os.environ["OPENAI_API_KEY"] = str(os.environ.get("OPENAI_API_KEY"))
pdf_path = "ncert.pdf"

def extract_text_from_pdf(pdf_path):
    file = fitz.open(pdf_path)
    text = ""
    batches = []

    for page_num in range(0,10):
        page = file[page_num]
        text += f"Page {page_num + 1}\n"
        batch_text = page.get_text()
        text += batch_text  # Extract text from the page
        batches.append(batch_text)
        text += "\n"

    return text, batches

text,batches = extract_text_from_pdf(pdf_path)

client = instructor.from_openai(OpenAI())

class Question(BaseModel):
    question: str = Field(description="The question")
    answer: str = Field(description="The question's corresponding answer")

class Questions(BaseModel):
    questions: list[Question] = Field(description="A list of question-answer pairs")

def generate_dataset(data, temperature, df):

    context = data
    chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "system",
            "content": 
            
            """
            Given a context to you, your role is to generate 20 question answer pairs from that.
            
            Note: You should never use your general knowledge you should only generate question answer pairs
            based on the provided context. Both the question and answer should be based on the content in the
            context.
            """,
        },

        {
            "role" : "user",
            "content" : f"context : {context}"
        }
    ],
    temperature= temperature,
    model="gpt-4o-mini",
    response_model= Questions
    )

    temp_df = pd.DataFrame([pair.__dict__ for pair in chat_completion.questions])
    df = pd.concat([df, temp_df], ignore_index=True)
    
    return df

df = pd.DataFrame()
temperatures = np.linspace(0.1, 0.3, 5)

for t in temperatures:
    time.sleep(60)
    for index, batch  in enumerate(batches):
        print(f"\n\nGenerating questions for batch {index}, at temperature {t}")
        df = generate_dataset(batch, t, df)
        print("\n\n Current dataset : \n\n")
        print(df)

df.to_csv('data.csv', index=True)


