## Fine Tuning Llama 3 for custom PDF

For this example, I have taken a really simple PDF of just 10 pages. For simplicity, I'm ignoring all the textual and image based contents and just focusing on the textual content to fine tune the model.

### Loading the PDF

I have use `PyMuPDF` for loading and extracting the text contents. I have stored the text data in batches.

```python
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
```

### Dataset preperation

After extracting the text, I prepared a question and answer type dataset for this problem.

First I defined a schema using Pydantic base model to store the question answer pair.

```python
class Question(BaseModel):
    question: str = Field(description="The question")
    answer: str = Field(description="The question's corresponding answer")

class Questions(BaseModel):
    questions: list[Question] = Field(description="A list of question-answer pairs")
```

For this I used OpenAI's `gpt-4o-mini` and processed the text page by page to generate 20 question answer pairs for each pages at 5 different temperatures of LLM between 0.1 and 0.3.

```python
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
```

For the dataset generation I used the following client definition:

```python
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
```
