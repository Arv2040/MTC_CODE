from openai import AzureOpenAI
import os
from dotenv import load_dotenv
load_dotenv()

def gpt4oinit():
    return AzureOpenAI(
        api_key = os.getenv("api_key"),
        api_version = os.getenv("api_version"),
        azure_endpoint=os.getenv("azure_endpoint")
    )
    
def gpt4oresponse(client,prompt,images,text, max_tokens,skill,language):
    language_prompt = f"Respond in {language}. "
    full_prompt = language_prompt + prompt
    messages=[
        {"role": "system", "content": f"You are a helpful {skill}."},
        {"role": "user", "content": [
            {"type":"text", "text": prompt}
            ]}
    ]
    for text_data in text:
        messages[1]["content"].append({
            "type":"text",
            "text":text_data
        })
    
    for image_data in images:
        messages[1]["content"].append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{image_data}"
            }
        })

    response = client.chat.completions.create(
        model=os.getenv("deployment_name"),
        messages=messages,
        max_tokens=max_tokens
    )

    return response.choices[0].message.content

