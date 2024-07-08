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
    
def gpt4oresponse(client,prompt,max_tokens,skill):
    
    response = client.chat.completions.create(
            model=os.getenv("deployment_name"),
            messages=[
                {"role": "system", "content": f"You are a helpful {skill}."},
                {"role": "user", "content": f"{prompt}"}
            ],
            max_tokens=max_tokens
    )

    return response.choices[0].message.content