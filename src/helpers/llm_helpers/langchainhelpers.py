from langchain.chat_models import AzureChatOpenAI
import os
from dotenv import load_dotenv
load_dotenv()
def createlangchainllm():
    return AzureChatOpenAI(
        api_key = os.getenv("api_key"),
        api_version = os.getenv("api_version"),
        azure_endpoint=os.getenv("azure_endpoint"),
        deployment_name = os.getenv("deployment_name")
    )