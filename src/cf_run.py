import os
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents import SearchClient
from azure.ai.formrecognizer import AnalyzeResult

import streamlit as st
from azure.core.credentials import AzureKeyCredential
from openai import AzureOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from helpers.llm_helpers.langchainhelpers import createlangchainllm
from helpers.vector_helpers.getembedding import get_embedding
from helpers.input_helpers.speech import from_mic
from helpers.Azure_helpers.blobhelp import getdatafromblob,getbloblist,uploaddata
from helpers.llm_helpers.gpt4o import gpt4oinit,gpt4oresponse
from azure.ai.formrecognizer import DocumentAnalysisClient
import base64
import json


load_dotenv()
document_client = DocumentAnalysisClient(os.getenv("doc_endpoint"), AzureKeyCredential(os.getenv("doc_apikey")))

st.set_page_config(layout="wide")
st.header("CREDISHIELD: THE FRAUD DETECTION COPILOT")

# Initialize session state
if 'initial_response' not in st.session_state:
    st.session_state.initial_response = ""
if 'conversation' not in st.session_state:
    st.session_state.conversation = None
if 'follow_up_response' not in st.session_state:
    st.session_state.follow_up_response = ""



col1, col2 = st.columns(2)

blob_list = getbloblist(os.getenv("CONTAINER_NAME_FRAUD"))

index_name = "fraudindex10"
search_client = SearchIndexClient(os.getenv("service_endpoint"), AzureKeyCredential(os.getenv("admin_key")))



# ADA-002 MODEL FOR EMBEDDING
client = AzureOpenAI(
    azure_deployment=os.getenv("azure_openai_model_dep_name_em"),
    api_version=os.getenv("azure_openai_version_em"),
    azure_endpoint=os.getenv("ADA_ENDPOINT"),
    api_key=os.getenv("azure_openai_key"),
)



#MAIN CODE
search_client = SearchClient(endpoint=os.getenv("service_endpoint"), index_name=index_name, credential=AzureKeyCredential(os.getenv("admin_key")))

# Function to vectorize the prompt
field_string = "CompanyID_Vector"

# Initialize Langchain components
if st.session_state.conversation is None:
    memory = ConversationBufferMemory()
    llm = createlangchainllm()
    st.session_state.conversation = ConversationChain(
        llm=llm,
        memory=memory,
        verbose=True
    )

with col1:
    speech_bool = st.button("TALK TO COPILOT")

query = ""

if speech_bool:
    
    st.write("Listening...")
    query = from_mic()
else:
    with col2:
        input = st.text_input("Enter your query")
        text_bool = st.button("CHAT WITH COPILOT")

        if text_bool:
            query = input

if query:
    st.write(f"Your query is: {query}")
    prompt = f"This is the customer id {query}, please return it to be in this format - CompanyID: i, where i can be the number provided by the query.Do not give anything else in the response, just give CompanyID: i."
    
    content = get_embedding(query,"CompanyID_Vector",client)
    

    
    select = [
    "CompanyID",
    "CompanyName",
    "Date",
    "Debit_Credit",
    "Amount",
    "CompanyAccount",
    "TransactionDescription",
    "FinalBalance",
    "TransactionID",
    "MerchantFirmName",
    "MerchantID",
    "Collateral"
]


    results = search_client.search(
        search_text=None,
        vector_queries=[content],
        select=select,
    )
    
    result = next(results)
    
    containername = result['CompanyID']
    blob_list = getbloblist(containername)
    document_text_list = []
    image_list = []
    text_list = []
    
   
    for blob in blob_list:
        if '.jpg' in blob.name or '.jpeg' in blob.name or '.png' in blob.name:
            image_content = getdatafromblob(blob.name,containername)
            base64_image = base64.b64encode(image_content).decode('utf-8')
            image_list.append(base64_image)

        elif '.pdf' in blob.name:
            
        
            pdf_content = getdatafromblob(blob.name, containername)
            poller = document_client.begin_analyze_document("prebuilt-document", pdf_content)
            result = poller.result()
            full_text = ""
            for page in result.pages:
                for line in page.lines:
                    full_text += line.content + "\n"
            document_text_list.append(full_text)
        else:
            text_content = getdatafromblob(blob.name,containername)
            text_content = text_content.decode('utf-8')
            text_list.append(text_content)
    
    st.write(f"Number of images processed: {len(image_list)}")
    if image_list:
        st.write("Sample of first image data (first 100 characters):")
        st.write(image_list[0][:20] + "...")
    
    context = {
        "result":result,
        "document":document_text_list,
        "text":text_list,
        "image":image_list
    }
    st.write(context)
    
    
    with st.spinner("ANALYSING THE DATA AND GENERATING REPORT"):
       
      
        prompt1 = f"Analyse {image_list} and {document_text_list}"
        
        openaiclient = gpt4oinit()
        response = gpt4oresponse(openaiclient,prompt1,image_list, document_text_list, 4000,"fraud detection expert")

        st.session_state.initial_response = response
        # st.write(st.session_state.initial_response)

        # Add the initial interaction to Langchain memory
        st.session_state.conversation.predict(input=f"User: {query}\nAI: {st.session_state.initial_response}")

# Display the initial response if it exists
if st.session_state.initial_response:
    st.write("Initial Report:")
    st.write(st.session_state.initial_response)

# Option for follow-up questions using Langchain
st.write("You can ask follow-up questions about the report:")
follow_up = st.text_input("Follow-up question:", key="follow_up")
if st.button("Ask Follow-up"):
    follow_up_response = st.session_state.conversation.predict(input=follow_up)
    st.session_state.follow_up_response = follow_up_response
    st.rerun()

# Display the follow-up response if it exists
if st.session_state.follow_up_response:
    st.write("Follow-up Response:")
    st.write(st.session_state.follow_up_response)
        
