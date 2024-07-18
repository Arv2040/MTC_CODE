import os
import asyncio
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents import SearchClient
from azure.ai.formrecognizer import AnalyzeResult, DocumentAnalysisClient

import streamlit as st
from azure.core.credentials import AzureKeyCredential
from openai import AzureOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain
from helpers.llm_helpers.langchainhelpers import createlangchainllm
from helpers.vector_helpers.getembedding import get_embedding
from helpers.input_helpers.speech import from_mic
from helpers.Azure_helpers.blobhelp import getdatafromblob, getbloblist, uploaddata
from helpers.llm_helpers.gpt4o import gpt4oinit, gpt4oresponse
import azure.cognitiveservices.speech as speechsdk
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
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False

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
llm = createlangchainllm()
if st.session_state.conversation is None:
    memory = ConversationBufferMemory()
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
        st.write("CompanyID:", end=" ")
        company_id = st.selectbox("", options=["100001", "100002", "100003", "100004", "100005"], label_visibility="collapsed")
        text_bool = st.button("CHAT WITH COPILOT")

        if text_bool:
            query = company_id

async def process_file(blob, containername):
    if '.jpg' in blob.name or '.jpeg' in blob.name or '.png' in blob.name:
        image_content = await asyncio.to_thread(getdatafromblob, blob.name, containername)
        base64_image = base64.b64encode(image_content).decode('utf-8')
        return ('image', base64_image)
    elif '.pdf' in blob.name:
        pdf_content = await asyncio.to_thread(getdatafromblob, blob.name, containername)
        poller = await asyncio.to_thread(document_client.begin_analyze_document, "prebuilt-document", pdf_content)
        result = await asyncio.to_thread(poller.result)
        full_text = "\n".join([line.content for page in result.pages for line in page.lines])
        return ('document', full_text)
    elif '.wav' in blob.name:
        try:
            audio_content = await asyncio.to_thread(getdatafromblob, blob.name, containername)
            speech_config = speechsdk.SpeechConfig(subscription=os.getenv("speech_key"), region=os.getenv("speech_region"))
            
            # Save audio content to a temporary file
            temp_audio_path = f"temp_{blob.name}"
            with open(temp_audio_path, "wb") as temp_file:
                temp_file.write(audio_content)
            
            audio_config = speechsdk.audio.AudioConfig(filename=temp_audio_path)
            speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)
            
            result = await asyncio.to_thread(speech_recognizer.recognize_once_async().get)
            
            # Remove the temporary file
            os.remove(temp_audio_path)
            
            if result.reason == speechsdk.ResultReason.RecognizedSpeech:
                return ('audio', result.text)
            elif result.reason == speechsdk.ResultReason.NoMatch:
                return ('audio', "No speech could be recognized")
            elif result.reason == speechsdk.ResultReason.Canceled:
                return ('audio', "Speech recognition canceled")
        except RuntimeError as e:
            error_message = str(e)
            if "SPXERR_INVALID_HEADER" in error_message:
                return ('audio', f"Error: Invalid WAV file header in {blob.name}. Please check the file format.")
            else:
                return ('audio', f"Error processing audio file {blob.name}: {error_message}")
    else:
        text_content = await asyncio.to_thread(getdatafromblob, blob.name, containername)
        text_content = text_content.decode('utf-8')
        return ('text', text_content)

async def process_files(blob_list, containername):
    tasks = [process_file(blob, containername) for blob in blob_list]
    results = await asyncio.gather(*tasks)
    document_text_list, image_list, text_list, audio_transcript_list = [], [], [], []
    for file_type, content in results:
        if file_type == 'document':
            document_text_list.append(content)
        elif file_type == 'image':
            image_list.append(content)
        elif file_type == 'text':
            text_list.append(content)
        elif file_type == 'audio':
            audio_transcript_list.append(content)
    return document_text_list, image_list, text_list, audio_transcript_list

def process_data(query):
    content = get_embedding(query, "CompanyID_Vector", client)
    
    select = [
        "CompanyID", "CompanyName", "Date", "Debit_Credit", "Amount",
        "CompanyAccount", "TransactionDescription", "FinalBalance",
        "TransactionID", "MerchantFirmName", "MerchantID", "Collateral"
    ]

    results = search_client.search(
        search_text=None,
        vector_queries=[content],
        select=select,
    )
    
    result = next(results)
    
    # Create a dictionary with all selected fields
    company_data = {field: result.get(field) for field in select}
    
    containername = result['CompanyID']
    blob_list = getbloblist(containername)
    
    document_text_list, image_list, text_list, audio_transcript_list = asyncio.run(process_files(blob_list, containername))
    
    # Create a text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=5000,
        chunk_overlap=100,
        length_function=len,
    )

    # Split the documents
    docs = [Document(page_content=text) for text in document_text_list]
    split_docs = text_splitter.split_documents(docs)

    # Initialize the summarization chain
    chain = load_summarize_chain(llm, chain_type="map_reduce")

    # Process chunks and summarize
    document_summary = chain.run(split_docs)

    # Process images separately if needed
    openaiclient = gpt4oinit()
    image_analysis = gpt4oresponse(openaiclient, "Analyze these images for potential fraud indicators and validate against the company data", image_list, [json.dumps(company_data)], 2000, "fraud detection expert")

    # Process text data
    text_analysis = gpt4oresponse(openaiclient, "Analyze this text data for potential fraud indicators and validate against the company data", [], text_list + [json.dumps(company_data)], 2000, "fraud detection expert")

    # Process audio transcripts
    audio_analysis_prompt = f"""
    Analyze these call recordings (transcripts provided) in the context of potential fraudulent behavior by the company (CompanyID: {company_data['CompanyID']}).
    Consider the following:
    1. Does the company representative make any suspicious claims or offers?
    2. Are there any inconsistencies between what's said in the calls and the company's official data?
    3. Are there any high-pressure sales tactics or attempts to mislead customers?
    4. Is there any mention of practices that could be considered unethical or illegal?
    5. Do the call recordings provide any evidence that supports or contradicts the company's reported financial activities?

    Provide a detailed analysis focusing on how these call recordings might indicate fraudulent behavior by the company, not the caller.

    Company Data for reference: {json.dumps(company_data, indent=2)}
    """

    audio_analysis = gpt4oresponse(openaiclient, audio_analysis_prompt, [], audio_transcript_list, 2000, "fraud detection expert")

    # Generate overall analysis
    overall_analysis_prompt = f"""
    As a fraud detection expert, provide a comprehensive analysis based on the following information:

    1. Company Data: {json.dumps(company_data, indent=2)}
    2. Document Summary: {document_summary}
    3. Image Analysis: {image_analysis}
    4. Text Analysis: {text_analysis}
    5. Audio Analysis (Call Recordings): {audio_analysis}

    Please include:
    1. A summary of the key points from the company data
    2. Validation of the company data against the information found in documents, images, and especially the call recordings
    3. Identification of any discrepancies or potential fraud indicators, with particular attention to evidence from the call recordings
    4. An overall risk assessment, considering all sources of information but weighing heavily on the call recording analysis
    5. Recommendations for further investigation, if necessary

    Remember, the call recordings should be treated as a potentially critical piece of evidence in determining if the company is engaging in fraudulent activities.

    Provide your analysis in a clear, structured format.
    """

    overall_analysis = gpt4oresponse(openaiclient, overall_analysis_prompt, [], [], 3000, "fraud detection expert")

    # Combine all analyses into the final response
    final_response = f"""Overall Analysis:
    {overall_analysis}

    Detailed Company Data:
    {json.dumps(company_data, indent=2)}

    Document Summary:
    {document_summary}

    Image Analysis:
    {image_analysis}

    Text Analysis:
    {text_analysis}

    Call Recording Analysis:
    {audio_analysis}

    Note: The call recording analysis is a critical component in assessing potential fraudulent behavior by the company."""
    
    return final_response

if query and not st.session_state.processing_complete:
    st.write(f"Your query is: {query}")
    
    with st.spinner("Processing..."):
        # Process data
        st.session_state.initial_response = process_data(query)
        st.session_state.processing_complete = True

    # Add the initial interaction to Langchain memory
    st.session_state.conversation.predict(input=f"User: {query}\nAI: {st.session_state.initial_response}")

# Display the initial response if it exists and processing is complete
if st.session_state.initial_response and st.session_state.processing_complete:
    st.write("Final Report:")
    st.write(st.session_state.initial_response)

# Option for follow-up questions using Langchain
st.write("You can ask follow-up questions about the report:")
follow_up = st.text_input("Follow-up question:", key="follow_up")
if st.button("Ask Follow-up"):
    follow_up_response = st.session_state.conversation.predict(input=follow_up)
    st.session_state.follow_up_response = follow_up_response
    st.experimental_rerun()

# Display the follow-up response if it exists
if st.session_state.follow_up_response:
    st.write("Follow-up Response:")
    st.write(st.session_state.follow_up_response)