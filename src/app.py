# app.py (Flask Backend)

import os
import asyncio
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents import SearchClient
from azure.ai.formrecognizer import DocumentAnalysisClient
from openai import AzureOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain
from helpers.llm_helpers.langchainhelpers import createlangchainllm
from helpers.vector_helpers.getembedding import get_embedding
from helpers.Azure_helpers.blobhelp import getdatafromblob, getbloblist
from helpers.llm_helpers.gpt4o import gpt4oinit, gpt4oresponse
import azure.cognitiveservices.speech as speechsdk
import base64
import json

load_dotenv()

app = Flask(__name__)

# Initialize Azure clients and other necessary components
document_client = DocumentAnalysisClient(os.getenv("doc_endpoint"), AzureKeyCredential(os.getenv("doc_apikey")))
blob_list = getbloblist(os.getenv("CONTAINER_NAME_FRAUD"))
index_name = "fraudindex10"
search_client = SearchIndexClient(os.getenv("service_endpoint"), AzureKeyCredential(os.getenv("admin_key")))
search_client = SearchClient(endpoint=os.getenv("service_endpoint"), index_name=index_name, credential=AzureKeyCredential(os.getenv("admin_key")))

# ADA-002 MODEL FOR EMBEDDING
client = AzureOpenAI(
    azure_deployment=os.getenv("azure_openai_model_dep_name_em"),
    api_version=os.getenv("azure_openai_version_em"),
    azure_endpoint=os.getenv("ADA_ENDPOINT"),
    api_key=os.getenv("azure_openai_key"),
)

# Initialize Langchain components
llm = createlangchainllm()
memory = ConversationBufferMemory()
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True
)

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
            
            print("REACHED HERE")
            # Remove the temporary file
            if os.path.exists(temp_audio_path):
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

def categorize_fraud(analysis):
    openaiclient = gpt4oinit()
    
    sop = """
    Standard Operating Procedure (SOP) for Fraud Detection and Categorization

    1. Identity Fraud:
       - Definition: Unauthorized use of personal or business information for financial gain.
       - Key Indicators:
         a) Discrepancies in personal or business identification details
         b) Multiple accounts with similar details but different names
         c) Sudden changes in account holder information
       - Action Steps:
         1) Verify all identification documents
         2) Cross-reference with external databases
         3) Conduct enhanced due diligence on suspicious accounts

    2. Tax Fraud:
       - Definition: Intentional evasion of tax obligations or false tax claims.
       - Key Indicators:
         a) Inconsistencies between reported income and observed financial activity
         b) Large, unexplained deductions or credits
         c) Discrepancies in tax filings across different periods
       - Action Steps:
         1) Analyze tax returns and financial statements for anomalies
         2) Compare reported income with third-party information
         3) Investigate any substantial tax refunds or credits

    3. Transaction Fraud:
       - Definition: Unauthorized or deceptive financial transactions.
       - Key Indicators:
         a) Unusual patterns in transaction timing, frequency, or amounts
         b) Transactions with high-risk or sanctioned entities
         c) Circular transactions or unexplained fund movements
       - Action Steps:
         1) Implement real-time transaction monitoring
         2) Analyze transaction patterns for anomalies
         3) Investigate any high-risk or suspicious transactions immediately

    4. Operations Fraud:
       - Definition: Manipulation of business operations for fraudulent gains.
       - Key Indicators:
         a) Discrepancies between reported business activities and financial flows
         b) Unusual patterns in inventory, procurement, or sales data
         c) Inconsistencies in operational metrics and financial outcomes
       - Action Steps:
         1) Conduct regular audits of business operations
         2) Analyze operational data for inconsistencies
         3) Investigate any significant deviations from expected operational patterns

    5. Credit Fraud:
       - Definition: Obtaining credit through false pretenses or misuse of credit facilities.
       - Key Indicators:
         a) Rapid increase in credit utilization
         b) Discrepancies between reported income and credit behavior
         c) Unusual patterns in repayment behavior
       - Action Steps:
         1) Regularly review credit reports and utilization patterns
         2) Verify income and asset information for large credit requests
         3) Monitor repayment behavior for sudden changes or defaults

    General Guidelines:
    - Maintain strict confidentiality throughout the investigation process
    - Document all findings meticulously
    - Escalate confirmed fraud cases to the appropriate authorities
    - Regularly update fraud detection models and procedures based on new trends and patterns
    """

    categorization_prompt = f"""
    Based on the following fraud analysis and the provided Standard Operating Procedure (SOP), 
    categorize the potential fraud into one or more of these categories: 
    Identity Fraud, Tax Fraud, Transaction Fraud, Operations Fraud, or Credit Fraud.

    Provide a detailed explanation for each category you select, referencing specific points 
    from the analysis and how they align with the SOP guidelines.

    Analysis:
    {analysis}

    SOP:
    {sop}

    Your response should be structured as follows:
    1. Primary Fraud Category: [Category Name]
       Explanation: [Detailed justification]
    2. Secondary Fraud Category (if applicable): [Category Name]
       Explanation: [Detailed justification]
    3. Other Potential Categories: [List any other categories that might apply]
       Explanation: [Brief justification for each]
    4. Recommended Next Steps: [Based on the SOP, what actions should be taken next]
    """

    categorization = gpt4oresponse(openaiclient, categorization_prompt, [], [], 2000, "fraud detection expert")
    return categorization

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

    # Add fraud categorization
    fraud_categorization = categorize_fraud(overall_analysis)

    # Combine all analyses into the final response
    final_response = f"""Overall Analysis:
    {overall_analysis}
    Fraud Categorization:
    {fraud_categorization}

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

@app.route('/process', methods=['POST'])
def process():
    data = request.json
    query = data.get('query')
    if not query:
        return jsonify({"error": "No query provided"}), 400

    response = process_data(query)
    print("Response from process data: ", response)
    return jsonify({"response": response})

@app.route('/follow_up', methods=['POST'])
def follow_up():
    data = request.json
    question = data.get('question')
    company_data = data.get('company_data')
    if not question or not company_data:
        return jsonify({"error": "Missing question or company data"}), 400

    context = f"""
    Based on the previous analysis and the following company data:
    {json.dumps(company_data, indent=2)}
    
    Please answer the following question:
    {question}
    """
    response = conversation.predict(input=context)
    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)