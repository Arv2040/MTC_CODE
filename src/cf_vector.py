from azure.storage.blob import BlobServiceClient
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from sklearn.decomposition import PCA
import azure.cognitiveservices.speech as speechsdk

import os
import base64
from dotenv import load_dotenv
import numpy as np
load_dotenv()
from azure.core.credentials import AzureKeyCredential
from openai import AzureOpenAI
from azure.search.documents.indexes.models import (
    SearchFieldDataType,
    SimpleField,
    SearchableField,
    SearchField,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchProfile,
    SemanticConfiguration,
    SemanticPrioritizedFields,
    SemanticField,
    SemanticSearch,
    SearchIndex
)
import json
from helpers.Azure_helpers.blobhelp import getdatafromblob, getbloblist, uploaddata, createcontainer

# CLIENT FOR EMBEDDING - ADA - 002
client = AzureOpenAI(
    azure_deployment=os.getenv("azure_openai_model_dep_name_em"),
    api_version=os.getenv("azure_openai_version_em"),
    azure_endpoint=os.getenv("ADA_ENDPOINT"),
    api_key=os.getenv("azure_openai_key"),
)

document_client = DocumentAnalysisClient(os.getenv("doc_endpoint"), AzureKeyCredential(os.getenv("doc_apikey")))
main_folder_path = r"/Users/mihir/Desktop/MTC_Code/MTC_CODE/Datasets"
fraud_detection_folder = os.path.join(main_folder_path, "Fraud_Detection")
fraud_directory = next(os.walk(fraud_detection_folder))
current_directory, directories, files = fraud_directory
entries = os.listdir(fraud_detection_folder)
d = getdatafromblob('Structured_Data.json','corporatefraud')
d = json.loads(d)
document_text_final = []
image_final = []
text_final = []
audio_transcript_final = []

def transcribe_audio(audio_file_path):
    speech_config = speechsdk.SpeechConfig(subscription=os.getenv("speech_key"), region=os.getenv("speech_region"))
    audio_config = speechsdk.audio.AudioConfig(filename=audio_file_path)
    
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)
    
    result = speech_recognizer.recognize_once_async().get()
    
    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        return result.text
    elif result.reason == speechsdk.ResultReason.NoMatch:
        return "No speech could be recognized"
    elif result.reason == speechsdk.ResultReason.Canceled:
        return "Speech recognition canceled"

for dir in entries:
    sub_dir_path = os.path.join(fraud_detection_folder, dir)
    print(sub_dir_path)
    document_text_list = []
    image_list = []
    text_list = []
    audio_transcript_list = []
    for root, dirs, files in os.walk(sub_dir_path):
        for file in files:
            file_path = os.path.join(sub_dir_path, file)
            file_extension = os.path.splitext(file)[1].lower()
            
            if file_extension in ['.mp3', '.wav']:
                # Process audio file
                transcript = transcribe_audio(file_path)
                audio_transcript_list.append(transcript)
                # Upload transcript to blob storage
                uploaddata(f"{file}_transcript.txt", dir, transcript.encode('utf-8'))
            elif file_extension in ['.jpg', '.jpeg', '.png', '.gif']:
                with open(file_path, "rb") as image_file:
                    image_content = image_file.read()
                    base64_image = base64.b64encode(image_content).decode('utf-8')
                    image_list.append(base64_image)
            elif file_extension == '.pdf':
                with open(file_path, "rb") as pdf_file:
                    pdf_content = pdf_file.read()
                    poller = document_client.begin_analyze_document("prebuilt-document", pdf_content)
                    result = poller.result()
                    full_text = "\n".join([line.content for page in result.pages for line in page.lines])
                    document_text_list.append(full_text)
            elif file_extension == '.txt':
                with open(file_path, "r") as text_file:
                    text_content = text_file.read()
                    text_list.append(text_content)
            
            with open(file_path, "rb") as data:
                uploaddata(file, dir, data)
        break
    document_text_final.append(document_text_list)
    image_final.append(image_list)
    text_final.append(text_list)
    audio_transcript_final.append(audio_transcript_list)

for i in range(len(d)):
    element = d[i]
    element['document_text'] = '|||'.join(document_text_final[i])
    element['images'] = '|||'.join(image_final[i])
    element['txt_text'] = '|||'.join(text_final[i])
    element['audio_transcript'] = '|||'.join(audio_transcript_final[i])

data = d
print("done")

dataitem = data[0]
data_dict = dict(dataitem)
key_list = list(data_dict.keys())

#CREATING THE INDEX
index_name_fraud = "fraudindex10"
search_client = SearchIndexClient(os.getenv("service_endpoint"),AzureKeyCredential(os.getenv("admin_key")))
key_list = list(data_dict.keys())

fields = [
    SearchableField(name=f"{str(key_list[0])}", type=SearchFieldDataType.String, key=True, sortable=True, filterable=True, facetable=True),
]
for i in range(1, len(key_list),1):
    element = SearchableField(name=f"{str(key_list[i])}", type=SearchFieldDataType.String, sortable=True, filterable=True, facetable=True)
    fields.append(element)
for i in range(len(key_list)):
    element = SearchField(name=f"{str(key_list[i])}_Vector", type=SearchFieldDataType.Collection(SearchFieldDataType.Single), searchable=True, vector_search_dimensions=1536, vector_search_profile_name="myHnswProfile")
    fields.append(element)

names_lists = []
response_lists = []
embedding_lists = []

for i in range(len(key_list)):
    names_list = [str(dataitem[f'{key_list[i]}']) for dataitem in data]
    response = client.embeddings.create(input=names_list, model = os.getenv("azure_openai_em_name"))
    embedding_list = [item.embedding for item in response.data]
    embedding_lists.append(embedding_list)

for i, dataitem in enumerate(data):
    for j, key in enumerate(key_list):
        dataitem[f'{key}_Vector'] = embedding_lists[j][i]

# Open the local file and upload its contents
data_string = json.dumps(data)
uploaddata('llminputdata.json', os.getenv("CONTAINER_NAME_FRAUD"), data_string)

#defining the vector search algorithm
vector_search = VectorSearch(
    algorithms=[
        HnswAlgorithmConfiguration(
            name="myHnsw"
        )
    ],
    profiles=[
        VectorSearchProfile(
            name="myHnswProfile",
            algorithm_configuration_name="myHnsw",
        )
    ]
)

# Create the search index and defining the algorithm we previously created
index = SearchIndex(name="fraudindex10", fields=fields, vector_search=vector_search)
result = search_client.create_or_update_index(index)
print(f'{result.name} created')

# Getting the mapped and embedded data from the blob
prefinal_data = getdatafromblob('llminputdata.json', os.getenv("CONTAINER_NAME_FRAUD"))
json_data = json.loads(prefinal_data)

# Upload the documents to the vector store
search_client = SearchClient(endpoint=os.getenv("service_endpoint"), index_name="fraudindex10", credential=AzureKeyCredential(os.getenv("admin_key")))
results = search_client.upload_documents(json_data)
if(results):
    print("done")
else:
    print("No")