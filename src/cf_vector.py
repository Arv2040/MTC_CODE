from azure.storage.blob import BlobServiceClient
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
import os
from dotenv import load_dotenv
load_dotenv()
from azure.core.credentials import AzureKeyCredential
from openai import AzureOpenAI
from azure.search.documents.indexes.models import (
    SearchFieldDataType,
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
from helpers.Azure_helpers.blobhelp import getdatafromblob, getbloblist, uploaddata


# CLIENT FOR EMBEDDING - ADA - 002
client = AzureOpenAI(
    azure_deployment=os.getenv("azure_openai_model_dep_name_em"),
    api_version=os.getenv("azure_openai_version_em"),
    azure_endpoint=os.getenv("ADA_ENDPOINT"),
    api_key=os.getenv("azure_openai_key"),
)

#getting the list of blobbs from the container
blob_list = getbloblist(os.getenv("CONTAINER_NAME_FRAUD"))


# Creating document intelligence instance

document_client = DocumentAnalysisClient(os.getenv("doc_endpoint"), AzureKeyCredential(os.getenv("doc_apikey")))

# Creating a list of documents
document_list = []
for blob in blob_list:
    if blob.name == 'llminputdata.json' or blob.name == 'corporate_dataset.json':
        continue
    
    pdf_content = getdatafromblob(blob.name, os.getenv("CONTAINER_NAME_FRAUD"))
    poller = document_client.begin_analyze_document("prebuilt-document", pdf_content)
    result = poller.result()
    full_text = ""
    for page in result.pages:
        for line in page.lines:
            full_text += line.content + "\n"
    document_list.append(full_text)

data = getdatafromblob("corporate_dataset.json", os.getenv("CONTAINER_NAME_FRAUD"))
data = json.loads(data)

#mapped structured and unstructured data

for i in range(1,11,1):
    for document in document_list:
        if f"CompanyID = {str(i)}" in document:
            for customer in data:
                customer['document_text'] = "a"
                if(customer['CompanyID'] == str(i)):
                    customer['document_text'] += document


dataitem = data[0]
data_dict = dict(dataitem)
key_list = list(data_dict.keys())

#CREATING THE INDEX
index_name_fraud = os.getenv("corporate-index9")
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
    response_list = client.embeddings.create(input=names_list, model = os.getenv("azure_openai_em_name"))
    embedding_list = [item.embedding for item in response_list.data]
    names_lists.append(names_list)
    response_lists.append(response_list)
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
index = SearchIndex(name="corporate-index9", fields=fields, vector_search=vector_search)
result = search_client.create_or_update_index(index)
print(f'{result.name} created')

# Getting the mapped and embedded data from the blob
prefinal_data = getdatafromblob('llminputdata.json', os.getenv("CONTAINER_NAME_FRAUD"))
json_data = json.loads(prefinal_data)

# Upload the documents to the vector store
search_client = SearchClient(endpoint=os.getenv("service_endpoint"), index_name="corporate-index9", credential=AzureKeyCredential(os.getenv("admin_key")))
results = search_client.upload_documents(json_data)
print(f"Uploaded {len(json_data)} documents")

'''
# TESTING IF I CAN READ THE CONTENTS OF THE DATASET FOLDER
import os
from PIL import Image

main_folder_path = r"C:\Users\Arush\Desktop\MTC_CODE\Datasets"

# Walk through the directory
for root, dirs, files in os.walk(main_folder_path):
    print(f"Folder: {root}")
    
    image_file = None
    text_file = None
    
    # Identify image and text files
    for file in files:
        file_path = os.path.join(root, file)
        file_extension = os.path.splitext(file)[1].lower()
        
        if file_extension in ['.jpg', '.jpeg', '.png', '.gif']:
            image_file = file_path
        elif file_extension == '.txt':
            text_file = file_path
    
    # Process image file
    if image_file:
        print(f"  Image: {os.path.basename(image_file)}")
        try:
            with Image.open(image_file) as img:
                width, height = img.size
                print(f"    Dimensions: {width}x{height}")
                print(f"    Format: {img.format}")
        except Exception as e:
            print(f"    Error processing image: {str(e)}")
    
    # Process text file
    if text_file:
        print(f"  Text: {os.path.basename(text_file)}")
        try:
            with open(text_file, 'r', encoding='utf-8') as f:
                content = f.read()
                print(f"    Content: {content}...")
        except Exception as e:
            print(f"    Error reading text file: {str(e)}")

print("Done processing all folders and files.")
'''