from azure.storage.blob import BlobServiceClient
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents import SearchClient
import os
from dotenv import load_dotenv
load_dotenv()
from azure.core.credentials import AzureKeyCredential
from openai import AzureOpenAI
from helpers.Azure_helpers.blobhelp import getdatafromblob,getbloblist,uploaddata
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





#CLIENT FOR EMBEDDING - ADA-002
client = AzureOpenAI(
    azure_deployment= os.getenv("azure_openai_model_dep_name_em"),
    api_version=os.getenv("azure_openai_version_em"),
    azure_endpoint=os.getenv("ADA_ENDPOINT"),
    api_key=os.getenv("azure_openai_key"),
   
)




#getting the list of blobs from the container
blob_list = getbloblist(os.getenv("CONTAINER_NAME"))



#creating document intelligence instance
document_client = DocumentAnalysisClient(os.getenv("doc_endpoint"),AzureKeyCredential(os.getenv("doc_apikey")))


#extracting the text from the documents in the blob
document_list = []

for blob in blob_list:
    if blob.name == 'llminputdata.json' or blob.name == "llminputdatafinal.json" or blob.name == 'customer_data.json':
        continue
    
    pdf_content = getdatafromblob(blob.name,os.getenv("CONTAINER_NAME"))
    poller = document_client.begin_analyze_document("prebuilt-document",pdf_content)
    result = poller.result()
    full_text = ""
    for page in result.pages:
        for line in page.lines:
            full_text += line.content + "\n"
    document_list.append(full_text)


#getting the structured data from the blob
data = getdatafromblob('customer_data.json',os.getenv("CONTAINER_NAME"))
data = json.loads(data)



#mapped structured and unstructured data
for i in range(1,11,1):
    for document in document_list:
        if f"CustomerID: {str(i)}" in document:
            for customer in data:
                customer['document_text'] = "a"
                if(customer['CustomerID'] == str(i)):
                   
                    customer['document_text'] += document
                    

                    

dataitem = data[0]
data_dict = dict(dataitem)
key_list = list(data_dict.keys())

#CREATING THE INDEX
index_name = os.getenv("index_name")
search_client = SearchIndexClient(os.getenv("service_endpoint"),AzureKeyCredential(os.getenv("admin_key")))
key_list = list(data_dict.keys())




#FIELD SCHEMA FOR THE DATA STORED IN THE INDEX
fields = [
    SearchableField(name=f"{str(key_list[0])}", type=SearchFieldDataType.String, key=True, sortable=True, filterable=True, facetable=True),
]
for i in range(1,len(key_list),1):
    element = SearchableField(name = f"{str(key_list[i])}", type=SearchFieldDataType.String, sortable=True, filterable=True, facetable=True)
    fields.append(element)
for i in range(len(key_list)):

    element = SearchField(name=f"{str(key_list[i])}_Vector", type=SearchFieldDataType.Collection(SearchFieldDataType.Single), searchable=True, vector_search_dimensions=1536, vector_search_profile_name="myHnswProfile")
    fields.append(element)
   


names_lists = []
response_lists = []
embedding_lists = []
for i in range(len(key_list)):
    names_list = [str(dataitem[f'{key_list[i]}']) for dataitem in data]
    response_list= client.embeddings.create(input=names_list, model=os.getenv("azure_openai_em_name"))
    embedding_list= [item.embedding for item in response_list.data]
    names_lists.append(names_list)
    response_lists.append(response_list)
    embedding_lists.append(embedding_list)
    

for i,dataitem in enumerate(data):
    for j,key in enumerate(key_list):
        dataitem[f'{key}_Vector'] =  embedding_lists[j][i]


    
#upload the final embedded and mapped data into the blob
data_string = json.dumps(data)
uploaddata('llminputdatafinal.json',os.getenv("CONTAINER_NAME"),data_string)


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
# semantic_config = SemanticConfiguration(
#     name="customer-semantic-config",
#     prioritized_fields=SemanticPrioritizedFields(
#         title_field=SemanticField(field_name="CustomerID"),
#         keywords_fields=[
#             SemanticField(field_name="SeriousDlqin2yrs"),
#             SemanticField(field_name="RevolvingUtilizationOfUnsecuredLines"),
#             SemanticField(field_name="age"),
#             SemanticField(field_name="NumberOfTime30_59DaysPastDueNotWorse"),
#             SemanticField(field_name="DebtRatio"),
#             SemanticField(field_name="MonthlyIncome"),
#             SemanticField(field_name="NumberOfOpenCreditLinesAndLoans"),
#             SemanticField(field_name="NumberOfTimes90DaysLate"),
#             SemanticField(field_name="NumberRealEstateLoansOrLines"),
#             SemanticField(field_name="NumberOfTime60_89DaysPastDueNotWorse"),
#             SemanticField(field_name="NumberOfDependents"),
#             SemanticField(field_name="CreditScore"),
#             SemanticField(field_name="CreditHistoryLength"),
#             SemanticField(field_name="PaymentHistoryScore"),
#             SemanticField(field_name="LTV"),
#             SemanticField(field_name="TotalAssets"),
#             SemanticField(field_name="TotalLiabilities"),
#             SemanticField(field_name="EmploymentStatus_Retired"),
#             SemanticField(field_name="EmploymentStatus_Student"),
#             SemanticField(field_name="EmploymentStatus_Unemployed"),
#             SemanticField(field_name="EducationLevel_Bachelor_Degree"),
#             SemanticField(field_name="EducationLevel_High_School"),
#             SemanticField(field_name="EducationLevel_Master_Degree"),
#             SemanticField(field_name="EducationLevel_PhD")
#         ],
#         content_fields=[
#             SemanticField(field_name="CustomerFeedback"),
#             SemanticField(field_name="CustomerServiceLog"),
#             SemanticField(field_name="FeedbackSentimentScore"),
#             SemanticField(field_name="ServiceLogSentimentScore"),
#             SemanticField(field_name="document_text")
#         ]
#     )
# )



# semantic_search = SemanticSearch(configurations=[semantic_config])
# Create the search index and defining the algorithm we previously created
index = SearchIndex(name=index_name, fields=fields,
                    vector_search=vector_search)
result = search_client.create_or_update_index(index)
print(f' {result.name} created')


#getting the mapped and embedded data from the blob
prefinal_data = getdatafromblob('llminputdatafinal.json',os.getenv("CONTAINER_NAME"))
json_data = json.loads(prefinal_data)

#uploaded the documents to the vector store
search_client = SearchClient(endpoint=os.getenv("service_endpoint"), index_name=index_name, credential=AzureKeyCredential(os.getenv("admin_key")))
result = search_client.upload_documents(json_data)
print(f"Uploaded {len(json_data)} documents") 




