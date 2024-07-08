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
                    

                    






#CREATING THE INDEX
index_name = os.getenv("index_name")
search_client = SearchIndexClient(os.getenv("service_endpoint"),AzureKeyCredential(os.getenv("admin_key")))


#FIELD SCHEMA FOR THE DATA STORED IN THE INDEX
fields = [
    SearchableField(name="CustomerID", type=SearchFieldDataType.String, key=True, sortable=True, filterable=True, facetable=True),
    SearchableField(name="SeriousDlqin2yrs", type=SearchFieldDataType.String, sortable=True, filterable=True, facetable=True),
    SearchableField(name="RevolvingUtilizationOfUnsecuredLines", type=SearchFieldDataType.String, sortable=True, filterable=True, facetable=True),
    SearchableField(name="Age", type=SearchFieldDataType.String, sortable=True, filterable=True, facetable=True),
    SearchableField(name="NumberOfTime30_59DaysPastDueNotWorse", type=SearchFieldDataType.String, sortable=True, filterable=True, facetable=True),
    SearchableField(name="DebtRatio", type=SearchFieldDataType.String, sortable=True, filterable=True, facetable=True),
    SearchableField(name="MonthlyIncome", type=SearchFieldDataType.String, sortable=True, filterable=True, facetable=True),
    SearchableField(name="NumberOfOpenCreditLinesAndLoans", type=SearchFieldDataType.String, sortable=True, filterable=True, facetable=True),
    SearchableField(name="NumberOfTimes90DaysLate", type=SearchFieldDataType.String, sortable=True, filterable=True, facetable=True),
    SearchableField(name="NumberRealEstateLoansOrLines", type=SearchFieldDataType.String, sortable=True, filterable=True, facetable=True),
    SearchableField(name="NumberOfTime60_89DaysPastDueNotWorse", type=SearchFieldDataType.String, sortable=True, filterable=True, facetable=True),
    SearchableField(name="NumberOfDependents", type=SearchFieldDataType.String, sortable=True, filterable=True, facetable=True),
    SearchableField(name="CreditScore", type=SearchFieldDataType.String, sortable=True, filterable=True, facetable=True),
    SearchableField(name="CreditHistoryLength", type=SearchFieldDataType.String, sortable=True, filterable=True, facetable=True),
    SearchableField(name="PaymentHistoryScore", type=SearchFieldDataType.String, sortable=True, filterable=True, facetable=True),
    SearchableField(name="LTV", type=SearchFieldDataType.String, sortable=True, filterable=True, facetable=True),
    SearchableField(name="TotalAssets", type=SearchFieldDataType.String, sortable=True, filterable=True, facetable=True),
    SearchableField(name="TotalLiabilities", type=SearchFieldDataType.String, sortable=True, filterable=True, facetable=True),
    SearchableField(name="EmploymentStatus_Student", type=SearchFieldDataType.String, filterable=True),
    SearchableField(name="EmploymentStatus_Unemployed", type=SearchFieldDataType.String, filterable=True),
    SearchableField(name="EmploymentStatus_Retired", type=SearchFieldDataType.String, filterable=True),
    SearchableField(name="EducationLevel_Bachelor_Degree", type=SearchFieldDataType.String, filterable=True),
    SearchableField(name="EducationLevel_High_School", type=SearchFieldDataType.String, filterable=True),
    SearchableField(name="EducationLevel_Master_Degree", type=SearchFieldDataType.String, filterable=True),
    SearchableField(name="EducationLevel_PhD", type=SearchFieldDataType.String, filterable=True),
    SearchableField(name="CustomerFeedback", type=SearchFieldDataType.String),
    SearchableField(name="CustomerServiceLog", type=SearchFieldDataType.String),
    SearchableField(name="FeedbackSentimentScore", type=SearchFieldDataType.String, sortable=True, filterable=True, facetable=True),
    SearchableField(name="ServiceLogSentimentScore", type=SearchFieldDataType.String, sortable=True, filterable=True, facetable=True),
    SearchableField(name="document_text", type=SearchFieldDataType.String),
    SearchField(name="cust_id_Vector", type=SearchFieldDataType.Collection(SearchFieldDataType.Single), searchable=True, vector_search_dimensions=1536, vector_search_profile_name="myHnswProfile"),
    SearchField(name="serious_dlqin2yrs_Vector", type=SearchFieldDataType.Collection(SearchFieldDataType.Single), searchable=True, vector_search_dimensions=1536, vector_search_profile_name="myHnswProfile"),
    SearchField(name="revolving_utilization_of_unsecured_lines_Vector", type=SearchFieldDataType.Collection(SearchFieldDataType.Single), searchable=True, vector_search_dimensions=1536, vector_search_profile_name="myHnswProfile"),
    SearchField(name="age_Vector", type=SearchFieldDataType.Collection(SearchFieldDataType.Single), searchable=True, vector_search_dimensions=1536, vector_search_profile_name="myHnswProfile"),
    SearchField(name="num_time_30_59_days_past_due_not_worse_Vector", type=SearchFieldDataType.Collection(SearchFieldDataType.Single), searchable=True, vector_search_dimensions=1536, vector_search_profile_name="myHnswProfile"),
    SearchField(name="debt_ratio_Vector", type=SearchFieldDataType.Collection(SearchFieldDataType.Single), searchable=True, vector_search_dimensions=1536, vector_search_profile_name="myHnswProfile"),
    SearchField(name="monthly_income_Vector", type=SearchFieldDataType.Collection(SearchFieldDataType.Single), searchable=True, vector_search_dimensions=1536, vector_search_profile_name="myHnswProfile"),
    SearchField(name="num_open_credit_lines_and_loans_Vector", type=SearchFieldDataType.Collection(SearchFieldDataType.Single), searchable=True, vector_search_dimensions=1536, vector_search_profile_name="myHnswProfile"),
    SearchField(name="num_times_90_days_late_Vector", type=SearchFieldDataType.Collection(SearchFieldDataType.Single), searchable=True, vector_search_dimensions=1536, vector_search_profile_name="myHnswProfile"),
    SearchField(name="num_real_estate_loans_or_lines_Vector", type=SearchFieldDataType.Collection(SearchFieldDataType.Single), searchable=True, vector_search_dimensions=1536, vector_search_profile_name="myHnswProfile"),
    SearchField(name="num_time_60_89_days_past_due_not_worse_Vector", type=SearchFieldDataType.Collection(SearchFieldDataType.Single), searchable=True, vector_search_dimensions=1536, vector_search_profile_name="myHnswProfile"),
    SearchField(name="num_dependents_Vector", type=SearchFieldDataType.Collection(SearchFieldDataType.Single), searchable=True, vector_search_dimensions=1536, vector_search_profile_name="myHnswProfile"),
    SearchField(name="credit_score_Vector", type=SearchFieldDataType.Collection(SearchFieldDataType.Single), searchable=True, vector_search_dimensions=1536, vector_search_profile_name="myHnswProfile"),
    SearchField(name="credit_history_length_Vector", type=SearchFieldDataType.Collection(SearchFieldDataType.Single), searchable=True, vector_search_dimensions=1536, vector_search_profile_name="myHnswProfile"),
    SearchField(name="payment_history_score_Vector", type=SearchFieldDataType.Collection(SearchFieldDataType.Single), searchable=True, vector_search_dimensions=1536, vector_search_profile_name="myHnswProfile"),
    SearchField(name="ltv_Vector", type=SearchFieldDataType.Collection(SearchFieldDataType.Single), searchable=True, vector_search_dimensions=1536, vector_search_profile_name="myHnswProfile"),
    SearchField(name="total_assets_Vector", type=SearchFieldDataType.Collection(SearchFieldDataType.Single), searchable=True, vector_search_dimensions=1536, vector_search_profile_name="myHnswProfile"),
    SearchField(name="total_liabilities_Vector", type=SearchFieldDataType.Collection(SearchFieldDataType.Single), searchable=True, vector_search_dimensions=1536, vector_search_profile_name="myHnswProfile"),
    SearchField(name="employment_status_retired_Vector", type=SearchFieldDataType.Collection(SearchFieldDataType.Single), searchable=True, vector_search_dimensions=1536, vector_search_profile_name="myHnswProfile"),
    SearchField(name="employment_status_student_Vector", type=SearchFieldDataType.Collection(SearchFieldDataType.Single), searchable=True, vector_search_dimensions=1536, vector_search_profile_name="myHnswProfile"),
    SearchField(name="employment_status_unemployed_Vector", type=SearchFieldDataType.Collection(SearchFieldDataType.Single), searchable=True, vector_search_dimensions=1536, vector_search_profile_name="myHnswProfile"),
    SearchField(name="education_level_bachelor_Vector", type=SearchFieldDataType.Collection(SearchFieldDataType.Single), searchable=True, vector_search_dimensions=1536, vector_search_profile_name="myHnswProfile"),
    SearchField(name="education_level_high_school_Vector", type=SearchFieldDataType.Collection(SearchFieldDataType.Single), searchable=True, vector_search_dimensions=1536, vector_search_profile_name="myHnswProfile"),
    SearchField(name="education_level_master_Vector", type=SearchFieldDataType.Collection(SearchFieldDataType.Single), searchable=True, vector_search_dimensions=1536, vector_search_profile_name="myHnswProfile"),
    SearchField(name="education_level_phd_Vector", type=SearchFieldDataType.Collection(SearchFieldDataType.Single), searchable=True, vector_search_dimensions=1536, vector_search_profile_name="myHnswProfile"),
    SearchField(name="customer_feedback_Vector", type=SearchFieldDataType.Collection(SearchFieldDataType.Single), searchable=True, vector_search_dimensions=1536, vector_search_profile_name="myHnswProfile"),
    SearchField(name="customer_service_log_Vector", type=SearchFieldDataType.Collection(SearchFieldDataType.Single), searchable=True, vector_search_dimensions=1536, vector_search_profile_name="myHnswProfile"),
    SearchField(name="feedback_sentiment_score_Vector", type=SearchFieldDataType.Collection(SearchFieldDataType.Single), searchable=True, vector_search_dimensions=1536, vector_search_profile_name="myHnswProfile"),
    SearchField(name="service_log_sentiment_score_Vector", type=SearchFieldDataType.Collection(SearchFieldDataType.Single), searchable=True, vector_search_dimensions=1536, vector_search_profile_name="myHnswProfile"),
    SearchField(name="document_text_Vector", type=SearchFieldDataType.Collection(SearchFieldDataType.Single), searchable=True, vector_search_dimensions=1536, vector_search_profile_name="myHnswProfile"),
]


#storing all the keys in their respective newly created arrays
cust_ids = [str(dataitem['CustomerID']) for dataitem in data]
serious_dlqin2yrs = [str(dataitem['SeriousDlqin2yrs']) for dataitem in data]
revolving_utilization_of_unsecured_lines = [str(dataitem['RevolvingUtilizationOfUnsecuredLines']) for dataitem in data]
ages = [str(dataitem['age']) for dataitem in data]
num_time_30_59_days_past_due_not_worse = [str(dataitem['NumberOfTime30_59DaysPastDueNotWorse']) for dataitem in data]
debt_ratios = [str(dataitem['DebtRatio']) for dataitem in data]
monthly_incomes = [str(dataitem['MonthlyIncome']) for dataitem in data]
num_open_credit_lines_and_loans = [str(dataitem['NumberOfOpenCreditLinesAndLoans']) for dataitem in data]
num_times_90_days_late = [str(dataitem['NumberOfTimes90DaysLate']) for dataitem in data]
num_real_estate_loans_or_lines = [str(dataitem['NumberRealEstateLoansOrLines']) for dataitem in data]
num_time_60_89_days_past_due_not_worse = [str(dataitem['NumberOfTime60_89DaysPastDueNotWorse']) for dataitem in data]
num_dependents = [str(dataitem['NumberOfDependents']) for dataitem in data]
credit_scores = [str(dataitem['CreditScore']) for dataitem in data]
credit_history_lengths = [str(dataitem['CreditHistoryLength']) for dataitem in data]
payment_history_scores = [str(dataitem['PaymentHistoryScore']) for dataitem in data]
ltvs = [str(dataitem['LTV']) for dataitem in data]
total_assets = [str(dataitem['TotalAssets']) for dataitem in data]
total_liabilities = [str(dataitem['TotalLiabilities']) for dataitem in data]
employment_status_retired = [str(dataitem['EmploymentStatus_Retired']) for dataitem in data]
employment_status_student = [str(dataitem['EmploymentStatus_Student']) for dataitem in data]
employment_status_unemployed = [str(dataitem['EmploymentStatus_Unemployed']) for dataitem in data]
education_level_bachelor = [str(dataitem['EducationLevel_Bachelor_Degree']) for dataitem in data]
education_level_high_school = [str(dataitem['EducationLevel_High_School']) for dataitem in data]
education_level_master = [str(dataitem['EducationLevel_Master_Degree']) for dataitem in data]
education_level_phd = [str(dataitem['EducationLevel_PhD']) for dataitem in data]
customer_feedback = [dataitem['CustomerFeedback'] for dataitem in data] 
customer_service_log = [dataitem['CustomerServiceLog'] for dataitem in data]  
feedback_sentiment_score = [str(dataitem['FeedbackSentimentScore']) for dataitem in data]
service_log_sentiment_score = [str(dataitem['ServiceLogSentimentScore']) for dataitem in data]
document_texts = [str(dataitem['document_text']) for dataitem in data]

#creating the embeddings for the keys using their respective arrays
cust_ids_response = client.embeddings.create(input=cust_ids, model=os.getenv("azure_openai_em_name"))
cust_id_embeddings = [item.embedding for item in cust_ids_response.data]

serious_dlqin2yrs_response = client.embeddings.create(input=serious_dlqin2yrs, model=os.getenv("azure_openai_em_name"))
serious_dlqin2yrs_embeddings = [item.embedding for item in serious_dlqin2yrs_response.data]

revolving_utilization_of_unsecured_lines_response = client.embeddings.create(input=revolving_utilization_of_unsecured_lines, model=os.getenv("azure_openai_em_name"))
revolving_utilization_of_unsecured_lines_embeddings = [item.embedding for item in revolving_utilization_of_unsecured_lines_response.data]

ages_response = client.embeddings.create(input=ages, model=os.getenv("azure_openai_em_name"))
ages_embeddings = [item.embedding for item in ages_response.data]

num_time_30_59_days_past_due_not_worse_response = client.embeddings.create(input=num_time_30_59_days_past_due_not_worse, model=os.getenv("azure_openai_em_name"))
num_time_30_59_days_past_due_not_worse_embeddings = [item.embedding for item in num_time_30_59_days_past_due_not_worse_response.data]

debt_ratios_response = client.embeddings.create(input=debt_ratios, model=os.getenv("azure_openai_em_name"))
debt_ratios_embeddings = [item.embedding for item in debt_ratios_response.data]

monthly_incomes_response = client.embeddings.create(input=monthly_incomes, model=os.getenv("azure_openai_em_name"))
monthly_incomes_embeddings = [item.embedding for item in monthly_incomes_response.data]

num_open_credit_lines_and_loans_response = client.embeddings.create(input=num_open_credit_lines_and_loans, model=os.getenv("azure_openai_em_name"))
num_open_credit_lines_and_loans_embeddings = [item.embedding for item in num_open_credit_lines_and_loans_response.data]

num_times_90_days_late_response = client.embeddings.create(input=num_times_90_days_late, model=os.getenv("azure_openai_em_name"))
num_times_90_days_late_embeddings = [item.embedding for item in num_times_90_days_late_response.data]

num_real_estate_loans_or_lines_response = client.embeddings.create(input=num_real_estate_loans_or_lines, model=os.getenv("azure_openai_em_name"))
num_real_estate_loans_or_lines_embeddings = [item.embedding for item in num_real_estate_loans_or_lines_response.data]

num_time_60_89_days_past_due_not_worse_response = client.embeddings.create(input=num_time_60_89_days_past_due_not_worse, model=os.getenv("azure_openai_em_name"))
num_time_60_89_days_past_due_not_worse_embeddings = [item.embedding for item in num_time_60_89_days_past_due_not_worse_response.data]

num_dependents_response = client.embeddings.create(input=num_dependents, model=os.getenv("azure_openai_em_name"))
num_dependents_embeddings = [item.embedding for item in num_dependents_response.data]

credit_scores_response = client.embeddings.create(input=credit_scores, model=os.getenv("azure_openai_em_name"))
credit_scores_embeddings = [item.embedding for item in credit_scores_response.data]

credit_history_lengths_response = client.embeddings.create(input=credit_history_lengths, model=os.getenv("azure_openai_em_name"))
credit_history_lengths_embeddings = [item.embedding for item in credit_history_lengths_response.data]

payment_history_scores_response = client.embeddings.create(input=payment_history_scores, model=os.getenv("azure_openai_em_name"))
payment_history_scores_embeddings = [item.embedding for item in payment_history_scores_response.data]

ltvs_response = client.embeddings.create(input=ltvs, model=os.getenv("azure_openai_em_name"))
ltvs_embeddings = [item.embedding for item in ltvs_response.data]

total_assets_response = client.embeddings.create(input=total_assets, model=os.getenv("azure_openai_em_name"))
total_assets_embeddings = [item.embedding for item in total_assets_response.data]

total_liabilities_response = client.embeddings.create(input=total_liabilities, model=os.getenv("azure_openai_em_name"))
total_liabilities_embeddings = [item.embedding for item in total_liabilities_response.data]

employment_status_retired_response = client.embeddings.create(input=employment_status_retired, model=os.getenv("azure_openai_em_name"))
employment_status_retired_embeddings = [item.embedding for item in employment_status_retired_response.data]

employment_status_student_response = client.embeddings.create(input=employment_status_student, model=os.getenv("azure_openai_em_name"))
employment_status_student_embeddings = [item.embedding for item in employment_status_student_response.data]

employment_status_unemployed_response = client.embeddings.create(input=employment_status_unemployed, model=os.getenv("azure_openai_em_name"))
employment_status_unemployed_embeddings = [item.embedding for item in employment_status_unemployed_response.data]

education_level_bachelor_response = client.embeddings.create(input=education_level_bachelor, model=os.getenv("azure_openai_em_name"))
education_level_bachelor_embeddings = [item.embedding for item in education_level_bachelor_response.data]

education_level_high_school_response = client.embeddings.create(input=education_level_high_school, model=os.getenv("azure_openai_em_name"))
education_level_high_school_embeddings = [item.embedding for item in education_level_high_school_response.data]

education_level_master_response = client.embeddings.create(input=education_level_master, model=os.getenv("azure_openai_em_name"))
education_level_master_embeddings = [item.embedding for item in education_level_master_response.data]

education_level_phd_response = client.embeddings.create(input=education_level_phd, model=os.getenv("azure_openai_em_name"))
education_level_phd_embeddings = [item.embedding for item in education_level_phd_response.data]

customer_feedback_response = client.embeddings.create(input=customer_feedback, model=os.getenv("azure_openai_em_name"))
customer_feedback_embeddings = [item.embedding for item in customer_feedback_response.data]

customer_service_log_response = client.embeddings.create(input=customer_service_log, model=os.getenv("azure_openai_em_name"))
customer_service_log_embeddings = [item.embedding for item in customer_service_log_response.data]

feedback_sentiment_score_response = client.embeddings.create(input=feedback_sentiment_score, model=os.getenv("azure_openai_em_name"))
feedback_sentiment_score_embeddings = [item.embedding for item in feedback_sentiment_score_response.data]

service_log_sentiment_score_response = client.embeddings.create(input=service_log_sentiment_score, model=os.getenv("azure_openai_em_name"))
service_log_sentiment_score_embeddings = [item.embedding for item in service_log_sentiment_score_response.data]

document_texts_response = client.embeddings.create(input=document_texts, model=os.getenv("azure_openai_em_name"))
document_texts_embeddings = [item.embedding for item in document_texts_response.data]


#for each data items in the mapped data, adding the respective embedded fields alongside their old fields
for i, dataitem in enumerate(data):
    dataitem['cust_id_Vector'] = cust_id_embeddings[i]
    dataitem['serious_dlqin2yrs_Vector'] = serious_dlqin2yrs_embeddings[i]
    dataitem['revolving_utilization_of_unsecured_lines_Vector'] = revolving_utilization_of_unsecured_lines_embeddings[i]
    dataitem['age_Vector'] = ages_embeddings[i]
    dataitem['num_time_30_59_days_past_due_not_worse_Vector'] = num_time_30_59_days_past_due_not_worse_embeddings[i]
    dataitem['debt_ratio_Vector'] = debt_ratios_embeddings[i]
    dataitem['monthly_income_Vector'] = monthly_incomes_embeddings[i]
    dataitem['num_open_credit_lines_and_loans_Vector'] = num_open_credit_lines_and_loans_embeddings[i]
    dataitem['num_times_90_days_late_Vector'] = num_times_90_days_late_embeddings[i]
    dataitem['num_real_estate_loans_or_lines_Vector'] = num_real_estate_loans_or_lines_embeddings[i]
    dataitem['num_time_60_89_days_past_due_not_worse_Vector'] = num_time_60_89_days_past_due_not_worse_embeddings[i]
    dataitem['num_dependents_Vector'] = num_dependents_embeddings[i]
    dataitem['credit_score_Vector'] = credit_scores_embeddings[i]
    dataitem['credit_history_length_Vector'] = credit_history_lengths_embeddings[i]
    dataitem['payment_history_score_Vector'] = payment_history_scores_embeddings[i]
    dataitem['ltv_Vector'] = ltvs_embeddings[i]
    dataitem['total_assets_Vector'] = total_assets_embeddings[i]
    dataitem['total_liabilities_Vector'] = total_liabilities_embeddings[i]
    dataitem['employment_status_retired_Vector'] = employment_status_retired_embeddings[i]
    dataitem['employment_status_student_Vector'] = employment_status_student_embeddings[i]
    dataitem['employment_status_unemployed_Vector'] = employment_status_unemployed_embeddings[i]
    dataitem['education_level_bachelor_Vector'] = education_level_bachelor_embeddings[i]
    dataitem['education_level_high_school_Vector'] = education_level_high_school_embeddings[i]
    dataitem['education_level_master_Vector'] = education_level_master_embeddings[i]
    dataitem['education_level_phd_Vector'] = education_level_phd_embeddings[i]
    dataitem['customer_feedback_Vector'] = customer_feedback_embeddings[i]
    dataitem['customer_service_log_Vector'] = customer_service_log_embeddings[i]
    dataitem['feedback_sentiment_score_Vector'] = feedback_sentiment_score_embeddings[i]
    dataitem['service_log_sentiment_score_Vector'] = service_log_sentiment_score_embeddings[i]
    dataitem['document_text_Vector'] = document_texts_embeddings[i]






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




