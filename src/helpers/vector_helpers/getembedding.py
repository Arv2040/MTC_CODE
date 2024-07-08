import os 
from azure.search.documents.models import VectorizedQuery
from dotenv import load_dotenv
load_dotenv()



def get_embedding(query,fields_string,client):
        embedding = client.embeddings.create(input=query, model=os.getenv("azure_openai_em_name")).data[0].embedding
        vector_query = VectorizedQuery(vector=embedding, k_nearest_neighbors=3, fields=fields_string)
        return vector_query
