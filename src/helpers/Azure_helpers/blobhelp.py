import os
from dotenv import load_dotenv
load_dotenv()
from azure.storage.blob import BlobServiceClient


blob_service_client = BlobServiceClient(account_url=f'https://{os.getenv("AZURE_STORAGE_ACCOUNT_NAME")}.blob.core.windows.net', credential=os.getenv("AZURE_STORAGE_ACCOUNT_KEY"))



def getdatafromblob(blobname,containername):
    container_client = blob_service_client.get_container_client(container=containername)
    blob_client = container_client.get_blob_client(blobname)
    downloaded_blob = blob_client.download_blob()
    data= downloaded_blob.readall()
    return data

def getbloblist(containername):
    container_client = blob_service_client.get_container_client(container=containername)
    return container_client.list_blobs()

def uploaddata(newblobname,containername,data):
    new_blob_client = blob_service_client.get_blob_client(container=containername, blob=newblobname)
    new_blob_client.upload_blob(data, overwrite=True)
    
    print(f"File {newblobname} uploaded to Azure Blob Storage in container")

def createcontainer(containername):
    container_client = blob_service_client.create_container(container = containername)
    return container_client