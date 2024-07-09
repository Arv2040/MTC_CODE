# MTC_CODE

## CrediShield

This is an application designed to cater to the needs of professionals working in the FSI Sector.
It automates the entire process of determining the Creditworthiness of a customer for Loan Approval as well as effective detection of Corporate Frauds.

It leverages cutting-edge technologies like Generative AI and Retrieval Augmented Generation (RAG) at its core to give seamless and detailed results.

**GPT-4o** with its multimodal capabilities plays a vital role in both our use cases to present a state-of-the-art solution which is able to determine creditworthiness and cases of corporate fraud at the click of a button with raw media files as inputs along with the regular structured data. 
(ex. FORM 16, images of pledged collaterals, call logs, emails)

<img width="861" alt="Screenshot 2024-07-09 at 6 36 21 PM" src="https://github.com/Arv2040/MTC_CODE/assets/34931141/05a795bf-7b38-4c08-853d-2031c2f37ef7">

The system also leverages the capabilities of LangChain to maintain a conversational interface in the form of a **Copilot** which keeps context of previous queries and can be used to repeatedly reason on top of the report which the system has originally generated.

Multilingual and Voice inputs are also key features of the system which distinguish it from other mainstream solutions.

## To run the code on your system

Clone this repository into your IDE.

Install the dependencies for the code to execute effectively.
```
pip install -r requirements.txt
```
To perform the indexing of vector data for Creditworthiness and execute the RAG on it, run the following command:
```
python cw_vector.py
```
To perform the indexing of vector data for Fraud Detection and execute the RAG on it, run the following command:
```
python cf_vector.py
```
To run the frontend module for Creditworthiness, run the following command:
```
streamlit run cw_run.py
```
To run the frontend module for Fraud Detection, run the following command:
```
streamlit run cf_run.py
```
