# MTC_CODE

## CrediShield

This is an application designed to cater to the needs of professionals working in the FSI Sector.
It automates the entire process of determining the Creditworthiness of a customer for Loan Approval as well as effective detection of Corporate Frauds.

It leverages cutting-edge technologies like Generative AI and Retrieval Augmented Generation (RAG) at its core to give seamless and detailed results.

**GPT-4o** with its multimodal capabilities plays a vital role in both our use cases to present a state-of-the-art solution which is able to determine creditworthiness and cases of corporate fraud at the click of a button with raw media files as inputs along with the regular structured data. 
(ex. FORM 16, images of pledged collaterals, call logs, emails)

<img width="809" alt="Screenshot 2024-07-22 at 1 17 05 PM" src="https://github.com/user-attachments/assets/a8331bcf-4deb-49ba-9feb-9a4462986277">

The system also leverages the capabilities of LangChain to maintain a conversational interface in the form of a **Copilot** which keeps context of previous queries and can be used to repeatedly reason on top of the report which the system has originally generated.

Multilingual and Voice inputs are also key features of the system which distinguish it from other mainstream solutions.

## Standard Operating Procedure (SOP) for Fraud Detection and Categorization

     1. Identity Falsification:
       - Definition: Unauthorized use of personal or business information for financial gain.
       - Key Indicators:
         a) Discrepancies in personal or business identification details
         b) Multiple accounts with similar details but different names
         c) Sudden changes in account holder information
       

     2. Tax Compliance Violation:
       - Definition: Intentional evasion of tax obligations or false tax claims.
       - Key Indicators:
         a) Inconsistencies between reported income and observed financial activity
         b) Large, unexplained deductions or credits
         c) Discrepancies in tax filings across different periods
       

     3. Transaction Misappropriation:
       - Definition: Unauthorized or deceptive financial transactions.
       - Key Indicators:
         a) Unusual patterns in transaction timing, frequency, or amounts
         b) Transactions with high-risk or sanctioned entities
         c) Circular transactions or unexplained fund movements
       
     4. Business Process Manipulation:
       - Definition: Manipulation of business operations for fraudulent gains.
       - Key Indicators:
         a) Discrepancies between reported business activities and financial flows
         b) Unusual patterns in inventory, procurement, or sales data
         c) Inconsistencies in operational metrics and financial outcomes
       
     5. Financial Misrepresentation:
       - Definition: Obtaining credit through false pretenses or misuse of credit facilities.
       - Key Indicators:
         a) Rapid increase in credit utilization
         b) Discrepancies between reported income and credit behavior
         c) Unusual patterns in repayment behavior

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
To execute the frontend module for Creditworthiness, run the following command:
```
streamlit run cw_run.py
```

To execute the frontend module for Fraud Detection, run the following command:
```
streamlit run app.py
streamlit run temp.py
```
