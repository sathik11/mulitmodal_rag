import os,io
from dotenv import load_dotenv
load_dotenv()
import base64  
from datetime import datetime, timedelta
import fitz
from io import BytesIO  
from PIL import Image 
from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeResult, AnalyzeDocumentRequest
from azure.storage.blob import BlobServiceClient, ResourceTypes, AccountSasPermissions, generate_account_sas
from azure.search.documents import SearchClient  
 
# storage_connection_string = os.getenv("azure_storage_connection_string")

# Azure Blob Storage Configuration  
# BLOB_CONNECTION_STRING = os.getenv("BLOB_CONNECTION_STRING") 
BLOB_ACCOUNT_NAME = os.getenv("BLOB_ACCT_NAME")
BLOB_SAS_TOKEN = os.getenv("BLOB_SAS_TOKEN")

BLOB_CONTAINER_NAME = "sats-wfs"  
FOLDER_NAME = "parent_folder"
PDF_BLOB_NAME = "WFS-SATS-PR-28-Sep-2022.pdf" 
  
# Azure Document Intelligence Configuration  
DOCUMENT_INTELLIGENCE_ENDPOINT = os.getenv("DOCUMENT_INTELLIGENCE_ENDPOINT")  
DOCUMENT_INTELLIGENCE_KEY = os.getenv("DOCUMENT_INTELLIGENCE_KEY")  
  
# Azure Cognitive Search Configuration  
SEARCH_SERVICE_ENDPOINT = os.getenv("SEARCH_SERVICE_ENDPOINT")  
SEARCH_API_KEY = os.getenv("SEARCH_API_KEY")  
SEARCH_INDEX_NAME = "cargospot-codebase-index"  
 
# OpenAI Configuration (If using Option 1 or 3)  
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  
  
# Chunking Configuration  
MAX_CHUNK_SIZE = 1000  # Maximum number of characters per chunk  

# ----------------------- Step 1: Access PDF from Blob Storage -----------------------  
def get_blob_sas_url():
    blob_url_path = f"https://{BLOB_ACCOUNT_NAME}.blob.core.windows.net/{BLOB_CONTAINER_NAME}/{FOLDER_NAME}/{PDF_BLOB_NAME}"
    blob_sas_url_path = f"{blob_url_path}?{BLOB_SAS_TOKEN}"
    return blob_sas_url_path
# def download_pdf_from_blob():  
#     blob_service_client = BlobServiceClient.from_connection_string(BLOB_CONNECTION_STRING)  
#     blob_client = blob_service_client.get_blob_client(container=BLOB_CONTAINER_NAME, blob=PDF_BLOB_NAME)  
#     download_stream = blob_client.download_blob()  
#     pdf_content = download_stream.readall()  
#     return pdf_content  

# ----------------------- Step 2: Analyze PDF with Document Intelligence ----------------------- 
def analyze_pdf(blob_path):  
    client = DocumentIntelligenceClient(  
        endpoint=DOCUMENT_INTELLIGENCE_ENDPOINT,  
        credential=AzureKeyCredential(DOCUMENT_INTELLIGENCE_KEY)  
    )  
    poller = client.begin_analyze_document("prebuilt-layout", AnalyzeDocumentRequest(url_source=blob_path))  
    result = poller.result()  
    return result 

# ----------------------- Step 3: Extract Text and Images -----------------------  

def extract_images_from_pdf(pdf_bytes):  
    images = []  
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:  
        for page_num in range(len(doc)):  
            page = doc.load_page(page_num)  
            image_list = page.get_images(full=True)  
            for img_index, img in enumerate(image_list):  
                xref = img[0]  
                base_image = doc.extract_image(xref)  
                image_bytes = base_image["image"]  
                images.append(image_bytes)  
    return images 


def extract_content(document_analysis_result, pdf_bytes) :  
    text_content = ""  
    images = []  
  
    # Extract text  
    for page in document_analysis_result.pages:  
        for line in page.lines:  
            text_content += line.content + "\n"  
    print(text_content)
    if document_analysis_result.figures:
        for page_idx, page in enumerate(document_analysis_result.pages):  
            for image_idx, img in enumerate(page.figures):  
                if hasattr(img, 'content'):  
                    # If the image content is directly available  
                    images.append(img.content)  
                else:  
                    print(f"Image data not directly accessible for page {page_idx + 1}, image {image_idx + 1}")  
            # If no images extracted via SDK, fallback to PDF parsing  
    # if not images:  
    #     print("No images extracted via SDK. Attempting to extract images using PyMuPDF...")  
    #     images = extract_images_from_pdf(pdf_bytes)  
    #     print(f"Extracted {len(images)} images using PyMuPDF.")    
    return text_content, images  
  
# ----------------------- Step 4: Generate Image Summaries -----------------------  
  
# Option 1: Using LLM to Generate Image Summaries (OpenAI)
def generate_image_summary_openai(image_bytes):  
    # Convert image bytes to base64 string  
    image_base64 = base64.b64encode(image_bytes).decode('utf-8')  
    user_prompt = f"Provide a detailed summary for the following image:"  ## Modify prompt as needed
    system_prompt = f"You are an AI assistant that summarizes images based on provided data." ## Modify prompt as needed
    client = AzureOpenAI(
                api_key = os.getenv("AZURE_OPENAI_API_KEY"),  
                api_version = "2024-06-01",
                azure_endpoint =os.getenv("AZURE_OPENAI_ENDPOINT") 
                )
    response = client.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {"type": "image_url", "image_url": {"url": image_base64},
                    },
                    ],
                },
            ],
            max_tokens=2000,
            temperature=0.7,
        )

    summary = response.choices[0].message['content'].strip()  
    return summary 

# Function to choose summarization method  
def generate_image_summary(image_bytes, option="openai"):  
    if option == "openai":  
        return generate_image_summary_openai(image_bytes)  
    else:  
        raise ValueError("Invalid option selected for image summarization.")  

# ----------------------- Step 5: Chunking/Embedding Strategy -----------------------  
def get_embedding(text_to_embed):
    """Calls Azure OpenAI API to get embeddings for the input text."""
    try:
        client = AzureOpenAI(
                api_key = os.getenv("AZURE_OPENAI_API_KEY"),  
                api_version = "2024-06-01",
                azure_endpoint =os.getenv("AZURE_OPENAI_ENDPOINT") 
                )

        response = client.embeddings.create(
            input = text_to_embed,
            model= os.getenv("AZURE_OPENAI_EMBEDDING")
        )

        return response.data[0].embedding
    except Exception as e:
        print(f"Error fetching embedding: {e}")

def chunk_text(text, max_chunk_size=MAX_CHUNK_SIZE):  
    words = text.split()  
    chunks = []
    embeddings = []  
    current_chunk = ""  
  
    for word in words:  
        if len(current_chunk) + len(word) + 1 > max_chunk_size:  
            chunks.append(current_chunk)  
            current_chunk = word  
        else:  
            if current_chunk:  
                current_chunk += " " + word  
            else:  
                current_chunk = word  
  
    if current_chunk:  
        chunks.append(current_chunk)
        embeddings.append(get_embedding(current_chunk))  
  
    return chunks, embeddings

# ----------------------- Step 6: Upload Documents to Search Index -----------------------  
  
def upload_documents_to_search(chunks, image_summaries, embeddings,other_metadata):  
    search_client = SearchClient(  
        endpoint=SEARCH_SERVICE_ENDPOINT,  
        index_name=SEARCH_INDEX_NAME,  
        credential=AzureKeyCredential(SEARCH_API_KEY)  
    )  
  
    documents = []  
    for idx, chunk in enumerate(chunks):  
        doc = {  
            "id": idx,  
            "content": chunk,  
            "content_vector": embeddings[idx] if idx < len(embeddings) else "",
            "image_summary": image_summaries[idx] if idx < len(image_summaries) else ""  
        }  
        documents.append(doc)  
  
    result = search_client.upload_documents(documents)  
    print(f"Uploaded {len(result)} documents to the search index.")  

# ----------------------- Main Execution -----------------------  
  
def main():  
    # Step 1: Download PDF from Blob Storage  
    print("Downloading PDF from Blob Storage...")  
    pdf_bytes = get_blob_sas_url()#download_pdf_from_blob()  
    print("PDF downloaded successfully.")  
  
    # Step 2: Analyze PDF with Document Intelligence  
    print("Analyzing PDF with Azure Document Intelligence...")  
    analysis_result = analyze_pdf(pdf_bytes)  
    print("PDF analysis completed.")  
  
    # Step 3: Extract Text and Images  
    print("Extracting text and images from analysis result...")  
    text, images = extract_content(analysis_result,pdf_bytes)  
    print(f"Extracted {len(images)} images from the document.")  
  
    # Step 4: Generate Image Summaries  
    print("Generating summaries for extracted images using Azure Computer Vision...")  
    image_summaries = []  
    for idx, img_bytes in enumerate(images):  
        if img_bytes:  
            print(f"Processing image {idx + 1}/{len(images)}...")  
            summary = generate_image_summary(img_bytes, option="azure_cv")  # Change option if needed  
            image_summaries.append(summary)  
        else:  
            print(f"No image data available for image {idx + 1}.")  
            image_summaries.append("")  
  
    print("Image summaries generated.") 
  
    # Step 5: Chunk Text  
    print("Chunking text for optimal indexing...")  
    chunks = chunk_text(text)  
    print(f"Text divided into {len(chunks)} chunks.")  
  
  
    # # Step 6: Upload Documents to Search Index  
    # print("Uploading documents to Azure Cognitive Search...")  
    # upload_documents_to_search(chunks, image_summaries)  
    # print("All documents uploaded successfully.")  
  
if __name__ == "__main__":  
    main()  