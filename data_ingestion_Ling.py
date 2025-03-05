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
from azure.storage.blob import BlobServiceClient, ResourceTypes, AccountSasPermissions, generate_account_sas, BlobClient
from azure.search.documents import SearchClient  
import requests
import json
import re
 
# Azure Blob Storage Configuration  
BLOB_CONNECTION_STRING = os.getenv("azure_storage_connection_string")
BLOB_ACCOUNT_NAME = os.getenv("storage_account")
BLOB_SAS_TOKEN = os.getenv("sas")

BLOB_CONTAINER_NAME = os.getenv("container_name")  
FOLDER_NAME = os.getenv("subfolder_name")
  
# Azure Document Intelligence Configuration  
DOCUMENT_INTELLIGENCE_ENDPOINT = os.getenv("document_intelligence_endpoint")  
DOCUMENT_INTELLIGENCE_KEY = os.getenv("document_intelligence_key")  
  
# Azure Cognitive Search Configuration  
SEARCH_SERVICE_ENDPOINT = os.getenv("azure_search_service_endpoint")  
SEARCH_API_KEY = os.getenv("azure_search_api_key")  
SEARCH_INDEX_NAME = os.getenv("search_index_name")

 
# OpenAI Configuration (If using Option 1 or 3)  
OPENAI_API_KEY = os.getenv("open_ai_credential")  
  
# Chunking Configuration  
MAX_CHUNK_SIZE = 1000  # Maximum number of characters per chunk  


client = AzureOpenAI(
    api_key = OPENAI_API_KEY,  
    api_version = "2024-06-01",
    azure_endpoint =os.getenv("open_ai_endpoint") 
)


# ----------------------- Step 0: listing files that are located in a data lake storage account -----------------------
def get_list_of_blob_names():
    blob_service_client = BlobServiceClient.from_connection_string(BLOB_CONNECTION_STRING)
    container_client = blob_service_client.get_container_client(BLOB_CONTAINER_NAME)
    blob_list = container_client.list_blobs(name_starts_with=FOLDER_NAME+"/1")
    return [blob.name for blob in blob_list if blob.name.endswith(".pdf")]

# ----------------------- Step 1: Access PDF from Blob Storage -----------------------  
def get_blob_sas_url(c_file):
    blob_url_path = f"https://{BLOB_ACCOUNT_NAME}.blob.core.windows.net/{BLOB_CONTAINER_NAME}/{c_file}"
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
    try:   
        client = DocumentIntelligenceClient(  
            endpoint=DOCUMENT_INTELLIGENCE_ENDPOINT,  
            credential=AzureKeyCredential(DOCUMENT_INTELLIGENCE_KEY)  
        )  
        poller = client.begin_analyze_document("prebuilt-layout", AnalyzeDocumentRequest(url_source=blob_path))  
        result = poller.result()  
        return result
    except Exception as e:
        print(f"Error in analyzing PDF: {e}")
        return None

# ----------------------- Step 3: Extract Text and Images -----------------------  

def extract_images_from_pdf(pdf_bytes):  
    images = []  
    response = requests.get(pdf_bytes)
    if response.status_code != 200:
        raise ValueError(f"Failed to download PDF. HTTP Status: {response.status_code}")
    
    pdf_bytes = response.content  # Get the PDF content as bytes

    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            image_list = page.get_images(full=True)

            if not image_list:
                continue

            for img in image_list:
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                images.append((page_num, image_bytes))  # Store page number for linking

    return images

def extract_content(document_analysis_result, pdf_bytes):
    text_content = {}
    images = extract_images_from_pdf(pdf_bytes)

    # Extract text
    for page_idx, page in enumerate(document_analysis_result.pages):
        page_text = "\n".join([line.content for line in page.lines])
        text_content[page_idx] = page_text  # Store page number for linking

    return text_content, images
  
# ----------------------- Step 4: Generate Image Summaries -----------------------  
  
# Option 1: Using LLM to Generate Image Summaries (OpenAI)
def generate_image_summary_openai(image_bytes):  
    # Convert image bytes to base64 string  
    image_base64 = base64.b64encode(image_bytes).decode('utf-8')
    image_data_uri = f"data:image/png;base64,{image_base64}"

    user_prompt = f"Provide a detailed summary for the following image:"  ## Modify prompt as needed
    system_prompt = f"You are an AI assistant that summarizes images based on provided data." ## Modify prompt as needed

    response = client.chat.completions.create(
        model=os.getenv("open_ai_model_gpt"),
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": image_data_uri},},
                ],
            },
        ],
        max_tokens=1000,
        temperature=0.1,
    )

    return response.choices[0].message.content.strip()  

# Function to choose summarization method  
def generate_image_summaries(images, option="openai"):  
    if option == "openai":  
        #Generate summaries for all images on a page and return a list per page.
        image_summary_dict = {}

        for page_num, img_bytes in images:
            img_summary = generate_image_summary_openai(img_bytes)
            image_summary_dict.setdefault(page_num, []).append(img_summary)

        return image_summary_dict  # Each page now has a list of image summaries

    else:  
        raise ValueError("Invalid option selected for image summarization.")  

# ----------------------- Step 5: Chunking/Embedding Strategy -----------------------  
def get_embedding(text_to_embed):
    """Calls Azure OpenAI API to get embeddings for the input text."""
    try:

        response = client.embeddings.create(
            input = text_to_embed,
            model= os.getenv("open_ai_model_name")
        )

        return response.data[0].embedding
    except Exception as e:
        print(f"Error fetching embedding: {e}")
        return None

def chunk_text_with_image(text_content, image_summaries, max_chunk_size=MAX_CHUNK_SIZE):  
    chunks = []
    
    for page_idx, page_text in text_content.items():  # Ensure correct iteration
        words = page_text.split()
        current_chunk = ""
        chunk_list = []

        for word in words:
            if len(current_chunk) + len(word) + 1 > max_chunk_size:
                chunk_list.append(current_chunk)
                current_chunk = word
            else:
                current_chunk += " " + word if current_chunk else word

        if current_chunk:
            chunk_list.append(current_chunk)

        # Get all image summaries for the page and Get embedding for the combined image summaries (only if summary exists)
        image_summary = " ".join(image_summaries.get(page_idx, []))
        image_summary_embedding = get_embedding(image_summary) if image_summary else []

        for chunk in chunk_list:
            chunks.append({
                "chunk": chunk,
                "chunk_embedding": get_embedding(chunk),
                "combine_content": f"{chunk} {image_summary}".strip(),
                "image_summary": image_summary,
                "image_summary_embedding": image_summary_embedding
            })

    return chunks

# ----------------------- Step 6: Return metadata ----------------------- 
def extract_metadata(blob_url):

    # Create a BlobClient using the SAS URL
    blob_client = BlobClient.from_blob_url(blob_url)

    # Get metadata from the blob
    return blob_client.get_blob_properties().metadata


# ----------------------- Step 7: Upload Documents to Search Index -----------------------  
def valid_document_key(document_key):
    return re.sub(r'[^a-zA-Z0-9_-]', '', document_key)

def upload_documents_to_search(chunks, metadata):  
    search_client = SearchClient(  
        endpoint=os.getenv('azure_search_service_endpoint'),  
        index_name=os.getenv('search_index_name'),  
        credential=AzureKeyCredential(os.getenv('azure_search_api_key'))  
    )  


    documents = []

    for chunk_id, chunk in enumerate(chunks):
        doc = {
            # "parent_id": metadata.get("metadata_file_name", "unknown"),
            # "title": metadata.get("metadata_file_name", "unknown"),
            # # Clean the document key to ensure it's valid
            # "chunk_id": valid_document_key(f"page-{metadata['metadata_file_name']}-{chunk_id}"),
            # "chunk": chunk["chunk"],
            # "combined_content": chunk["combine_content"],
            # "text_vector": chunk["chunk_embedding"],
            # "file_path": metadata.get("metadata_pdf_path", ""),
            # "video_path": metadata.get("metadata_video_path", ""),
            # "image_summaries": chunk["image_summary"],
            # "image_summary_embedding": chunk["image_summary_embedding"],
            # "has_image": bool(chunk["image_summary"])

            "chunk_id": valid_document_key(f"page-{metadata['metadata_file_name']}-{chunk_id}"),
            "parent_id": metadata.get("metadata_file_name", "unknown"),
            "title": metadata.get("metadata_file_name", "unknown"),

            "chunk": chunk["chunk"],
            "text_vector": chunk["chunk_embedding"],

            "image_summaries": chunk["image_summary"],
            "image_summary_embedding": chunk["image_summary_embedding"],

            "combined_content": chunk["combine_content"],

            "file_path": metadata.get("metadata_pdf_path", ""),
            "video_path": metadata.get("metadata_video_path", "")
        }
        documents.append(doc)

    search_client.upload_documents(documents)
    print(f"Uploaded {len(documents)} documents.")

# ----------------------- Main Execution -----------------------  


def main(blob_url):  

    # Step 2: Analyze PDF with Document Intelligence  
    analysis_result = analyze_pdf(blob_url)
    #print("-----------------------------------------------2")
    #print(analysis_result)  
    # Step 3: Extract Text and Images  
    extracted_text, extracted_images = extract_content(analysis_result,blob_url)  
    #print(extracted_text, extracted_images)
    #print("-----------------------------------------------3")
    #print(extracted_text)
    # Step 4: Generate Image Summaries  
    image_summaries = generate_image_summaries(extracted_images)
    #print("-----------------------------------------------4")
    #print(image_summaries)
    # Step 5: Chunk Text  
    chunk_info = chunk_text_with_image(extracted_text, image_summaries)  
    #print("-----------------------------------------------5")
    #print(chunk_info)
    # Step 6: get metadata
    metadata = extract_metadata(blob_url)
    #print("-----------------------------------------------6")
    #print(metadata)
    # Step 7: Upload Documents to Search Index  
    #print("-----------------------------------------------7")
    upload_documents_to_search(chunk_info, metadata)



if __name__ == "__main__":  
    file_list = get_list_of_blob_names()
    for file in file_list:
        print(f"Processing file: {file}")
        #Step 1: Download PDF from Blob Storage  
        blob_url = get_blob_sas_url(file)
        main(blob_url)