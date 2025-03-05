import os
from azure.core.credentials import AzureKeyCredential
from azure.identity import get_bearer_token_provider
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    AzureOpenAIVectorizer,
    AzureOpenAIVectorizerParameters,
    HnswAlgorithmConfiguration,
    HnswParameters,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SemanticConfiguration,
    SemanticField,
    SemanticPrioritizedFields,
    SemanticSearch,
    SimpleField,
    VectorSearch,
    VectorSearchProfile,
    VectorSearchVectorizer,
)


from dotenv import load_dotenv
load_dotenv()


search_endpoint=os.getenv("azure_search_service_endpoint")
search_credential=AzureKeyCredential(os.getenv("azure_search_api_key"))
index_name=os.getenv("search_index_name")


open_ai_endpoint = os.getenv("open_ai_endpoint")
open_ai_model_name=os.getenv("open_ai_model_name")
open_ai_deployment=os.getenv("open_ai_deployment")
open_ai_apiKey = os.getenv("open_ai_credential")
open_ai_dimensions=1536

# Create a search index  
index_client = SearchIndexClient(endpoint=search_endpoint, credential=search_credential)  

fields = [
    SimpleField(name="chunk_id", type="Edm.String", key=True, sortable=True),
    SimpleField(name="parent_id", type="Edm.String", filterable=True),

    SearchableField(name="title", type="Edm.String",sortable=True,filterable=True, analyzer_name="en.microsoft"),
    
    SearchableField(name="chunk", type="Edm.String", analyzer_name="en.microsoft"),
    SearchField(name="text_vector", type=SearchFieldDataType.Collection(SearchFieldDataType.Single), vector_search_dimensions=open_ai_dimensions, vector_search_profile_name="textvector-azureOpenAi-text-profile"),

    SearchableField(name="image_summaries", type="Edm.String", analyzer_name="en.microsoft"),
    SearchField(name="image_summary_embedding", type=SearchFieldDataType.Collection(SearchFieldDataType.Single), vector_search_dimensions=open_ai_dimensions, vector_search_profile_name="textvector-azureOpenAi-text-profile"),
    
    # Ensure combined content is properly searchable
    SearchableField(name="combined_content", type="Edm.String", analyzer_name="en.microsoft"),

    SimpleField(name="file_path", type="Edm.String"),
    SimpleField(name="video_path", type="Edm.String")
]


# Semantic Search Configuration
semantic_search = {
    "configurations": [
        {
            "name": "textvector-semantic-configuration",
            "prioritizedFields": {
            "titleField": {
                "fieldName": "title"
            },
            "prioritizedContentFields": [
                {
                "fieldName": "chunk"
                }
            ],
            "prioritizedKeywordsFields": []
            }
        }
    ]
}

# Configure the vector search configuration  
vector_search = VectorSearch(  
    algorithms=[  
        {
            "name": "textvector-algorithm",  # Define the name for the vector algorithm
            "kind": "hnsw",  # Hierarchical Navigable Small World (HNSW) is the algorithm used for vector search
            "hnswParameters": {
                "metric": "cosine",  # Using cosine similarity for vector distance metric
                "m": 4,  # Controls the number of connections in the graph
                "efConstruction": 400,  # Controls the construction time and accuracy of the index
                "efSearch": 500  # Controls the search accuracy (higher = more accurate but slower)
            }
        }
    ],  
    profiles=[  
        {
            "name": "textvector-azureOpenAi-text-profile",  # Profile name to use in the index
            "algorithm": "textvector-algorithm",  # Associate with the above-defined algorithm
            "vectorizer": "textvector-azureOpenAi-text-vectorizer"  # Specify the vectorizer to use
        }
    ],  
    vectorizers=[  
        {
            "name": "textvector-azureOpenAi-text-vectorizer",  # Define vectorizer name
            "kind": "azureOpenAI",  # Using Azure OpenAI as the vectorizer
            "azureOpenAIParameters": {
                "resourceUri": open_ai_endpoint,  # Endpoint for Azure OpenAI
                "deploymentId": open_ai_deployment,  # Deployment ID of the Azure OpenAI model
                "apiKey": AzureKeyCredential(open_ai_apiKey),  # API Key to authenticate requests
                "modelName": open_ai_model_name  # Model name (e.g., "text-embedding-ada-002")
            }
        }
    ],
)


print("Creating the index...")
# Create the search index
index = SearchIndex(
    name=index_name,
    fields=fields,
    semantic_search=semantic_search,  
    vector_search=vector_search      
)  

index_client.create_or_update_index(index)  
print(f"Index '{index_name}' created successfully!")  
