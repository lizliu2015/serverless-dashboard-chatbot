############################################################################################################################################################################
# This Lambda function processes a PDF document to create and store a searchable index. When a new document is added to the queue:
#
#    It updates the document's status in DynamoDB to "PROCESSING."
#    Downloads the PDF file from the S3 bucket.
#    Utilizes PyPDFLoader to load the PDF file.
#    Creates an instance of BedrockEmbeddings for embedding generation, using Amazon's titan-embed-text-v1 model.
#    VectorstoreIndexCreator is used to create a searchable index of the document's content.
#    The generated index files (FAISS and pickle formats) are saved locally and then uploaded to S3 in the user's specific directory.
#    Finally, it updates the document's status to "READY" in DynamoDB.
#
# This script is key for converting PDF documents into a format that can be efficiently searched, thereby facilitating the document chat feature of the application.
############################################################################################################################################################################
    
import os, json
import boto3
from aws_lambda_powertools import Logger
from langchain.embeddings import BedrockEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.vectorstores import FAISS

# Environment Variables
DOCUMENT_TABLE = os.environ["DOCUMENT_TABLE"]
BUCKET = os.environ["BUCKET"]

s3 = boto3.client("s3")
ddb = boto3.resource("dynamodb")
document_table = ddb.Table(DOCUMENT_TABLE)
logger = Logger()


def set_doc_status(user_id, document_id, status):
    document_table.update_item(
        Key={"userid": user_id, "documentid": document_id},
        UpdateExpression="SET docstatus = :docstatus",
        ExpressionAttributeValues={":docstatus": status},
    )

@logger.inject_lambda_context(log_event=True)
#
#    It updates the document's status in DynamoDB to "PROCESSING."
#    Downloads the PDF file from the S3 bucket.
#    Utilizes PyPDFLoader to load the PDF file.
#    Creates an instance of BedrockEmbeddings for embedding generation, using Amazon's titan-embed-text-v1 model.
#    VectorstoreIndexCreator is used to create a searchable index of the document's content.
#    The generated index files (FAISS and pickle formats) are saved locally and then uploaded to S3 in the user's specific directory.
#    Finally, it updates the document's status to "READY" in DynamoDB.

def lambda_handler(event, context):
    # List all PDF files in the bucket
    response = s3.list_objects_v2(Bucket=BUCKET)
    if 'Contents' not in response:
        return "No files found in the bucket."

    event_body = json.loads(event["Records"][0]["body"])
    document_id = event_body["documentid"]
    user_id = event_body["user"]
    # key = event_body["key"]

    set_doc_status(user_id, document_id, "PROCESSING")

    loaders = []  # List to hold all PyPDFLoaders
    for file in response['Contents']:
        key = file['Key']
        if not key.lower().endswith('.pdf'):
            continue  # Process only PDF files
        file_name_full = key.split('/')[-1]
        
        # Download the PDF file from the S3 bucket
        s3.download_file(BUCKET, key, f"/tmp/{file_name_full}")

        # Utilizes PyPDFLoader to load the PDF file
        loader = PyPDFLoader(f"/tmp/{file_name_full}")
        loaders.append(loader)

    bedrock_runtime = boto3.client(
        service_name="bedrock-runtime",
        region_name="us-east-1",
    )

#    Creates an instance of BedrockEmbeddings for embedding generation, using Amazon's titan-embed-text-v1 model.
    embeddings = BedrockEmbeddings(
        model_id="amazon.titan-embed-text-v1",
        client=bedrock_runtime,
        region_name="us-east-1",
    )
#    VectorstoreIndexCreator is used to create a searchable index of the document's content.
    index_creator = VectorstoreIndexCreator(
        vectorstore_cls=FAISS,
        embedding=embeddings,
    )
#    The generated index files (FAISS and pickle formats) are saved locally and then uploaded to S3 in the user's specific directory.
    index_from_loader = index_creator.from_loaders(loaders)

    index_from_loader.vectorstore.save_local("/tmp")

    s3.upload_file(
        "/tmp/index.faiss", BUCKET, f"{user_id}/index.faiss"
    )
    s3.upload_file("/tmp/index.pkl", BUCKET, f"{user_id}/index.pkl")

#    Finally, it updates the document's status to "READY" in DynamoDB.
    set_doc_status(user_id, document_id, "READY")
