import boto3
import pymongo
import json
import os
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.mongodb import MongoDBAtlas
from langchain.chains import RetrievalQA
from langchain.chat_models import Bedrock

# AWS Bedrock setup
bedrock_client = boto3.client("bedrock-runtime", region_name="us-east-1")

# MongoDB Atlas connection
MONGO_URI = os.environ["MONGO_URI"]
client = pymongo.MongoClient(MONGO_URI)
db = client["rag_knowledge_base"]
collection = db["documents"]

# Role-Based Access Control (RBAC)
def check_user_permissions(user_id, team_id):
    user_roles = db["user_roles"].find_one({"user_id": user_id})
    if not user_roles or team_id not in user_roles.get("teams", []):
        return False
    return True

# Initialize Vector Store
vector_store = MongoDBAtlas(
    connection_string=MONGO_URI,
    database_name="rag_knowledge_base",
    collection_name="vector_store",
    embedding_function=OpenAIEmbeddings(model_name="text-embedding-ada-002")
)

# Function to retrieve relevant context
def retrieve_context(query, team_id, use_case):
    query_filter = {"team_id": team_id, "use_case": use_case}
    results = vector_store.similarity_search(query, k=3, filter=query_filter)
    return results

# Function to generate response from AWS Bedrock
def generate_response(query, context):
    payload = json.dumps({
        "inputText": f"{query}\nContext: {context}",
        "modelId": "amazon.titan-text-v1"
    })
    response = bedrock_client.invoke_model(
        body=payload,
        modelId="amazon.titan-text-v1"
    )
    return json.loads(response["body"].read().decode("utf-8"))["outputText"]

# Secure API Gateway wrapper
def process_request(user_id, team_id, use_case, query):
    if not check_user_permissions(user_id, team_id):
        return {"statusCode": 403, "body": json.dumps({"error": "Unauthorized access"})}
    
    if not query:
        return {"statusCode": 400, "body": json.dumps({"error": "Query parameter is required"})}
    
    retrieved_context = retrieve_context(query, team_id, use_case)
    context_text = "\n".join([doc["text"] for doc in retrieved_context])
    response = generate_response(query, context_text)
    
    return {
        "statusCode": 200,
        "body": json.dumps({"response": response})
    }

# AWS Lambda handler
def lambda_handler(event, context):
    query_params = event.get("queryStringParameters", {})
    user_id = query_params.get("user_id", "")
    team_id = query_params.get("team_id", "default_team")
    use_case = query_params.get("use_case", "default_use_case")
    query = query_params.get("query", "")
    
    return process_request(user_id, team_id, use_case, query)

# Admin Console for Teams to Manage RAG Jobs
def create_rag_job(user_id, team_id, use_case, documents):
    if not check_user_permissions(user_id, team_id):
        return {"statusCode": 403, "body": json.dumps({"error": "Unauthorized access"})}
    
    for doc in documents:
        doc["team_id"] = team_id
        doc["use_case"] = use_case
        collection.insert_one(doc)
    return {"statusCode": 200, "body": json.dumps({"message": "RAG job created successfully"})}

# Batch Processing for Large Document Ingestion
def batch_ingest_documents(user_id, team_id, use_case, documents):
    if not check_user_permissions(user_id, team_id):
        return {"statusCode": 403, "body": json.dumps({"error": "Unauthorized access"})}
    
    collection.insert_many([{**doc, "team_id": team_id, "use_case": use_case} for doc in documents])
    return {"statusCode": 200, "body": json.dumps({"message": "Batch document ingestion completed"})}

# AWS Lambda handler for Admin Console
def admin_lambda_handler(event, context):
    body = json.loads(event.get("body", "{}"))
    action = body.get("action", "")
    user_id = body.get("user_id", "")
    team_id = body.get("team_id", "")
    use_case = body.get("use_case", "")
    documents = body.get("documents", [])
    
    if action == "create_rag_job":
        return create_rag_job(user_id, team_id, use_case, documents)
    elif action == "batch_ingest":
        return batch_ingest_documents(user_id, team_id, use_case, documents)
    
    return {"statusCode": 400, "body": json.dumps({"error": "Invalid action"})}
