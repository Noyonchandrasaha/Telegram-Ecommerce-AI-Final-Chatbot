import json
from langchain_core.documents import Document
from langchain.vectorstores import FAISS
from app.core.config import hf_embeddings

# Load your JSON file manually
file_path = "app/db/grocery_products_50.json"

with open(file_path, 'r', encoding='utf-8') as f:
    raw_json = json.load(f)

# Assume the main content is in raw_json[0]['content']
records = raw_json[0]['content']

documents = []
for record in records:
    # Create formatted string for page_content (similar to your jq schema)
    content = f"""Product ID: {record.get('product_id', '')}
Product Name: {record.get('product_name', '')}
Category: {record.get('category', '')}
Sub Category: {record.get('sub_category', '')}
Description: {record.get('description', '')}
Price: {record.get('price', 0)} {record.get('currency', '')}
Discount: {record.get('discount', 0)}%
Stock Status: {record.get('stock_status', '')}
Unit: {record.get('unit', '')}
Brand: {record.get('brand', '')}
Origin: {record.get('origin', '')}
Tags: {', '.join(record.get('tags', []))}
Rating: {record.get('rating', 0)}
Reviews:
"""
    for review in record.get('reviews', []):
        content += f"  - User: {review.get('user', '')}, Comment: {review.get('comment', '')}, Rating: {review.get('rating', 0)}\n"

    content += f"Recommended For: {', '.join(record.get('recommended_for', []))}"

    # Store the original JSON record as metadata
    metadata = record

    documents.append(Document(page_content=content, metadata=metadata))

# Now documents have formatted content and full metadata
for doc in documents:
    print("Page Content:")
    print(doc.page_content)
    print("\nMetadata:")
    print(doc.metadata)
    print("="*50)

faiss_index = FAISS.from_documents(documents, hf_embeddings)

# Optionally save to disk
faiss_index.save_local("faiss_grocery_index_hf")
print("FAISS index created and saved successfully.")