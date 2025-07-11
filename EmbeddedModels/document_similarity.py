from langchain_huggingface import HuggingFaceEmbeddings
embedding=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

from sklearn.metrics.pairwise import cosine_similarity

document=["Delhi is capital of India",
          "Delhi is capital of India and it is the largest city in India",
          "Mumbai is the financial capital of India",
          "Gurgaon is a city near Delhi"]

query="What is the financial capital of India?"

document_embedding=embedding.embed_documents(document)
query_embedding=embedding.embed_query(query)

similarity_scores = cosine_similarity([query_embedding], document_embedding)[0]
index,score=(sorted(list(enumerate(similarity_scores)),key=lambda x:x[1]))[-1]

print(query)
print(document[index])
print("Similarity Score:", score)


