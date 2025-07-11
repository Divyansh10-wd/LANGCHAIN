from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

messages=[
    SystemMessage(content="You are a helpful assistant that provides information"),
    HumanMessage(content="What is the capital of India?"),
    ]

result=model.invoke(messages)

messages.append(AIMessage(content=result.content))
print(messages)