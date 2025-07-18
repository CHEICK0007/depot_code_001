from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

loader = TextLoader("restaurant_info.json", encoding="utf-8")
documents = loader.load()

#  Découpe le texte en morceaux plus courts
text_splitter = CharacterTextSplitter(chunk_size=250, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

#  Embeddings + indexation
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(docs, embedding)

#  Construction de la chaîne RAG
retriever = vectorstore.as_retriever()
llm = Ollama(model="gemma3:1b")
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# Interrogez votre base !
while True:
    query = input("Posez une question : ")
    if query.lower() in ["exit", "quit"]:
        break
    result = qa_chain.run(query)
    print("Réponse :", result)
