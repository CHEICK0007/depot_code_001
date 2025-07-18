import queue
import json
import sounddevice as sd
from vosk import Model, KaldiRecognizer

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 1️⃣ VOSK — Reconnaissance vocale
vosk_model_path = r"C:\Users\PC\Documents\MES_CODES\python\projet_IA\vosk-model-small-fr-0.22\vosk-model-small-fr-0.22"
vosk_model = Model(vosk_model_path)
recognizer = KaldiRecognizer(vosk_model, 16000)
audio_queue = queue.Queue()

def audio_callback(indata, frames, time, status):
    if status:
        print("⚠️", status)
    audio_queue.put(bytes(indata))

# 2️⃣ LangChain — Chargement du RAG
loader = TextLoader("restaurant_info.json", encoding="utf-8")
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=50)
docs = splitter.split_documents(documents)

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(docs, embedding)
retriever = vectorstore.as_retriever()
llm = Ollama(model="gemma3:1b")  # ou "gemma:1b-instruct"

# 🔧 Prompt contrôlé (phrases simples, pas de redirection)
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
Tu es une secrétaire. Réponds uniquement à partir des informations suivantes, avec des phrases simples et précises.

 Ne redirige jamais vers un site web ou un numéro de téléphone.
 Si tu ne sais pas, dis : "Je n’ai pas cette information."

Informations :
{context}

Question : {question}
Réponse :
"""
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt}
)

# 3️⃣ Boucle : écoute → texte → réponse
print("🎙️ Assistant en écoute... (Ctrl+C pour arrêter)")
with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype='int16',
                       channels=1, callback=audio_callback):
    try:
        while True:
            data = audio_queue.get()
            if recognizer.AcceptWaveform(data):
                result = json.loads(recognizer.Result())
                phrase = result.get("text", "").strip()
                if phrase:
                    print("🗣️ Question :", phrase)
                    response = qa_chain.run(phrase)
                    print("🤖 Réponse :", response)
    except KeyboardInterrupt:
        print("\n Assistant arrêté.")
