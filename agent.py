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


# comfigurattion

# 📄 Charge un document (ex: fichier JSON ou texte)
loader = TextLoader("restaurant_info.json", encoding="utf-8")
documents = loader.load()

# ✂️ Découpe le texte en morceaux plus courts
text_splitter = CharacterTextSplitter(chunk_size=250, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

# 🧠 Embeddings + indexation
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(docs, embedding)

# 🔍 Construction de la chaîne RAG
retriever = vectorstore.as_retriever()
llm = Ollama(model="gemma3:1b")



# Template & RAG

prompt_template1 = PromptTemplate(
    input_variables=["context", "question"],
    template="""
Tu es un assistant de restaurant qui ne salue JAMAIS. Réponds strictement selon ces règles :

1. **Sources** : Utilise uniquement ces données :
{context}

2. **Format** : 
- Pas de demande d'informations personnelles

3. **Restrictions** :

- Jamais de sites web/numéros de téléphone
- Jamais de "veuillez contacter..." ou "vous pouvez trouver..."
- ne donne pas de réponses si tu n'en n'as pas 
Question : {question}
Réponse :
"""
)

qa_chain1 = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt_template1}
)


# discussion Vocal

# 💾 Initialise un historique de conversation
conversation_history = ""
q = queue.Queue()
def callback(indata, frames, time, status):
    if status:
        print("⚠️", status)
    q.put(bytes(indata))

model = Model(r"C:\Users\PC\Documents\MES_CODES\python\projet_IA\vosk-model-small-fr-0.22\vosk-model-small-fr-0.22")
recognizer = KaldiRecognizer(model, 16000)

print("🎙️ Assistant en écoute... (Ctrl+C pour arrêter)")
print("🎙️ Assistant en écoute... (Ctrl+C pour arrêter)")

with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype='int16',
                       channels=1, callback=callback):
    try:
        info_dis = []
        while True:
            data = q.get()
            if recognizer.AcceptWaveform(data):
                result = json.loads(recognizer.Result())
                phrase = result.get("text", "").strip()
                if phrase:
                    print("🗣️ Question :", phrase)
                    info_dis.append(phrase)

                    # 🔁 On construit le prompt avec l'historique
                    full_prompt = f"""
                    [Historique de la conversation jusqu'à présent]
                    {conversation_history}

                    Maintenant, réponds à SEULEMENT et cette nouvelle question de l'utilisateur :

                    [Nouvelle question]
                    {phrase}
                    """
                    # 🧠 Requête au modèle
                    response = qa_chain1.run(full_prompt)

                    # 🔂 Ajoute l'échange à l'historique pour créer une chaîne de questions-réponses (conversation)
                    conversation_history += f"\nUtilisateur : {phrase}\nAssistant : {response}"

                    print("🤖 Réponse :", response)
                    lire_texte(response)

    except KeyboardInterrupt:
     print("\n🛑 Assistant arrêté.")
