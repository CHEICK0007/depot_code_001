import pyttsx3
import time

# 🔁 Initialise un moteur TTS à chaque lecture (corrige les blocages)
def lire_texte(texte):
    engine = pyttsx3.init()
    engine.setProperty('rate', 210)  # vitesse

    # ✅ Recherche d'une voix française
   
    engine.setProperty('voice', "HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_FR-FR_HORTENSE_11.0")


    # 🗣️ Lecture
    engine.say(texte)
    engine.runAndWait()

    # 🔒 Nettoie proprement
    engine.stop()
