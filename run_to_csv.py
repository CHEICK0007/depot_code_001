

# 1. Modèle Ollama configuré
llm = Ollama(model="gemma3:1B", temperature=0.1)  # Température plus basse pour moins de variations

# 2. Prompt optimisé avec instructions claires
prompt_template2 = PromptTemplate(
    input_variables=["text"],
    template="""
Tu es un assistant qui extrait des informations avec précision. 
Retourne UNIQUEMENT un objet valide contenant :
- "nom" (prénom + nom)
- "ville" (nom de ville)
- "prix" (nombre )

Si une information est manquante, utilise null.

Exemples valides :
Input: "Je m'appelle Jean Martin de Paris"
Output: {{"nom": "Jean Martin", "ville": "Paris", "prix": null}}

Input: "Prix: 50€, Ville: Lyon"
Output: {{"nom": null, "ville": "Lyon", "prix": "50"}}

Texte à analyser :
{text}

""")

# 3. Pipeline optimisé avec gestion d'erreurs
def extract_infos(text):
    try:
        # Nettoyage du texte
        cleaned_text = re.sub(r"\s+", " ", text).strip()
        
        # Appel au LLM
        chain = LLMChain(llm=llm, prompt=prompt_template2)
        response = chain.run(text=cleaned_text)
        
        # Extraction depuis la réponse
        json_match = re.search(r"\{.*\}", response, re.DOTALL)
        if not json_match:
            raise ValueError("Format JSON non trouvé")
            
        data = json.loads(json_match.group())
        
        # Validation et formatage
        def clean_field(value, pattern):
            if not value or value == "null":
                return None
            value = str(value).strip()
            return value if re.fullmatch(pattern, value) else None
            
        return {
            "nom": clean_field(data.get("nom"), r"^[A-Za-zÀ-ÿ\s-]{3,}$"),
            "ville": clean_field(data.get("ville"), r"^[A-Za-zÀ-ÿ\s-]{2,}$"),
            "prix": clean_field(data.get("prix"), r"^\d+[\.,]?\d*$")
        }
        
    except Exception as e:
        print(f"Erreur d'extraction: {e}")
        return {"nom": None, "ville": None, "prix": None}



def run_to_csv(test_cases, output_file="output.csv"):
    # Création d'un buffer CSV en mémoire
    csv_buffer = StringIO()
    writer = csv.DictWriter(csv_buffer, 
                          fieldnames=["nom", "ville", "prix"],
                          delimiter=";",
                          quoting=csv.QUOTE_MINIMAL)
    
    # Écriture de l'en-tête
    writer.writeheader()
    
    # Traitement des cas de test
    for text in test_cases:
        result = extract_infos(text)
        writer.writerow(result)
    
    # Réinitialisation du buffer et écriture dans le fichier
    csv_buffer.seek(0)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(csv_buffer.getvalue())
    
    print(f"Résultats exportés dans {output_file}")

# Exécution et export
run_to_csv(info_dis)
