!pip install torch transformers requests gradio tqdm

# test_wikipedia_translation.py
import requests
import torch
from transformers import pipeline
import gradio as gr
from tqdm import tqdm

# 1. Récupération d'articles Wikipédia simplifiée
def get_wikipedia_article(title="Afrique"):
    """Récupère un seul article Wikipédia en français"""
    url = f"https://fr.wikipedia.org/w/api.php?action=query&format=json&prop=extracts&titles={title}&explaintext=1"
    response = requests.get(url)
    data = response.json()
    page = next(iter(data['query']['pages'].values()))
    return page['extract']

# 2. Traduction avec un modèle plus léger
class LocalTranslator:
    def __init__(self):
        # Modèle plus petit pour tester localement
        self.model = pipeline(
            "translation",
            model="facebook/nllb-200-distilled-600M",
            device="cuda" if torch.cuda.is_available() else "cpu"
        )

        self.lang_codes = {
            'fon': 'fon_Latn',
            'ewe': 'ewe_Latn',
            'yoruba': 'yor_Latn'
        }

    def translate(self, text, target_lang):
        """Traduction simple pour test local"""
        if target_lang not in self.lang_codes:
            raise ValueError(f"Langue non supportée. Choisissez parmi {list(self.lang_codes.keys())}")

        result = self.model(
            text,
            src_lang="fra_Latn",
            tgt_lang=self.lang_codes[target_lang],
            max_length=400
        )
        return result[0]['translation_text']

# 3. Interface de test
def test_translation():
    translator = LocalTranslator()

    # Récupération d'un article
    article = get_wikipedia_article()
    print(f"Article source (extrait):\n{article[:200]}...\n")

    # Test de traduction
    target_lang = "fon"  # Changez ici pour tester différentes langues
    translated = translator.translate(article[:500], target_lang)  # On ne traduit que les 500 premiers caractères pour le test

    print(f"\nTraduction en {target_lang}:\n{translated[:200]}...")

# 4. Interface Gradio simplifiée
def create_demo():
    translator = LocalTranslator()

    def translate_article(article, lang):
        try:
            translated = translator.translate(article, lang)
            return translated
        except Exception as e:
            return f"Erreur: {str(e)}"

    demo = gr.Interface(
        fn=translate_article,
        inputs=[
            gr.Textbox(label="Article Français", lines=10),
            gr.Dropdown(["fon", "ewe", "yoruba"], label="Langue Cible")
        ],
        outputs=gr.Textbox(label="Traduction", lines=10),
        examples=[
            ["La France est un pays d'Europe.", "yoruba"],
            ["Le Bénin est riche en culture.", "fon"]
        ]
    )

    return demo

if __name__ == "__main__":
    # Test rapide en console
    test_translation()

    # Démarrage de l'interface (décommentez pour tester)
    demo = create_demo()
    demo.launch()