# Téléchargement des packages
!pip install torch transformers requests gradio tqdm evaluate datasets unbabel-comet sacrebleu

# %% [markdown]
# ## 1. Configuration et Importations
import os
import re
import json
import time
from tqdm import tqdm
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
import evaluate
from datasets import Dataset, concatenate_datasets
import gradio as gr

# Configuration des devices
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# %% [markdown]
# ## 2. Classe pour la Récupération des Articles Wikipédia
class WikipediaFrenchArticleScraper:
    def __init__(self, max_articles=1000, batch_size=50):
        self.base_url = "https://fr.wikipedia.org/w/api.php"
        self.max_articles = max_articles
        self.batch_size = batch_size
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'Mozilla/5.0'})

    def _get_random_article_titles(self, count):
        """Récupère des titres d'articles aléatoires"""
        params = {
            'action': 'query',
            'format': 'json',
            'list': 'random',
            'rnnamespace': 0,
            'rnlimit': count
        }
        response = self.session.get(self.base_url, params=params)
        data = response.json()
        return [article['title'] for article in data['query']['random']]

    def _get_article_content(self, title):
        """Récupère le contenu d'un article spécifique"""
        params = {
            'action': 'query',
            'format': 'json',
            'prop': 'extracts',
            'titles': title,
            'explaintext': True,
            'exsectionformat': 'plain'
        }
        response = self.session.get(self.base_url, params=params)
        data = response.json()
        page = next(iter(data['query']['pages'].values()))
        return page.get('extract', '')

    def _clean_content(self, text):
        """Nettoie le contenu de l'article"""
        # Suppression des références [1], [2], etc.
        text = re.sub(r'\[\d+\]', '', text)
        # Suppression des sauts de ligne multiples
        text = re.sub(r'\n+', '\n', text)
        # Suppression des espaces multiples
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def fetch_articles(self):
        """Récupère les articles par lots avec une barre de progression"""
        all_articles = []
        title_count = min(self.max_articles, 500)  # Limite API

        with tqdm(total=title_count, desc="Récupération des articles") as pbar:
            for _ in range(0, title_count, self.batch_size):
                batch_size = min(self.batch_size, title_count - len(all_articles))
                titles = self._get_random_article_titles(batch_size)

                for title in titles:
                    content = self._get_article_content(title)
                    clean_content = self._clean_content(content)

                    if len(clean_content.split()) > 50:  # Filtre les articles trop courts
                        all_articles.append({
                            'title': title,
                            'content': clean_content,
                            'language': 'fr'
                        })
                    pbar.update(1)

                time.sleep(1)  # Respect des limites de l'API

        return pd.DataFrame(all_articles)

# %% [markdown]
# ## 3. Pipeline de Traduction Multilingue
class MultilingualWikipediaTranslator:
    LANG_CONFIG = {
        'fon': {'code': 'fon_Latn', 'family': 'Niger-Congo'},
        'ewe': {'code': 'ewe_Latn', 'family': 'Niger-Congo'},
        'dendi': {'code': 'dje_Latn', 'family': 'Songhai'},
        'yoruba': {'code': 'yor_Latn', 'family': 'Niger-Congo'}
    }

    def __init__(self, model_name="facebook/nllb-200-3.3B"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(DEVICE)
        self.bleu = evaluate.load('bleu')
        self.comet = evaluate.load('comet')

    def _batch_translate(self, texts, target_lang):
        """Traduction par lots pour meilleure performance"""
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(DEVICE)

        translated = self.model.generate(
            **inputs,
            forced_bos_token_id=self.tokenizer.lang_code_to_id[target_lang],
            max_length=512,
            num_beams=3
        )

        return [self.tokenizer.decode(t, skip_special_tokens=True) for t in translated]

    def translate_articles(self, df, target_langs=None):
        """Traduit les articles vers plusieurs langues"""
        if target_langs is None:
            target_langs = self.LANG_CONFIG.keys()

        results = []
        for lang in target_langs:
            lang_code = self.LANG_CONFIG[lang]['code']

            # Découpage en lots pour éviter les OOM
            batch_size = 8
            translated_contents = []

            for i in tqdm(range(0, len(df), batch_size), desc=f"Traduction en {lang}"):
                batch = df['content'].iloc[i:i+batch_size].tolist()
                translated_batch = self._batch_translate(batch, lang_code)
                translated_contents.extend(translated_batch)

            # Création d'un nouveau DataFrame pour la langue cible
            lang_df = df.copy()
            lang_df['content'] = translated_contents
            lang_df['language'] = lang
            lang_df['target_lang'] = lang_code

            results.append(lang_df)

        return pd.concat(results, ignore_index=True)

# %% [markdown]
# ## 4. Évaluation et Post-Traitement
class TranslationEvaluator:
    def __init__(self):
        self.metrics = {
            'bleu': evaluate.load('bleu'),
            'comet': evaluate.load('comet'),
            'ter': evaluate.load('ter')
        }

    def evaluate_translation(self, source_texts, translated_texts, reference_texts=None):
        """Évalue la qualité des traductions"""
        results = {}

        # Calcul BLEU si des références sont disponibles
        if reference_texts:
            results['bleu'] = self.metrics['bleu'].compute(
                predictions=translated_texts,
                references=[[ref] for ref in reference_texts]
            )['bleu']

        # Calcul COMET (ne nécessite pas de références)
        comet_inputs = [{'src': src, 'mt': mt} for src, mt in zip(source_texts, translated_texts)]
        results['comet'] = self.metrics['comet'].compute(comet_inputs)['mean_score']

        return results

    def cultural_adaptation(self, text, lang):
        """Post-traitement culturel spécifique à la langue"""
        # À implémenter avec des règles spécifiques
        return text

# %% [markdown]
# ## 5. Interface Utilisateur
class TranslationUI:
    def __init__(self, translator, evaluator):
        self.translator = translator
        self.evaluator = evaluator

    def create_interface(self):
        """Crée une interface Gradio interactive"""
        with gr.Blocks(title="Traduction Wikipédia Multilingue") as demo:
            gr.Markdown("# Traduction d'articles Wikipédia Français")

            with gr.Row():
                with gr.Column():
                    lang_choice = gr.Dropdown(
                        list(self.translator.LANG_CONFIG.keys()),
                        label="Langue cible"
                    )
                    translate_btn = gr.Button("Traduire", variant="primary")

                with gr.Column():
                    output_content = gr.Textbox(label="Article Traduit", lines=20)
                    metrics = gr.JSON(label="Métriques de Qualité")

            # Section pour afficher un article aléatoire
            with gr.Accordion("Article Français Aléatoire"):
                sample_article = gr.Textbox(label="Article Source", lines=10, interactive=False)
                refresh_btn = gr.Button("Nouvel Article")

            # Callbacks
            refresh_btn.click(
                self._get_random_article,
                outputs=sample_article
            )

            translate_btn.click(
                self._translate_article,
                inputs=[sample_article, lang_choice],
                outputs=[output_content, metrics]
            )

        return demo

    def _get_random_article(self):
        """Récupère un nouvel article aléatoire"""
        scraper = WikipediaFrenchArticleScraper(max_articles=1)
        article = scraper.fetch_articles().iloc[0]
        return article['content']

    def _translate_article(self, content, lang):
        """Traduit l'article et calcule les métriques"""
        lang_code = self.translator.LANG_CONFIG[lang]['code']

        # Traduction
        translated = self.translator._batch_translate([content], lang_code)[0]

        # Post-traitement culturel
        adapted = self.evaluator.cultural_adaptation(translated, lang)

        # Évaluation (sans référence dans ce cas)
        metrics = self.evaluator.evaluate_translation([content], [adapted])

        return adapted, metrics

# %% [markdown]
# ## 6. Pipeline Complet
def run_full_pipeline():
    # 1. Récupération des articles
    print("Étape 1: Récupération des articles Wikipédia français...")
    scraper = WikipediaFrenchArticleScraper(max_articles=100)  # À augmenter en production
    french_articles = scraper.fetch_articles()

    # 2. Initialisation du traducteur
    print("\nÉtape 2: Initialisation du modèle de traduction...")
    translator = MultilingualWikipediaTranslator()

    # 3. Traduction multilingue
    print("\nÉtape 3: Traduction des articles...")
    translated_articles = translator.translate_articles(
        french_articles,
        target_langs=['fon', 'ewe', 'dendi', 'yoruba']
    )

    # 4. Sauvegarde des résultats
    print("\nÉtape 4: Sauvegarde des résultats...")
    os.makedirs("results", exist_ok=True)
    translated_articles.to_parquet("results/translated_articles.parquet")

    # 5. Lancement de l'interface
    print("\nÉtape 5: Lancement de l'interface...")
    evaluator = TranslationEvaluator()
    ui = TranslationUI(translator, evaluator)
    ui.create_interface().launch(share=True)

# %% [markdown]
# ## 7. Exécution
if __name__ == "__main__":
    run_full_pipeline()