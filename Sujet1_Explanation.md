# Detailed Explanation of the Multilingual Wikipedia Translation Pipeline Code

## 1. Package Installation
The first command installs necessary Python packages:
- torch, transformers: for deep learning and translation models
- requests, tqdm: for HTTP requests and progress bars
- gradio: for the interactive web UI
- evaluate, datasets: for translation quality metrics
- unbabel-comet, sacrebleu: additional evaluation tools


## 2. Configuration and Imports
- Imports standard Python libraries (os, re, json, time).
- Imports tqdm for progress bars.
- Requests and BeautifulSoup for web requests and HTML parsing.
- Pandas and NumPy for data manipulation.
- PyTorch and Hugging Face transformers for model loading and translation.
- Evaluate and datasets libraries for automatic evaluation of translations.
- Gradio for building an interactive interface.
- Device configuration to use CUDA (NVIDIA GPU), Apple MPS (Apple Silicon GPU), or CPU.


## 3. WikipediaFrenchArticleScraper Class
Purpose: Fetch random French Wikipedia articles and clean their content.

- `__init__`: Sets API base URL, max articles to fetch, batch size, and a requests session with user-agent headers.
- `_get_random_article_titles(count)`: Fetches random article titles from Wikipedia API.
- `_get_article_content(title)`: Retrieves plain-text content for a given article title.
- `_clean_content(text)`: Cleans the text by removing references like [1], multiple line breaks, and extra spaces.
- `fetch_articles()`: Retrieves articles in batches (default batch size 50), filters out short articles (<50 words), and returns a Pandas DataFrame containing titles and cleaned content.

API respects Wikipedia limits by capping max articles at 500 and adding delay.


## 4. MultilingualWikipediaTranslator Class
Purpose: Translate articles into multiple African languages using Facebook's NLLB-200 model.

- Language config dictionary with language codes and language families.
- Loads tokenizer and model from Hugging Face.
- Loads BLEU and COMET evaluation metrics.
- `_batch_translate(texts, target_lang)`: Tokenizes and translates a batch of texts, forcing the beginning-of-sequence token to the target language code, returns decoded translations.
- `translate_articles(df, target_langs)`: Translates all articles in a DataFrame into specified languages, processing in batches of 8 to manage memory, returns concatenated DataFrame with translations and language labels.


## 5. TranslationEvaluator Class
Purpose: Evaluate translation quality.

- Loads BLEU, COMET, and TER metrics.
- `evaluate_translation(source_texts, translated_texts, reference_texts)`:  
  - Calculates BLEU if reference translations are available.  
  - Calculates COMET score using source and machine-translated texts.  
  - Returns a dictionary of evaluation scores.
- `cultural_adaptation(text, lang)`: Placeholder for future post-processing to adapt translations culturally.


## 6. TranslationUI Class
Purpose: Create an interactive web interface for translation using Gradio.

- Displays a dropdown to select target language.
- Shows a random French Wikipedia article.
- Buttons for fetching a new article and translating it.
- Shows translated article and quality metrics.
- Calls backend methods to fetch articles and translate them on demand.



## 7. run_full_pipeline Function
Coordinates the full workflow:

1. Fetches 100 French Wikipedia articles.  
2. Initializes the translation model.  
3. Translates articles into Fon, Ewe, Dendi, Yoruba.  
4. Saves results to a Parquet file.  
5. Launches the Gradio interactive interface with sharing enabled.


## 8. Script Entry Point
When executed as a script, calls `run_full_pipeline()` to run everything.


# Summary
This code implements a complete pipeline from data fetching to translation, evaluation, and user interaction — enabling exploration of African language translations of French Wikipedia articles with state-of-the-art multilingual models.


# Usage Requirements
- Python 3.8+  
- GPU recommended (CUDA or Apple MPS) for translation speed  
- Internet connection for Wikipedia API and model downloads



