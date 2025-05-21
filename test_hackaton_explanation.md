# Detailed Explanation of the Simple Wikipedia Translation Script

## 1. Package Installation
- Installs essential libraries: torch, transformers, requests, gradio, tqdm.

## 2. Fetching a Wikipedia Article
- Function `get_wikipedia_article(title="Afrique")`:
  - Calls the French Wikipedia API to retrieve plain-text extract for the given title.
  - Returns the article content as a string.

## 3. LocalTranslator Class
- Uses Hugging Face's pipeline API with the lightweight `"facebook/nllb-200-distilled-600M"` model for translation.
- Supports three target African languages with their NLLB language codes: Fon, Ewe, Yoruba.
- Method `translate(text, target_lang)`:
  - Validates target language.
  - Translates the given French text snippet to the target language.
  - Returns translated text.

## 4. Test Translation Function
- `test_translation()` demonstrates:
  - Fetching the Wikipedia article "Afrique".
  - Printing a snippet of the source article.
  - Translating the first 500 characters to Fon.
  - Printing a snippet of the translation.

## 5. Gradio Interface
- `create_demo()` creates a simple web UI with:
  - Input textbox for French article text.
  - Dropdown to select target language.
  - Output textbox showing the translation.
  - Preloaded example inputs for quick testing.
- Handles translation errors gracefully.

## 6. Script Execution
- Runs the test translation on script start.
- Launches the Gradio demo interface (currently uncommented).



# Summary
This script is a lightweight, easy-to-run tool for fetching a French Wikipedia article and translating it locally into African languages using a smaller NLLB model, complete with a simple user interface for manual testing.



# Requirements
- Python 3.8+
- GPU recommended but CPU supported
- Internet connection for Wikipedia API and model download



# Usage
Run the script to test translation in console and/or launch the web interface for interactive use.


