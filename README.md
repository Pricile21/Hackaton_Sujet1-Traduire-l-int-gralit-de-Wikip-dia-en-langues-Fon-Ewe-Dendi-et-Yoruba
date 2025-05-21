# Multilingual Wikipedia Translation Pipeline

## Description

This project provides a complete pipeline to fetch random articles from French Wikipedia, automatically translate them into multiple African languages (Fon, Ewe, Dendi, Yoruba) using Facebook’s NLLB-200 model, and then evaluate the quality of these translations.

It also offers a simple interactive web interface (via Gradio) to test article translations on demand.

## Key Features

- Fetch random articles from French Wikipedia with text cleaning  
- GPU-optimized automatic translation into multiple African languages  
- Automatic translation quality evaluation using BLEU, COMET, and TER metrics  
- Interactive user interface with Gradio for real-time translation testing  
- Results saved in Parquet file format  

## Technologies and Libraries Used

- Python 3.8+  
- PyTorch, Transformers (Hugging Face)  
- Requests, BeautifulSoup  
- Pandas, NumPy  
- Evaluate, Datasets  
- Gradio  
- Tqdm  

## Installation

1. Clone this repository or download the script.

2. Install the dependencies:

```bash
pip install -r requirements.txt