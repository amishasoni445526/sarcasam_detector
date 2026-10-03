# 🎭 Sarcasm Detection System

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1q7CZXoZUgMmTPGPcP5n959FnPJqBCNxn)

A robust machine learning project that detects whether a given text is sarcastic or non-sarcastic. The system provides a simple, interactive web interface where users can type sentences and immediately get a prediction, along with the confidence score of the model.

## ✨ Features

- **Accurate NLP Models**: Leverages Hugging Face's `transformers` library, specifically using RoBERTa-based models (e.g., `cardiffnlp/twitter-roberta-base-sarcasm`) trained on social media text to accurately classify sarcasm.
- **Hybrid Detection Approach**: Includes a hybrid detector that supplements the ML model predictions with a rule-based pattern matching approach, significantly improving accuracy on common tricky sarcastic phrasing (e.g., "Oh great, another Monday").
- **Interactive UI**: Built with `Gradio`, the project features an easy-to-use web interface complete with styling, examples, and emoji indicators (😏/😊) depending on the result.
- **Confidence Scoring**: Returns not just a binary prediction, but also confidence percentages and raw model scores, allowing users to understand the model's certainty.

## 🛠️ Technology Stack

- **Python 3**
- **PyTorch**: Used as the tensor backend for running the deep learning models.
- **Transformers**: For downloading and running the pre-trained NLP classification models.
- **Gradio**: For generating the interactive web UI and API.

## 🚀 Getting Started

### Prerequisites

Ensure you have Python installed on your system. It's recommended to use a virtual environment. 

### Installation

1. Clone or download this repository.
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

### Usage

1. Open the provided Jupyter Notebook (`sarcasam detector.ipynb`).
2. Run the cells sequentially.
3. The notebook will automatically download the necessary models from Hugging Face and start a local Gradio server.
4. A public/local link will be printed in the output. Click the link to open the web interface in your browser and start testing sarcastic sentences!

## 🧠 How it Works

The project consists of multiple stages defined in the notebook:
1. **Model Loading**: It attempts to load `cardiffnlp/twitter-roberta-base-sarcasm`. If unavailable, it falls back to an alternative RoBERTa model.
2. **Text Processing**: Input text is tokenized with `AutoTokenizer` and fed into the PyTorch `AutoModelForSequenceClassification`.
3. **Prediction Extraction**: The output logits are passed through a softmax function to obtain probabilities for the "sarcastic" and "non-sarcastic" classes.
4. **Hybrid Fallback (Optional)**: If the standard model struggles, the `HybridSarcasmDetector` class can be used. It looks for known sarcastic patterns and keywords while adjusting the ML score to provide a guaranteed output for tough edge cases.
