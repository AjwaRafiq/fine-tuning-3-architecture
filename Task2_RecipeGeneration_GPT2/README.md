

Task 2 — Decoder Model (GPT-2): Recipe Generation
================================================

Objective:
-----------
Fine-tune the GPT-2 language model to generate coherent and creative cooking recipes based on either:
- A list of ingredients, or
- A recipe title.

Dataset:
--------
Source: https://www.kaggle.com/datasets/nazmussakibrupol/3a2mext/data

The dataset contains:
- Title — dish name
- Ingredients — list of required ingredients
- Instructions / Steps — procedure for preparing the dish

Make sure to download the dataset manually from Kaggle and place it in your 'data/' folder.

Example:
title: Pancakes
ingredients: flour, milk, egg, sugar, butter
instructions: Mix ingredients, pour into pan, cook until golden brown.

-------------------------------------------------------------

Environment Setup:
------------------
pip install transformers datasets accelerate torch sentencepiece sacrebleu rouge_score streamlit

-------------------------------------------------------------

Project Structure:
------------------
Recipe_Generation_GPT2/
│
├── data/
│   └── recipes.csv
│
├── tokenize_and_format.py
├── train_gpt2.py
├── evaluate.py
├── app_streamlit.py
│
├── outputs/
│   └── gpt2_recipes/
│
└── README.txt

-------------------------------------------------------------

Steps to Run:
-------------

1) Tokenization and Dataset Formatting
--------------------------------------
python tokenize_and_format.py --input_csv data/recipes.csv --output_dir data/tokenized --max_length 512

2) Fine-Tune GPT-2
------------------
python train_gpt2.py --tokenized_dataset data/tokenized --model_name gpt2 --output_dir outputs/gpt2_recipes --epochs 3 --batch_size 2

3) Evaluation
-------------
python evaluate.py --model_path outputs/gpt2_recipes --test_csv data/recipes.csv --sample 100 --save_n 20

Outputs:
- generation_examples.json with ingredients, reference, generated recipe, and BLEU/ROUGE scores

4) Streamlit App
----------------
streamlit run app_streamlit.py --server.port=8501

Open browser at http://localhost:8501

-------------------------------------------------------------

Evaluation Methods:
-------------------
Automated: BLEU and ROUGE metrics compare generated recipes to ground truth.
Human Evaluation: Participants rate outputs for relevance, coherence, and completeness.
Qualitative: Manual inspection for ingredient-recipe consistency.

-------------------------------------------------------------

Key Components:
---------------
Tokenizer: GPT-2 tokenizer with padding token <|pad|>
Model: GPT-2 fine-tuned as a decoder
Training Loop: Implemented using Hugging Face Trainer
Metrics: BLEU (sacrebleu), ROUGE (rouge_score)
UI: Streamlit app

-------------------------------------------------------------

Example Generation:
-------------------
Input Prompt:
Title: Chocolate Cake
Ingredients:
flour, cocoa powder, sugar, eggs, milk, butter
Recipe:

Generated Recipe:
1. Mix flour, cocoa, and sugar in a bowl.
2. Add eggs and milk; whisk until smooth.
3. Melt butter and combine with the mixture.
4. Pour into a baking pan and bake at 180°C for 25 minutes.
5. Cool and serve with chocolate icing.
