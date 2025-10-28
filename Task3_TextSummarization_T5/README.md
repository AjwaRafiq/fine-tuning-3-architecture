Task 3: Encoder–Decoder Model (T5) — Text Summarization
=======================================================

Objective:
-----------
Fine-tune a pre-trained T5 model (t5-small or t5-base) on the CNN/DailyMail dataset
to generate abstractive summaries of long news articles.

Dataset:
--------
CNN/DailyMail News Summarization Dataset (Kaggle)
https://www.kaggle.com/datasets/gowrishankarp/newspaper-text-summarization-cnn-dailymail

--------------------------------------------------
Project Structure:
--------------------------------------------------
preprocess_t5.py      - Preprocessing script for text cleaning and formatting
finetune_t5.py        - Fine-tuning T5 model using Hugging Face Trainer
evaluate_t5.py        - Evaluation and summarization example script
data/train.csv        - Training data (input articles and summaries)
data/test.csv         - Testing data
data/processed_cnn_dm - Preprocessed dataset (arrow format)

--------------------------------------------------
Setup Instructions:
--------------------------------------------------
1. Install dependencies:

    pip install torch torchvision torchaudio
    pip install transformers datasets evaluate sentencepiece

(Optional for GPU optimization):
    pip install accelerate

--------------------------------------------------
Step 1: Data Preprocessing
--------------------------------------------------
Command:
    python preprocess_t5.py       --input_path data/train.csv       --input_col article       --target_col summary       --output_dir data/processed_cnn_dm       --max_input_length 512       --max_target_length 150

What it does:
- Cleans and truncates articles and summaries
- Saves processed dataset in Hugging Face arrow format

--------------------------------------------------
Step 2: Fine-tuning T5 Model
--------------------------------------------------
Command:
    python finetune_t5.py       --model_name t5-small       --dataset_path data/processed_cnn_dm       --output_dir ./t5-cnn-finetuned       --per_device_train_batch_size 4       --per_device_eval_batch_size 4       --num_train_epochs 3       --fp16

What it does:
- Loads tokenizer and model
- Tokenizes input and summary pairs
- Fine-tunes the model using Seq2SeqTrainer
- Evaluates using ROUGE metrics
- Saves fine-tuned model in output_dir

--------------------------------------------------
Step 3: Evaluation and Sample Summarization
--------------------------------------------------
Command:
    python evaluate_t5.py       --model_dir ./t5-cnn-finetuned       --test_path data/test.csv       --num_samples 5

Output Example:
================================================================================
SOURCE (truncated): The prime minister met with world leaders at the summit...
REFERENCE: Prime Minister discusses global policies with other leaders.
PREDICTION: PM met with leaders at global summit to discuss international policies.

Interactive Mode (no test file):
    python evaluate_t5.py --model_dir ./t5-cnn-finetuned

--------------------------------------------------
Evaluation Metrics (ROUGE)
--------------------------------------------------
Example results:
    {'rouge1': 41.23, 'rouge2': 18.45, 'rougeL': 38.72, 'gen_len': 25.6}

--------------------------------------------------
Tips for Better Performance:
--------------------------------------------------
- Use t5-base for better accuracy.
- Enable --fp16 for faster training on GPU.
- Increase --num_train_epochs for larger datasets.
- Use gradient accumulation for small GPU memory:
      --gradient_accumulation_steps 4
- Experiment with learning rates between 3e-5 and 1e-4.

--------------------------------------------------
Example Workflow Summary:
--------------------------------------------------
1. Preprocess ->  python preprocess_t5.py ...
2. Train ->       python finetune_t5.py ...
3. Evaluate ->    python evaluate_t5.py ...

--------------------------------------------------
Dependencies:
--------------------------------------------------
- Python 3.8 or higher
- PyTorch
- Transformers
- Datasets
- Evaluate
- Sentencepiece

Use this model directory for summarization tasks.


