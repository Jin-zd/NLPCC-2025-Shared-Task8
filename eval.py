import os
import json
import numpy as np
from tqdm import tqdm
from openai import OpenAI
from time import sleep
from functools import lru_cache
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer

# Set your OpenAI API key here or as an environment variable.
client = OpenAI(
    api_key="",
)


def load_json(file_path):
    """Loads data from a JSON file.
    
    Args:
        file_path (str): Path to the JSON file to be loaded.
        
    Returns:
        dict/list: Parsed JSON data as Python objects.
    """
    with open(file_path, 'r') as f:
        return json.load(f)
    

def get_role_texts(data, role="seeker"):
    """Extracts role texts from dialogue data.

    Args:
        data (dict): Dialogue data containing whole conversations.
        role (str): The role to extract texts for ("seeker" or "supporter").

    Returns:
        list: List of role utterances extracted from every other turn in dialogues.
    """
    role_texts = []
    for chat_info in data.values():
        whole_dialogue = chat_info["whole_dialog"]
        start_index = 0 if role == "seeker" else 1
        for i in range(start_index, len(whole_dialogue), 2):
            role_texts.append(whole_dialogue[i].replace(f"{role}:", ""))   
    return role_texts



def distinct_n(texts, n):
    """Calculates Distinct-n metric for text diversity measurement.
    
    Args:
        texts (list): List of text strings to analyze.
        n (int): Size of n-grams to consider.
        
    Returns:
        float: Ratio of unique n-grams to total n-grams.
    """
    total_ngrams = 0
    unique_ngrams = set()

    for text in texts:
        tokens = text.lower().split()
        ngrams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
        total_ngrams += len(ngrams)
        unique_ngrams.update(ngrams)

    if total_ngrams == 0:
        return 0.0
    return len(unique_ngrams) / total_ngrams


def vector_extrema_similarity(x, y, glove_dict):
    """Calculates semantic similarity using GloVe vector extrema.
    
    Args:
        x (str): First text string for comparison.
        y (str): Second text string for comparison.
        glove (list): Pre-loaded GloVe word vectors.
        
    Returns:
        float: Cosine similarity score between vector extrema representations.
    """
    def word2vec(text, glove_dict):
        words = text.split()
        word_vectors = []
        for w in words:
            if w in glove_dict:
                word_vectors.append(glove_dict[w])
        return word_vectors

    x_vecs, y_vecs = word2vec(x, glove_dict), word2vec(y, glove_dict)
    if len(x_vecs) == 0 or len(y_vecs) == 0:
        return 0.0
    
    x_extrema, y_extrema = np.max(np.array(x_vecs), axis=0), np.max(np.array(y_vecs), axis=0)

    assert len(x_extrema) == len(y_extrema), "The dimensions of the two vectors do not match."
    
    zero_vector = np.zeros_like(x_extrema)
    if np.array_equal(x_extrema, zero_vector) or np.array_equal(y_extrema, zero_vector):
        return float(1) if np.array_equal(x_extrema, y_extrema) else float(0)
    
    dot_product = np.sum(x_extrema * y_extrema)
    norm_x = np.sqrt(np.sum(x_extrema * x_extrema))
    norm_y = np.sqrt(np.sum(y_extrema * y_extrema))
    
    cos = dot_product / (norm_x * norm_y)
    return cos


def chatgpt_query(prompt):
    """Synchronous query to OpenAI ChatGPT for evaluation score.

    Args:
        prompt (str): Prompt text for evaluation.
    
    Returns:
        str: Response from ChatGPT model.
    """
    response = client.chat.completions.create( 
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Only return the numerical rating, no explanation."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=5
    )
    return response.choices[0].message.content.strip()


@lru_cache(maxsize=1000)
def build_prompt(input_text, reference, generated):
    """Builds a prompt for evaluation of generated text.

    Args:
        input_text (str): Input text for the model.
        reference (str): Reference text for evaluation.
        generated (str): Generated text for evaluation.
    
    Returns:
        str: Formatted prompt for evaluation
    """
    return f"""
Evaluate the generated response (1-5 scale) considering:
1. Relevance to input
2. Match with reference
3. Fluency and coherence

Input: {input_text}
Reference: {reference}
Generated: {generated}

Score (1-5 only):
"""


def get_chatgpt_score(inputs, references, generateds):
    """Sequentially queries ChatGPT for evaluation scores with progress bar.

    Args:
        inputs (list): List of input texts.
        references (list): List of reference texts.
        generateds (list): List of generated texts.
    Returns:
        list: List of evaluation scores for each generated text
    """
    scores = []
    for inp, ref, gen in tqdm(zip(inputs, references, generateds), total=len(inputs), desc="Calculating GPT-4 scores"):
        prompt = build_prompt(inp, ref, gen)
        raw_score = chatgpt_query(prompt)
        sleep(0.5)
        scores.append(min(max(float(raw_score), 1), 5) / 5)
    
    return scores


def calculate_metrics(input_text, pred_text, ref_text, glove_dict):
    """Calculates multiple evaluation metrics for text generation.
    
    Args:
        pred_text (list): List of generated/predicted texts.
        ref_text (list): List of reference/target texts.
        glove (list): Pre-loaded GloVe word vectors.
        
    Returns:
        dict: Dictionary containing averaged scores for:
            - METEOR
            - BLEU
            - ROUGE-L
            - Vector Extrema
            - Distinct-2
            - Distinct-3
            - ChatGPT Score
    """
    print("Calculating metrics...")
    metrics = {
        'meteor': 0.0,
        'bleu': 0.0,
        'rouge': 0.0,
        'extrema': 0.0,
        'g_score': 0.0
    }

    print("Calculating GPT4 score...")
    g_scores = get_chatgpt_score(input_text, ref_text, pred_text)
    metrics['g_score'] = np.mean(g_scores)

    print("GPT4 score calculated. Calculating other metrics...")
    rouge_scorer_obj = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    smoothie = SmoothingFunction().method1
    for ref, pred in tqdm(zip(ref_text, pred_text), total=len(ref_text), desc="Calculating other metrics"):
        metrics["meteor"] += meteor_score([ref.split()], pred.split())
        metrics["bleu"] += sentence_bleu([ref.split()], pred.split(), weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie)
        metrics["rouge"] += rouge_scorer_obj.score(ref, pred)['rougeL'].fmeasure
        metrics["extrema"] += vector_extrema_similarity(pred, ref, glove_dict)
    metrics["distinct_2"] = distinct_n(pred_text, 2)
    metrics["distinct_3"] = distinct_n(pred_text, 3)
    print("Metrics calculated.")

    for key in ['meteor', 'bleu', 'rouge', 'extrema']:
        metrics[key] /= len(pred_text)
    
    return metrics


def calculate_total_score(metrics, weights=None):
    """Calculates weighted sum of evaluation metrics.
    
    Args:
        metrics (dict): Dictionary of metric scores.
        weights (dict, optional): Dictionary of metric weights. 
            Defaults to equal weights.
            
    Returns:
        float: Weighted average score of all metrics.
    """
    if weights is None:
        weights = {key: 1.0 for key in metrics}
    total_score = sum(metrics[key] * weights[key] for key in metrics)
    total_weight = sum(weights.values())
    return total_score / total_weight


def main():
    """Main execution function for validation pipeline.
    Loads data, computes metrics, and prints results.
    """

    # The file val_results.json is expected to follow the same structure as val.json.
    ref_path, pred_path = "val.json", "val_results.json"
    ref_data, pred_data = load_json(ref_path), load_json(pred_path)

    input_text, ref_text, pred_text = get_role_texts(ref_data, role="seeker"), get_role_texts(ref_data, role="supporter"), get_role_texts(pred_data, role="supporter")
    assert len(pred_text) == len(ref_text), "Number of predictions and references do not match."

    # The file glove.6B.50d.txt can be downloaded from https://nlp.stanford.edu/projects/glove/.
    glove_path = "glove.6B.50d.txt"
    glove_dict = {}
    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Loading GloVe"):
            parts = line.strip().split()
            word = parts[0]
            vector = [float(x) for x in parts[1:]]
            glove_dict[word] = vector

    metrics = calculate_metrics(input_text, pred_text, ref_text, glove_dict)

    weights = {
        'meteor': 0.2,
        'bleu': 0.2,
        'rouge': 0.2,
        'extrema': 0.1,
        'distinct_2': 0.1,
        'distinct_3': 0.1,
        'g_score': 0.1
    }
    total_score = calculate_total_score(metrics, weights)
    
    print("\nValidation Results:")
    for metric, value in metrics.items():
        print(f"{metric.upper()}: {value:.4f}")
    print(f"\nTotal Score: {total_score:.4f}")
    

if __name__ == '__main__':
    main()