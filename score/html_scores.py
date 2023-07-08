import Levenshtein
import bs4
import difflib
import pandas as pd
import os, re, sys, json, evaluate
from bart_score import BARTScorer
os.environ["HF_HOME"] = ""

# pip install levenshtein difflib beautifulsoup4

RECOGNIZED_HTML_TAGS = [
    "table", "tr", "th", "td",
    "ul", "ol", "li",
    "div", "span", "p",
    "a", "img", "embed", "pre",
    "h1", "h2", "h3", "h4", "h5", "h6",
    "input", "button",
]

def levenshtein_similarity(pred, gold):
    if not gold:
        # If gold is empty, give zero score
        return 0.0
    return 1 - Levenshtein.distance(pred, gold) / (2 * len(gold))

def difflib_similarity(pred, gold):
    sm = difflib.SequenceMatcher(None, pred, gold)
    return sm.ratio()

def _build_simple_tree_recurse(curr, record):
    if not curr.descendants:
        return
    for desc in curr.descendants:
        if desc.name in RECOGNIZED_HTML_TAGS:
            record.append(f"<{desc.name}>")
            _build_simple_tree_recurse(desc, record)
            record.append(f"</{desc.name}>")
        
def build_simple_tree(soup):
    record = []
    _build_simple_tree_recurse(soup, record)
    return record

def score_html(pred_html, gold_html):
    soup_pred = bs4.BeautifulSoup(pred_html)
    soup_gold = bs4.BeautifulSoup(gold_html)

    # Heuristic to see what kind of structure we're looking for
    if soup_gold.find_all("table"):
        mode = "table"
        cell_types = ["th", "td"]
    elif soup_gold.find_all("ul"):
        mode = "ul"
        cell_types = ["li"]
    elif soup_gold.find_all("ol"):
        mode = "ol"
        cell_types = ["li"]
    elif soup_gold.find_all("li"):
        mode = "li"
        cell_types = ["li"]
    else:
        mode = "unknown"
        cell_types = []

    sub_scores = {}

    # Content score
    if cell_types:
        text_ls_gold = [x.text for x in soup_gold.find_all(cell_types)]
        text_ls_pred = [x.text for x in soup_pred.find_all(cell_types)]
    else:
        # If it's not a table/ul/ol/li, we are much more lenient about what we consider a cell
        text_ls_gold = [x.text for x in soup_gold.find_all(RECOGNIZED_HTML_TAGS) if x.text]
        text_ls_pred = [x.text for x in soup_pred.find_all(RECOGNIZED_HTML_TAGS) if x.text]

    # [num_cells_match]: 0/1 exact match for whether the number of content cells is identical
    sub_scores["num_cells_match"] = (len(text_ls_pred) == len(text_ls_gold)) * 1.0
    # [cells_text_levenshtein_score]: Levenshtein score on cells
    sub_scores["cells_text_levenshtein_score"] = levenshtein_similarity(text_ls_pred, text_ls_gold)
    # [cells_text_difflib_score]: difflib score on cells
    sub_scores["cells_text_difflib_score"] = difflib_similarity(text_ls_pred, text_ls_gold)

    # Structure score (ignores text content)
    # We build a simple tree structure based on the recognized tags, and flatten
    flat_gold = build_simple_tree(soup_gold)
    flat_pred = build_simple_tree(soup_pred)
    # [struct_levenshtein_score]: Levenshtein score on flattened structured
    sub_scores["struct_levenshtein_score"] = levenshtein_similarity(flat_pred, flat_gold)
    # [struct_difflib_score]: Difflib score on flattened structured
    sub_scores["struct_difflib_score"] = difflib_similarity(flat_pred, flat_gold)
    return sub_scores

ref_file_path = sys.argv[1]
pre_file_path = sys.argv[2]


if ref_file_path.find(".txt") != -1 or ref_file_path.find(".data") != -1:

    with open(ref_file_path, 'r') as ref_file, open(pre_file_path, "r") as pre_file:
        reference_lines = ref_file.readlines()
        prediction_lines = pre_file.readlines()
    #print(reference_lines[0:])
    #print(prediction_lines[0:])

elif ref_file_path.find(".json") != -1:

    with open(ref_file_path, 'r') as ref_file, open(pre_file_path, "r") as pre_file:
        ref_lines = json.load(ref_file)
       
        reference_lines = []
        prediction_lines = pre_file.readlines()
        for i in ref_lines:
            reference_lines.append(i["output"])

else:
    print("input file error!")
    sys.exit()

pre = prediction_lines[0:]
ref = reference_lines[0:]

#our score
content_scores = 0.0
format_scores = 0.0
for i in range(len(pre)):
    our_score = score_html(pre[i], ref[i])
    content_scores = content_scores + our_score["cells_text_difflib_score"] + our_score["cells_text_levenshtein_score"]
    format_scores = format_scores + our_score["num_cells_match"] + our_score["struct_levenshtein_score"] + our_score["struct_difflib_score"]
content_scores = content_scores / len(pre)
format_scores = format_scores / len(pre)

#bertscore
bertscore = evaluate.load("bertscore")
bertscore_results = bertscore.compute(predictions=pre, references=ref, lang="en")

#rouge
rouge = evaluate.load('rouge')
rouge_results = rouge.compute(predictions=pre, references=ref)

#bleurt
bleurt = evaluate.load("bleurt", module_type="metric")
bleurt_results = bleurt.compute(predictions=pre, references=ref)

#bart_scorer
bart_scorer = BARTScorer(device='cuda:0', checkpoint='facebook/bart-large-cnn')
bart_score = bart_scorer.score(['This is interesting.'], ['This is fun.'], batch_size=4)
'''
#sacrebleu
sacrebleu = evaluate.load("sacrebleu")
sacrebleu_results = sacrebleu.compute(predictions=pre, references=ref)
'''
test_type = sys.argv[3]
s = f"################html {test_type} gpt##################"
print(s)
print("Our score")
print("Content_score: ", content_scores)
print("Structure_score: ", format_scores)
print("--------------------")

bert_score = sum(bertscore_results["f1"]) / len(pre)
print(f"bertscore: {bert_score}")
print("---------------------")
rouge_score = rouge_results["rougeL"]
print(f"rouge score: {rouge_score}")
print("---------------------")
bleurt_score = sum(bleurt_results["scores"]) / len(pre)
print(f"bleurt score: {bleurt_score}")
print("---------------------")
BART_score = sum(bart_score) / len(pre)
print(f"BART score: {BART_score}")
'''
print("---------------------")
sacrebleu_score = round(sacrebleu_results["score"], 1)
print(f"sacrebleu score: {sacrebleu_score}")
'''