import regex
import Levenshtein, evaluate
import difflib
import pandas as pd
import os, re, sys, json
from bart_score import BARTScorer
os.environ["HF_HOME"] = ""

NEWLINE_REGEX = re.compile(r'\n|<NEWLINE>|<newline>')

#functions for our score

def levenshtein_similarity(pred, gold):
    if not gold:
        # If gold is empty, give zero score
        return 0.0
    return 1 - Levenshtein.distance(pred, gold) / (2 * len(gold))

def difflib_similarity(pred, gold):
    sm = difflib.SequenceMatcher(None, pred, gold)
    return sm.ratio()

def find_idx_after(text, start, query):
    assert start < len(text)
    find_idx = text[start:].find(query)
    if find_idx == -1:
        return -1
    return start + text[start:].find(query)

def get_table_artifacts(raw_table_text):
    split_list = NEWLINE_REGEX.split(raw_table_text)
    data_rows = []
    for row in split_list:
        data_row = [x.strip() for x in row.split("|")]
        data_rows.append(data_row)

    # Clean separator rows
    data_rows = [
        data_row
        for data_row in data_rows
        if not {c for x in data_row for c in x } <= {"-", "="}
    ]

    # Clean empty columns
    if not any(data_row[0] for data_row in data_rows):
        data_rows = [data_row[1:] for data_row in data_rows]
    if not any(data_row[-1] for data_row in data_rows):
        data_rows = [data_row[:-1] for data_row in data_rows]

    if len(data_rows) == 0:
        return {
            "num_rows": 0,
            "num_cols": 0,
            "column_names": [],
            "data_rows": [],
        }
    num_rows = len(data_rows)
    num_cols = max(len(data_row) for data_row in data_rows)
    column_names = data_rows[0]
    data_rows = data_rows[1:]
    return {
        "num_rows": num_rows,
        "num_cols": num_cols,
        "column_names": column_names,
        "data_rows": data_rows,
    }

def flatten_table(table):
    return [x for row in table for x in row ]

def get_team_and_player_tables(text):
    lower_text = text.lower()
    team_start_idx = lower_text.find("team")
    player_start_idx = find_idx_after(lower_text, start=team_start_idx, query="player")
    team_start_marker = find_idx_after(lower_text, start=team_start_idx, query="|")
    team_end_marker = text[:player_start_idx].rfind("|")
    player_start_marker = find_idx_after(lower_text, start=player_start_idx, query="|")
    player_end_marker = text.rfind("|")
    team_raw_table = text[team_start_marker:team_end_marker + 1]
    player_raw_table = text[player_start_marker:player_end_marker + 1]
    return team_raw_table, player_raw_table
    

def score_artifacts(artifacts_1, artifacts_2):
    sub_scores = {}
    sub_scores["num_rows_match"] = (artifacts_1["num_rows"] == artifacts_2["num_rows"]) * 1.0
    sub_scores["num_cols_match"] = (artifacts_1["num_cols"] == artifacts_2["num_cols"]) * 1.0
    sub_scores["columns_levenshtein_score"] = levenshtein_similarity(
        artifacts_1["column_names"],
        artifacts_2["column_names"],
    )
    sub_scores["columns_difflib_score"] = difflib_similarity(
        artifacts_1["column_names"],
        artifacts_2["column_names"],
    )
    sub_scores["data_levenshtein_score"] = levenshtein_similarity(
        flatten_table(artifacts_1["data_rows"]),
        flatten_table(artifacts_2["data_rows"]),
    )
    sub_scores["data_difflib_score"] = difflib_similarity(
        flatten_table(artifacts_1["data_rows"]),
        flatten_table(artifacts_2["data_rows"]),
    )
    return sub_scores

def score_raw_tables(pred_text, gold_text):
    team_raw_table_pred, player_raw_table_pred = get_team_and_player_tables(pred_text)
    team_raw_table_gold, player_raw_table_gold = get_team_and_player_tables(gold_text)

    team_table_artifacts_pred = get_table_artifacts(team_raw_table_pred)
    player_table_artifacts_pred = get_table_artifacts(player_raw_table_pred)
    team_table_artifacts_gold = get_table_artifacts(team_raw_table_gold)
    player_table_artifacts_gold = get_table_artifacts(player_raw_table_gold)

    team_artifacts = score_artifacts(team_table_artifacts_pred, team_table_artifacts_gold)
    player_artifacts = score_artifacts(player_table_artifacts_pred, player_table_artifacts_gold)

    return pd.DataFrame({
        "team": team_artifacts,
        "player": player_artifacts,
    }).mean(1).to_dict()

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
    our_score = score_raw_tables(pre[i], ref[i])
    content_scores = content_scores + our_score["data_levenshtein_score"] + our_score["data_difflib_score"]
    format_scores = format_scores + our_score["num_rows_match"] + our_score["num_cols_match"] + our_score["columns_levenshtein_score"] + our_score["columns_difflib_score"]
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
s = f"################table {test_type} gpt##################"
print(s)
print("Our score")
print("Content_score: ", content_scores)
print("Structure_score: ", format_scores)
print("---------------------")

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