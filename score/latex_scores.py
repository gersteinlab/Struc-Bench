import Levenshtein
import bs4
import difflib
from pylatexenc.latexwalker import (
    LatexWalker,
    LatexCharsNode,
    LatexCommentNode,
    LatexEnvironmentNode,
    LatexGroupNode,
    LatexMacroNode,
    LatexMathNode,
    LatexNode,
    LatexSpecialsNode,
)
import pylatexenc.latexwalker
import os, re, sys, json, evaluate
from bart_score import BARTScorer
os.environ["HF_HOME"] = "/gpfs/slayman/pi/gerstein/xt86/emin/cache"

# pip install levenshtein difflib pylatexenc

def levenshtein_similarity(pred, gold):
    if not gold:
        # If gold is empty, give zero score
        return 0.0
    return 1 - Levenshtein.distance(pred, gold) / (2 * len(gold))

def difflib_similarity(pred, gold):
    sm = difflib.SequenceMatcher(None, pred, gold)
    return sm.ratio()


class NodeProcessor:
    def pre_process(self, node):
        pass
        
    def post_process(self, node):
        pass
        
    def is_done(self) -> bool:
        False
        
    def skip_node(self, node) -> bool:
        False
        
    def get_result(self):
        return
    
    def walk(self, node_list):
        walk_nodes(node_list, self)
        return self.get_result()

class FindProcessor(NodeProcessor):
    def __init__(self, criteria_func, early_stop=False):
        super().__init__()
        self.criteria_func = criteria_func
        self.early_stop = early_stop
        self.found = []
    def pre_process(self, node):
        if self.criteria_func(node):
            self.found.append(node)
    def is_done(self):
        if self.early_stop and self.found:
            return True
        return False
    
    def get_result(self):
        if self.found:
            return self.found[0]
        else:
            return None

class StringProcessor(NodeProcessor):
    def __init__(self):
        super().__init__()
        self.string = ""
    def pre_process(self, node):
        if isinstance(node, LatexCharsNode):
            self.string += node.chars
        elif isinstance(node, LatexGroupNode):
            self.string += node.delimiters[0]
        elif isinstance(node, LatexSpecialsNode):
            self.string += node.specials_chars
    def post_process(self, node):
        if isinstance(node, LatexGroupNode):
            self.string += node.delimiters[1]
    def get_result(self):
        return self.string
            
class TabularAlignmentProcessor(NodeProcessor):
    def __init__(self):
        super().__init__()
        self.string = ""
    def pre_process(self, node):
        if isinstance(node, LatexCharsNode):
            self.string += node.chars
    def skip_node(self, node):
        # Note: this will ignore any sub-argumement
        # e.g. it will treat p{2cm} as just p
        return isinstance(node, LatexGroupNode)
    def get_result(self):
        return self.string
    
def _walk_node_recurse(node, processor):
    if processor.skip_node(node):
        return
    processor.pre_process(node)
    all_node_lists = []
    if hasattr(node, "nodeargd"):
        if node.nodeargd:
            all_node_lists += node.nodeargd.argnlist
    if hasattr(node, "nodelist"):
        all_node_lists += node.nodelist
        
    for desc in all_node_lists:
        out = _walk_node_recurse(desc, processor)
    processor.post_process(node)
    

def walk_nodes(node_list, processor):
    if not isinstance(node_list, list):
        node_list = [node_list]
    for node in node_list:
        _walk_node_recurse(node, processor)
        if processor.is_done():
            return
        
def find_tabular_node(base_nodelist):
    return FindProcessor(
        criteria_func=lambda _: isinstance(_, LatexEnvironmentNode) and _.envname == "tabular",
        early_stop=True,
    ).walk(base_nodelist)

def find_tabular_alignment_string(tabular_node):
    return TabularAlignmentProcessor().walk(
        tabular_node.nodeargd.argnlist[0]
    )

class TabularDataProcessor(NodeProcessor):
    def __init__(self, skip_macros=False):
        super().__init__()
        self.rows = []
        self.curr_row = []
        self.curr_string = ""
        self.skip_macros = skip_macros
    def pre_process(self, node):
        if isinstance(node, LatexCharsNode):
            self.curr_string += node.chars
        if isinstance(node, LatexSpecialsNode):
            if node.specials_chars == "&":
                self.curr_row.append(self.curr_string)
                self.curr_string = ""
            else:
                self.curr_string += node.specials_chars
        if isinstance(node, LatexMacroNode):
            if node.macroname == "\\":
                if self.curr_string:
                    self.curr_row.append(self.curr_string)
                self.curr_string = ""
                self.rows.append(self.curr_row)
                self.curr_row = []
            elif not self.skip_macros:
                # Will ignore macro arguments
                self.curr_string += f"[{node.macroname}]"
    def skip_node(self, node):
        # Note: this will ignore any sub-argumement
        # e.g. it will treat p{2cm} as just p
        return isinstance(node, LatexGroupNode)
    
    def get_result(self):
        if self.curr_string:
            self.curr_row.append(self.curr_string)
        if self.curr_row:
            self.rows.append(self.curr_row)
        return self.rows
    
class CaptionFindProcessor(NodeProcessor):
    def __init__(self):
        super().__init__()
        self.take_next_group = False
        self.found = None
    def pre_process(self, node):
        if self.take_next_group and isinstance(node, LatexGroupNode) and not self.found:
            self.found = node
    def post_process(self, node):
        if isinstance(node, LatexMacroNode) and node.macroname == "caption":
            self.take_next_group = True
    def is_done(self):
        if self.found:
            return True
        return False
    
    def get_result(self):
        if self.found:
            return StringProcessor().walk(self.found.nodelist)
        else:
            return "None"
    
def strip_table(table):
    table = [[x.strip() for x in row if x.strip()] for row in table]
    table = [row for row in table if row]
    return table

def flatten_table(table):
    return [x for row in table for x in row ]

def get_latex_artifacts(walker):
    nodelist, pos, len_ = walker.get_latex_nodes()
    tabular_node = find_tabular_node(nodelist)
    if not tabular_node:
        return {
            "tabular_alignment_string": "",
            "data_table": [],
            "caption": "None",
            "num_rows": 0,
            "num_cols": 0,
        }
    tabular_alignment_string = find_tabular_alignment_string(tabular_node)
    data_table = TabularDataProcessor(skip_macros=True).walk(tabular_node.nodelist)
    data_table = strip_table(data_table)
    caption = CaptionFindProcessor().walk(nodelist)
    num_rows, num_cols = get_dims(data_table)
    return {
        "tabular_alignment_string": tabular_alignment_string,
        "data_table": data_table,
        "caption": caption,
        "num_rows": num_rows,
        "num_cols": num_cols,        
    }

def get_dims(table):
    if len(table) > 0:
        return len(table), max(len(row) for row in table)
    else:
        return 0, 0

def score_latex(pred_latex, gold_latex):
    walker_gold = LatexWalker(gold_latex)
    walker_pred = LatexWalker(pred_latex)
    artifacts_pred = get_latex_artifacts(walker_pred)
    artifacts_gold = get_latex_artifacts(walker_gold)
    subscores = {}
    subscores["alignment_levenshtein_score"] = levenshtein_similarity(
        artifacts_pred["tabular_alignment_string"],
        artifacts_gold["tabular_alignment_string"],
    )
    subscores["alignment_difflib_score"] = difflib_similarity(
        artifacts_pred["tabular_alignment_string"],
        artifacts_gold["tabular_alignment_string"],
    )
    subscores["data_levenshtein_score"] = levenshtein_similarity(
        flatten_table(artifacts_pred["data_table"]),
        flatten_table(artifacts_gold["data_table"]),
    )
    subscores["data_difflib_score"] = difflib_similarity(
        flatten_table(artifacts_pred["data_table"]),
        flatten_table(artifacts_gold["data_table"]),
    )
    subscores["caption_match_score"] = (artifacts_pred["caption"] == artifacts_gold["caption"]) * 1.0
    subscores["num_rows_match"] = (artifacts_pred["num_rows"] == artifacts_gold["num_rows"]) * 1.0
    subscores["num_cols_match"] = (artifacts_pred["num_cols"] == artifacts_gold["num_cols"]) * 1.0
    return subscores

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
    our_score = score_latex(pre[i], ref[i])
    content_scores = content_scores + our_score["caption_match_score"] + our_score["data_levenshtein_score"] + our_score["data_difflib_score"]
    format_scores = format_scores + our_score["alignment_levenshtein_score"] + our_score["alignment_difflib_score"] + our_score["num_rows_match"] + our_score["num_cols_match"]
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
s = f"################latex {test_type} gpt##################"
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