## About score directory
1. Gpt_score subdirectory contains gpt score for table, html and latex
2. For other type of scores, they are all in the file table_scores.py, html_scores.py and latex_scores.py

## How to use xxx_scores.py
You can use the command:
```
python table_scores.py ref_file_path pred_file_path test_type
```
For example:
```
python table_scores.py "table_test.json" "table_test_output.txt" "ourtest"
```