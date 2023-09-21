# Struc-Bench

<p align="center"><a href="https://arxiv.org/abs/2309.08963">[📄 Paper]</a>


Welcome to Struc-Bench! We employ the same finetune.py and generate.py as used in alpaca-lora. Moreover, we have divided generate.py into three distinct categories: table, html, and latex. The scoring results can be found in the score directory, with scores derived from GPT housed in the gpt_score subdirectory. The remaining evaluations are segregated into table_scores.py, html_scores.py, and latex_scores.py, respectively.


## How to use generate.py
You can use the command:
```
python generate.py read_json_path output_file_path
```
For example:
```
python generate.py "table_test.json" "table_test_output.txt"
```

## Our data
You can access our data in the google drive: [data](https://drive.google.com/drive/folders/1XjlwdqdQxPQzTh0vqPsdUmpY5z5aD-v6?usp=drive_link)

## Citation
If you find our work useful in your research, please kindly consider cite:
```
@article{tang2023struc,
  title={Struc-Bench: Are Large Language Models Really Good at Generating Complex Structured Data?},
  author={Xiangru Tang and Yiming Zong and Jason Phang and Yilun Zhao and Wangchunshu Zhou and Arman Cohan and Mark B. Gerstein},
  journal={arXiv preprint arXiv:2309.08963},
  year={2023}
}
```
