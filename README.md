# Struc-Bench

<p align="center"><a href="https://aclanthology.org/2024.naacl-short.2/">[📄 Paper]</a>


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
@inproceedings{tang-etal-2024-struc,
    title = "Struc-Bench: Are Large Language Models Good at Generating Complex Structured Tabular Data?",
    author = "Tang, Xiangru  and
      Zong, Yiming  and
      Phang, Jason  and
      Zhao, Yilun  and
      Zhou, Wangchunshu  and
      Cohan, Arman  and
      Gerstein, Mark",
    editor = "Duh, Kevin  and
      Gomez, Helena  and
      Bethard, Steven",
    booktitle = "Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 2: Short Papers)",
    month = jun,
    year = "2024",
    address = "Mexico City, Mexico",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.naacl-short.2",
    pages = "12--34",
    abstract = "Despite the remarkable capabilities of Large Language Models (LLMs) like GPT-4, producing complex, structured tabular data remains challenging. Our study assesses LLMs{'} proficiency in structuring tables and introduces a novel fine-tuning method, cognizant of data structures, to bolster their performance. We unveil Struc-Bench, a comprehensive benchmark featuring prominent LLMs (GPT-NeoX-20B, GPT-3.5, GPT-4, and Vicuna), which spans text tables, HTML, and LaTeX formats. Our proposed FormatCoT aids in crafting format-specific instructions from the intended outputs to populate this benchmark. Addressing the gap in task-centered evaluation, we propose two innovative metrics, P-Score (Prompting Score) and H-Score (Heuristical Score), to more accurately gauge LLM performance. Our experiments show that applying our structure-aware fine-tuning to LLaMA-7B leads to substantial performance gains, outshining its LLM counterparts across most measures. In-depth error analysis and creating an ability map across six dimensions, coverage, formatting, reasoning, comprehension, pragmatics, and hallucination, highlight areas for future enhancements and suggest forthcoming research trajectories. Our code and models can be found at https://github.com/gersteinlab/Struc-Bench.",
}
```
