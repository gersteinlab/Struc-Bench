import openai
import json
import pandas as pd
import asyncio
import logging
import os, re, sys
from typing import Any
import tiktoken

import aiolimiter
import openai
import openai.error
from aiohttp import ClientSession
from tqdm.asyncio import tqdm_asyncio
import random

LATEX_EVAL_TEMPLATE = """We want to evaluate how similar the following LaTeX tables are.

Table 1:

```latex
{input1}
```

Table 2:

```latex
{input2}
```

Based on the above, we wanted to determine if the above tables are similar. Ideally they should have identical content and structure. Score the "content similarity" and "structural similarity" between 0 and 10. 

- Content similarity: 10 if the contents of the table cells are identical, 0 if they are entirely different. If about 50% of the cells have the same data, the score should be 5.
- Structural similarity: 10 if the tables have the same structure (e.g. same column and rows with identical ordering, same alignment, etc) although text formatting differences can be ignored (e.g. colors, font).

Output a JSON object such as the following:

```json
{{
  "content_similarity": ...
  "structural_similarity": ...
}}
```

Think carefully, and then output the scores."""


HTML_EVAL_TEMPLATE = """We want to evaluate how similar the following HTML tables/data structures are.

Table 1:

```html
{input1}
```

Table 2:

```html
{input2}
```

Based on the above, we wanted to determine if the above tables are similar. Ideally they should have identical content and structure. Score the "content similarity" and "structural similarity" between 0 and 10. 

- Content similarity: 10 if the contents of the table cells are identical, 0 if they are entirely different. If about 50% of the cells have the same data, the score should be 5.
- Structural similarity: 10 if the tables have the same structure (e.g. same column and rows with identical ordering, same alignment, etc) although text formatting differences can be ignored (e.g. colors, font).

Output a JSON object such as the following:

```json
{{
  "content_similarity": ...
  "structural_similarity": ...
}}
```

Think carefully, and then output the scores."""

RAW_EVAL_TEMPLATE = """We want to evaluate how similar the following tables/data structures are.

Table 1:

```
{input1}
```

Table 2:

```
{input2}
```

Based on the above, we wanted to determine if the above tables are similar. Ideally they should have identical content and structure. Score the "content similarity" and "structural similarity" between 0 and 10. 

- Content similarity: 10 if the contents of the table cells are identical, 0 if they are entirely different. If about 50% of the cells have the same data, the score should be 5.
- Structural similarity: 10 if the tables have the same structure (e.g. same column and rows with identical ordering, same alignment, etc) although text formatting differences can be ignored (e.g. colors, font).

Output a JSON object such as the following:

```json
{{
  "content_similarity": ...
  "structural_similarity": ...
}}
```

Think carefully, and then output the scores."""

"""Tools to generate from OpenAI prompts."""

def perturbation_prompt(question, instruction):
    message = [
        {"role": "system", "content": instruction},

        {"role": "user", "content": question},
    ]
    return message

def length_is_valid(input_table_str, max_tokens):
    enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
    input_token_len = len(enc.encode(input_table_str))
    return input_token_len <  4096 - max_tokens

async def _throttled_openai_chat_completion_acreate(
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
    top_p: float,
    limiter: aiolimiter.AsyncLimiter,
) -> dict[str, Any]:
    async with limiter:
        for _ in range(10):
            try:
                return await openai.ChatCompletion.acreate(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                )
            except openai.error.RateLimitError:
                logging.warning(
                    "OpenAI API rate limit exceeded. Sleeping for 20 seconds."
                )
                sleep_time = random.randint(10, 20)
                await asyncio.sleep(sleep_time)
            except asyncio.exceptions.TimeoutError or openai.error.Timeout or asyncio.TimeoutError:
                logging.warning("OpenAI API timeout. Sleeping for 10 seconds.")
                await asyncio.sleep(10)
            except openai.error.APIError as e:
                logging.warning(f"OpenAI API error: {e}")
                await asyncio.sleep(10)
            except:
                logging.warning("Unknown OpenAI API error. Sleeping for 10 seconds.")
                await asyncio.sleep(10)
        return {"choices": [{"message": {"content": ""}}]}


async def generate_from_openai_chat_completion(
    messages,
    engine_name: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    requests_per_minute: int = 50,
    api_key: str = "",
) -> list[str]:
    """Generate from OpenAI Chat Completion API.

    Args:
        full_contexts: List of full contexts to generate from.
        prompt_template: Prompt template to use.
        model_config: Model configuration.
        temperature: Temperature to use.
        max_tokens: Maximum number of tokens to generate.
        top_p: Top p to use.
        requests_per_minute: Number of requests per minute to allow.

    Returns:
        List of generated responses.
    """
    api_key = ""
    openai.api_key = api_key
    session = ClientSession()
    openai.aiosession.set(session)
    limiter = aiolimiter.AsyncLimiter(requests_per_minute)
    async_responses = [
        _throttled_openai_chat_completion_acreate(
            model=engine_name,
            messages=message,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            limiter=limiter,
        )
        for message in messages
    ]
    responses = await tqdm_asyncio.gather(*async_responses)
    await session.close()
    return [x["choices"][0]["message"]["content"] for x in responses]

def parse_json(text):
    if "```json" not in text:
        return None
    text = text.split("```json")[1]
    if "```" not in text:
        return None
    text = text.split("```")[0]
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    if "content_similarity" not in parsed:
        return None
    if "structural_similarity" not in parsed:
        return None
    return parsed
    

def _score_single(filter_lines, retries=5):
    parsed = {"content_similarity": None, "structural_similarity": None}
    messages = [[{"role": "system", "content": "You are a helpful data-processing assistant."},{"role": "user", "content": f": {filter_lines[i]}"}] for i in range(len(filter_lines))]
    responses = asyncio.run(generate_from_openai_chat_completion(
            messages=messages,
            engine_name="gpt-3.5-turbo",
            temperature=0.1,
            max_tokens=1333,
            top_p=1.0,
        ))
    new_response = []
    for i in responses:
        parsed = parse_json(i)
        #print(type(parsed))
        #print(parsed)
        new_response.append(parsed)
    return new_response
'''
    for i in range(retries):
        try:
            out = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-0613",
                messages=[
                    {"role": "system", "content": "You are a helpful data-processing assistant."},
                    {"role": "user", "content": score_text},
                ],
            )
            raw_out = out.choices[0].message.content
            parsed = parse_json(raw_out)
            if parsed:
                break
        except openai.error as e:
            print(f"An error occurred during the API call: {e}. Retrying {i+1}/{retries}")
        except Exception as e:
            print(f"An error occurred during the API call: {e}. Retrying {i+1}/{retries}")


def _score_double(input1, input2, template):
    score1 = _score_single(template.format(
        input1=input1,
        input2=input2,
    ))
    score2 = _score_single(template.format(
        input1=input2,
        input2=input1,
    ))
    return (pd.DataFrame([score1, score2]).mean() / 10).to_dict()


def score_latex(table1, table2):
    return _score_double(input1=table1, input2=table2, template=LATEX_EVAL_TEMPLATE)

def score_html(table1, table2):
    return _score_double(input1=table1, input2=table2, template=HTML_EVAL_TEMPLATE)

def score_raw(table1, table2):
    return _score_double(input1=table1, input2=table2, template=RAW_EVAL_TEMPLATE)
'''

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

filter_entry = []
for i in range(len(pre)):
    filter_entry.append(HTML_EVAL_TEMPLATE.format(input1=pre[i], input2=ref[i]))
    filter_entry.append(HTML_EVAL_TEMPLATE.format(input1=ref[i], input2=pre[i]))

results = _score_single(filter_entry)

content_score = 0.0
structure_score = 0.0
count = 0

for i in range(len(results)):
    #print(type(results[i]))
    #print(results[i])
    print(i)
    if i % 2 == 0:
        if results[i] is not None and results[i+1] is not None and results[i]["content_similarity"] is not None and results[i]["content_similarity"] is not None and results[i+1]["structural_similarity"] is not None and results[i+1]["structural_similarity"] is not None:
            if isinstance(results[i]["content_similarity"], (float, int)) and isinstance(results[i]["structural_similarity"], (float, int)) and (isinstance(results[i+1]["content_similarity"], (float, int)) or isinstance(results[i+1]["structural_similarity"], (float, int))):
                count = count + 2
                
                content_score = content_score + results[i]["content_similarity"] + results[i+1]["content_similarity"]
                structure_score = structure_score + results[i]["structural_similarity"] + results[i+1]["structural_similarity"]
        
    '''
    score = _score_double(pre[i], ref[i], HTML_EVAL_TEMPLATE)
    if "content_similarity" in score and "content_similarity" in score:
        content_score = content_score + score["content_similarity"]
        structure_score = structure_score + score["structural_similarity"]
        count = count + 1
    print(i)
'''
content_score = content_score / count
structure_score = structure_score / count
print(len(ref))
print(len(results))
test_type = sys.argv[3]
s = f"################html {test_type} gpt##################"
print(s)
print("Content_score: ", content_score)
print("Structure_score: ", structure_score)
    
