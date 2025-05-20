from transformers import GenerationConfig
import re

from utils import get_prompt

def get_cheatsheet(q, log_ids, corpus_logs, corpus_logs_repr, tokenizer, model):
  # corpus_logs for getting question
  # corpus_logs_repr for getting reasoning (note: corpus_logs_repr already has '\n\n' appended to the end)

  logs_serialized = []
  for log_id in log_ids:
    log_question = corpus_logs[log_id]['question']
    log_reasoning = corpus_logs_repr[log_id]
    logs_serialized.append(f'User question: {log_question}\nModel response:\n{log_reasoning}')
  logs_serialized = ''.join(logs_serialized)

  # removed two sections that don't really apply to our datasets to avoid confusions
  # - Reusable Code Snippets and Solution Strategies
  # - Optimization Techniques & Edge Cases

  prompt = f'''# CHEATSHEET CURATOR
### Purpose and Goals
You are responsible for maintaining, refining, and optimizing the Dynamic Cheatsheet, which serves as a compact yet evolving repository of problem-solving strategies, reusable code snippets, and meta-reasoning techniques. Your goal is to enhance the model's long-term performance by continuously updating the cheatsheet with high-value insights while filtering out redundant or trivial information.
- The cheatsheet should include quick, accurate, reliable, and practical solutions to a range of technical and creative challenges.
- After seeing each input, you should improve the content of the cheatsheet, synthesizing lessons, insights, tricks, and errors learned from past problems and adapting to new challenges.
————
### Core Responsibilities
Selective Knowledge Retention:
- Preserve only high-value strategies, code blocks, insights, and reusable patterns that significantly contribute to problem-solving.
- Discard redundant, trivial, or highly problem-specific details that do not generalize well.
- Ensure that previously effective solutions remain accessible while incorporating new, superior methods.
Continuous Refinement & Optimization:
- Improve existing strategies by incorporating more efficient, elegant, or generalizable techniques.
- Remove duplicate entries or rephrase unclear explanations for better readability.
- Introduce new meta-strategies based on recent problem-solving experiences.
Structure & Organization:
- Maintain a well-organized cheatsheet with clearly defined sections:
- General Problem-Solving Heuristics
- Specialized Knowledge & Theorems
————
### Principles and Best Practices
For every new problem encountered:
1. Evaluate the Solution's Effectiveness
- Was the applied strategy optimal?
- Could the solution be improved, generalized, or made more efficient?
- Does the cheatsheet already contain a similar strategy, or should a new one be added?
2. Curate & Document the Most Valuable Insights
- Extract key algorithms, heuristics, and reusable code snippets that would help solve similar problems in the future.
- Identify patterns, edge cases, and problem-specific insights worth retaining.
- If a better approach than a previously recorded one is found, replace the old version.
3. Maintain Concise, Actionable Entries
- Keep explanations clear, actionable, concise, and to the point.
- Include only the most effective and widely applicable methods.
- Seek to extract useful and general solution strategies and/or Python code snippets.
4. Implement a Usage Counter
- Each entry must include a usage count: Increase the count every time a strategy is successfully used in problem-solving.
- Use the count to prioritize frequently used solutions over rarely applied ones.
————
### Memory Update Format
Use the following structure for each memory item:
```
<memory_item>
<description>
[Briefly describe the problem context, purpose, and key aspects of the solution.]
</description>
<example>
[Provide a well-documented code snippet, worked-out solution, or efficient strategy.]
</example>
</memory_item>
** Count: [Number of times this strategy has been used to solve a problem.]

<memory_item>
[...]
</memory_item>
** Count: [...]

[...]

<memory_item>
[...]
</memory_item>
```
- Prioritize accuracy, efficiency & generalizability: The cheatsheet should capture insights that apply across multiple problems rather than just storing isolated solutions.
- Ensure clarity & usability: Every update should make the cheatsheet more structured, actionable, and easy to navigate.
- Maintain a balance: While adding new strategies, ensure that old but effective techniques are not lost.
- Keep it evolving: The cheatsheet should be a living document that continuously improves over time, enhancing test-time meta-learning capabilities.
N.B. Keep in mind that once the cheatsheet is updated, any previous content not directly included will be lost and cannot be retrieved. Therefore, make sure to explicitly copy any (or all) relevant information from the previous cheatsheet to the new cheatsheet! Furthermore, make sure that all information related to the cheatsheet is wrapped inside the <cheatsheet> block.

## Cheatsheet Template
Use the following format for creating and updating the cheatsheet:

NEW CHEATSHEET:
```
<cheatsheet>
Version: [Version Number]

## General Problem-Solving Heuristics 

<memory_item>
[...]
</memory_item>
[...]

## Specialized Knowledge & Theorems
<memory_item>
[...]
</memory_item>
</cheatsheet>
```
N.B. Make sure that all information related to the cheatsheet is wrapped inside the <cheatsheet> block.
The cheatsheet can be as long as circa 350 words.
————
## NOTES FOR CHEATSHEET
{logs_serialized}
————
Make sure that the cheatsheet can aid the model tackle the next question.
## NEXT INPUT:
{q['question']}'''
  
  prompt = get_prompt(prompt)
  input_ids = tokenizer.encode(prompt, return_tensors='pt', add_special_tokens=False).to(model.device)

  generation_config = GenerationConfig(
    do_sample=False,
    temperature=1.0,
    repetition_penalty=1.0,
    num_beams=1,
    eos_token_id=tokenizer.eos_token_id,
    max_new_tokens=512
  )

  output = model.generate(input_ids=input_ids, generation_config=generation_config, eos_token_id=[tokenizer.eos_token_id], tokenizer=tokenizer)

  input_length = input_ids.size(-1)
  output = tokenizer.decode(output[0][input_length:], skip_special_tokens=False)

  full_prompt = tokenizer.batch_decode(input_ids)[0]
  # print(full_prompt)
  # print('\n\n')
  # print(output)

  cheatsheet = extract_cheatsheet(output) + '\n\n'

  cheatsheet_input_ids = tokenizer.encode(cheatsheet, return_tensors='pt', add_special_tokens=False)
  return cheatsheet_input_ids, full_prompt, output

def extract_cheatsheet(text):
  matches = re.findall(r"```(.*?)```", text, re.DOTALL)

  if len(matches) == 0:
    return text
  
  return matches[0].strip()