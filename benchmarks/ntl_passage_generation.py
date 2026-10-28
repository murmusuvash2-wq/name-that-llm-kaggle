"""Benchmark task: generate host-format passages, one per prompt, for the GPT-5.6 trio.

Convention (copied from a working task): the first parameter is `llm`, and the file ends with
`task.run(kbench.llm)` so Kaggle can swap in whatever model `tasks run -m` asks for. No
`if __name__` guard.

Ten prompts for the pilot: the goal is to measure the real spend per passage and to see whether the
output looks like a host passage. The full fifty come after that number is known.
"""
import json

import kaggle_benchmarks as kbench

PROMPTS = [
    "Explain the difference between knowing a fact and understanding an idea, in a single paragraph.",
    "Is it possible to be mistaken about your own motives? Argue one side in a short paragraph.",
    "Explain why the sky is blue without using the word scattering, in a short paragraph.",
    "Describe what entropy means to someone who has never studied physics.",
    "Explain why there are infinitely many prime numbers, in a short paragraph.",
    "Explain how a cell knows when to divide, in a short paragraph.",
    "Explain what makes a sorting algorithm fast, in a short paragraph.",
    "Write a short paragraph about a lighthouse keeper who has stopped counting the days.",
    "Describe a kitchen at four in the morning in one paragraph.",
    "Write a paragraph about a train that arrives at a station nobody remembers building.",
]


@kbench.task(name="NTL passage generation",
             description="One self-contained 40-250 word paragraph per prompt, no markdown, no preamble.")
def generate(llm) -> None:
    for i, prompt in enumerate(PROMPTS):
        text = str(llm.prompt(prompt)).strip()
        kbench.assertions.assert_true(len(text) > 0, expectation=f"prompt {i} returned text")
        # both routes out: the run's output files, and the log, so nothing is lost if only one works
        with open("passages.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({"prompt_id": i, "prompt": prompt, "text": text}) + "\n")
        print(f"PASSAGE {i} >>> {text}", flush=True)


generate.run(kbench.llm)
