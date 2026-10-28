"""Benchmark task: generate one host-format passage per prompt.

Pushed with `kaggle benchmarks tasks push` and run server-side with `-m <model>`, which is the only
route to the full model catalogue (the local Model Proxy token is restricted to a curated subset that
does not include the GPT-5.6 trio).

The prompts are ours: fifty single-line requests spread over the six topics the competition uses
(philosophy, physics, mathematics, biology, computer science, creative prose). No competition text is
read, written or produced here.
"""
import pandas as pd
import kaggle_benchmarks as kbench

PROMPTS = [
    "Explain the difference between knowing a fact and understanding an idea, in a single paragraph.",
    "Is it possible to be mistaken about your own motives? Argue one side in a short paragraph.",
    "What does it mean for a life to go well? Answer in one paragraph without using a list.",
    "Describe the relationship between freedom and responsibility in a short passage.",
    "Can a society be unjust even if every law is followed? Answer briefly.",
    "What is the difference between pleasure and happiness? One paragraph.",
    "Argue that doubt is a kind of commitment, in a short paragraph.",
    "Explain what makes a promise binding, in a paragraph.",
    "Explain why the sky is blue without using the word scattering, in a short paragraph.",
    "What would change about everyday life if the speed of light were a thousand times slower?",
    "Describe what entropy means to someone who has never studied physics.",
    "Explain why a spinning top stays upright, in one paragraph.",
    "What is the difference between mass and weight, explained in a short paragraph?",
    "Describe how a refrigerator moves heat, in a paragraph.",
    "Explain why the night sky is dark, in a short passage.",
    "Explain why there are infinitely many prime numbers, in a short paragraph.",
    "What is a proof, and why does mathematics need them? One paragraph.",
    "Describe an intuitive reason the square root of two is not a fraction.",
    "Explain the difference between a conjecture and a theorem, briefly.",
    "What does it mean for a function to be continuous? Answer in a paragraph.",
    "Describe why zero was a difficult idea historically, in a short passage.",
    "Explain how a cell knows when to divide, in a short paragraph.",
    "Why do organisms age? Answer in one paragraph.",
    "Describe the difference between a virus and a bacterium for a general reader.",
    "Explain how vaccines train the immune system, in a paragraph.",
    "What is the role of chance in evolution? Answer briefly.",
    "Describe how plants decide when to flower, in a short paragraph.",
    "Explain what makes a sorting algorithm fast, in a short paragraph.",
    "Describe the difference between a process and a thread to a beginner.",
    "What is a hash collision and why does it matter? One paragraph.",
    "Explain why floating point arithmetic produces surprising results.",
    "Describe how a database keeps data consistent during a crash, in a paragraph.",
    "What does it mean for a program to be correct? Answer briefly.",
    "Explain the idea behind public key cryptography in a paragraph.",
    "Write a short paragraph about a lighthouse keeper who has stopped counting the days.",
    "Describe a kitchen at four in the morning in one paragraph.",
    "Write a paragraph about a train that arrives at a station nobody remembers building.",
    "Describe the sound of a city waking up, in one paragraph.",
    "Write a short paragraph about a library that rearranges itself at night.",
    "Describe a person learning to swim as an adult, in one paragraph.",
    "Write a paragraph about the last shop on a street that is being demolished.",
    "Describe an argument between two people who agree with each other, in one paragraph.",
    "Write a short paragraph about a map that is more accurate than the land.",
    "Describe a garden that grows only at night, in one paragraph.",
    "Explain why some problems are harder than others, in a short paragraph.",
    "Describe what a proof by contradiction is, briefly.",
    "Explain the greenhouse effect without naming any gas, in a paragraph.",
    "What is the difference between weather and climate? One paragraph.",
    "Describe how memory works at a high level, in a short paragraph.",
    "Explain why some medicines must be taken at the same time every day, in a paragraph.",
]


@kbench.task(name="ntl passage generation",
             description="One single-paragraph answer per prompt, used to build extra training passages.")
def generate(llm, prompt_text: str) -> str:
    """Answer the prompt in one self-contained paragraph, with no markdown and no preamble."""
    return llm.prompt(prompt_text)


if __name__ == "__main__":
    generate.evaluate(evaluation_data=pd.DataFrame({"prompt_text": PROMPTS}),
                      n_jobs=4, on_failure="continue")
