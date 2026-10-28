"""Benchmark task: generate host-format passages across all reachable models.

One prompt in, one self-contained 40-250 word paragraph out, in the same six topics the competition
uses. Run it server-side with `kaggle benchmarks tasks run -m <slug>` so the full model catalogue is
available (the local proxy token only reaches a curated subset).

Every model answering the same prompt is what makes this useful: it reproduces the host's structure,
where several models answer one question, and that pairing is the only signal that separates
siblings. No competition text is read, written or produced.
"""
import kaggle_benchmarks as kbench

PROMPTS = [
    'Explain knowing versus understanding in one paragraph of about 150 words. Single paragraph, no preamble, no bullet points, no markdown.',
    'Explain the difference between pleasure and happiness in one paragraph of about 150 words. Single paragraph, no preamble, no bullet points, no markdown.',
    'Explain whether we can be mistaken about our own motives in one paragraph of about 150 words. Single paragraph, no preamble, no bullet points, no markdown.',
    'Explain what makes a promise binding in one paragraph of about 150 words. Single paragraph, no preamble, no bullet points, no markdown.',
    'Explain whether a society can be unjust while obeying its own laws in one paragraph of about 150 words. Single paragraph, no preamble, no bullet points, no markdown.',
    'Explain the difference between freedom and responsibility in one paragraph of about 150 words. Single paragraph, no preamble, no bullet points, no markdown.',
    'Explain whether doubt is a kind of commitment in one paragraph of about 150 words. Single paragraph, no preamble, no bullet points, no markdown.',
    'Explain what it means for a life to go well in one paragraph of about 150 words. Single paragraph, no preamble, no bullet points, no markdown.',
    'Explain the difference between a reason and a cause in one paragraph of about 150 words. Single paragraph, no preamble, no bullet points, no markdown.',
    'Explain whether forgiveness can be owed to someone who never apologises in one paragraph of about 150 words. Single paragraph, no preamble, no bullet points, no markdown.',
    'Explain the difference between solitude and loneliness in one paragraph of about 150 words. Single paragraph, no preamble, no bullet points, no markdown.',
    'Explain whether moral luck undermines praise and blame in one paragraph of about 150 words. Single paragraph, no preamble, no bullet points, no markdown.',
    'Explain the difference between believing something and knowing it in one paragraph of about 150 words. Single paragraph, no preamble, no bullet points, no markdown.',
    'Explain whether it is rational to fear death in one paragraph of about 150 words. Single paragraph, no preamble, no bullet points, no markdown.',
    'Explain the difference between a rule and a principle in one paragraph of about 150 words. Single paragraph, no preamble, no bullet points, no markdown.',
    'Explain whether character or circumstance explains most of what people do in one paragraph of about 150 words. Single paragraph, no preamble, no bullet points, no markdown.',
    'Explain the difference between tolerance and indifference in one paragraph of about 150 words. Single paragraph, no preamble, no bullet points, no markdown.',
    'Explain whether there can be a private language in one paragraph of about 150 words. Single paragraph, no preamble, no bullet points, no markdown.',
    'Explain what makes an action voluntary in one paragraph of about 150 words. Single paragraph, no preamble, no bullet points, no markdown.',
    'Explain the difference between guilt and shame in one paragraph of about 150 words. Single paragraph, no preamble, no bullet points, no markdown.',
    'Explain whether attention is a moral matter in one paragraph of about 150 words. Single paragraph, no preamble, no bullet points, no markdown.',
    'Explain the difference between truth and sincerity in one paragraph of about 150 words. Single paragraph, no preamble, no bullet points, no markdown.',
    'Explain whether a person can choose their own values in one paragraph of about 150 words. Single paragraph, no preamble, no bullet points, no markdown.',
    'Explain what it means to understand another person in one paragraph of about 150 words. Single paragraph, no preamble, no bullet points, no markdown.',
    'Explain the difference between a habit and a virtue in one paragraph of about 150 words. Single paragraph, no preamble, no bullet points, no markdown.',
    'Explain whether boredom is a modern problem in one paragraph of about 150 words. Single paragraph, no preamble, no bullet points, no markdown.',
    'Explain the difference between explanation and excuse in one paragraph of about 150 words. Single paragraph, no preamble, no bullet points, no markdown.',
    'Explain whether reasons can be causes in one paragraph of about 150 words. Single paragraph, no preamble, no bullet points, no markdown.',
    'Explain what makes a life coherent in one paragraph of about 150 words. Single paragraph, no preamble, no bullet points, no markdown.',
    'Explain the difference between loyalty and obedience in one paragraph of about 150 words. Single paragraph, no preamble, no bullet points, no markdown.',
    'Explain whether it is possible to be objective about oneself in one paragraph of about 150 words. Single paragraph, no preamble, no bullet points, no markdown.',
    'Explain the difference between a desire and a preference in one paragraph of about 150 words. Single paragraph, no preamble, no bullet points, no markdown.',
    'Explain whether anger can ever be reasonable in one paragraph of about 150 words. Single paragraph, no preamble, no bullet points, no markdown.',
    'Explain what makes something sacred in one paragraph of about 150 words. Single paragraph, no preamble, no bullet points, no markdown.',
    'Explain the difference between meaning and purpose in one paragraph of about 150 words. Single paragraph, no preamble, no bullet points, no markdown.',
    'Explain whether we owe anything to future strangers in one paragraph of about 150 words. Single paragraph, no preamble, no bullet points, no markdown.',
    'Explain the difference between a tradition and a custom in one paragraph of about 150 words. Single paragraph, no preamble, no bullet points, no markdown.',
    'Explain what it means to be present in one paragraph of about 150 words. Single paragraph, no preamble, no bullet points, no markdown.',
    'Explain the difference between courage and recklessness in one paragraph of about 150 words. Single paragraph, no preamble, no bullet points, no markdown.',
    'Explain whether wisdom can be taught in one paragraph of about 150 words. Single paragraph, no preamble, no bullet points, no markdown.',
    'Explain why the sky is blue without using the word scattering in one paragraph of about 150 words. Single paragraph, no preamble, no bullet points, no markdown.',
    'Explain why the night sky is dark in one paragraph of about 150 words. Single paragraph, no preamble, no bullet points, no markdown.',
    'Explain what entropy means in one paragraph of about 150 words. Single paragraph, no preamble, no bullet points, no markdown.',
    'Explain why a spinning top stays upright in one paragraph of about 150 words. Single paragraph, no preamble, no bullet points, no markdown.',
    'Explain the difference between mass and weight in one paragraph of about 150 words. Single paragraph, no preamble, no bullet points, no markdown.',
    'Explain how a refrigerator moves heat in one paragraph of about 150 words. Single paragraph, no preamble, no bullet points, no markdown.',
    'Explain what would change if the speed of light were a thousand times slower in one paragraph of about 150 words. Single paragraph, no preamble, no bullet points, no markdown.',
    'Explain why ice floats in one paragraph of about 150 words. Single paragraph, no preamble, no bullet points, no markdown.',
    'Explain how a microwave heats food in one paragraph of about 150 words. Single paragraph, no preamble, no bullet points, no markdown.',
    'Explain why the tides happen twice a day in one paragraph of about 150 words. Single paragraph, no preamble, no bullet points, no markdown.',
    'Explain what a black hole is in one paragraph of about 150 words. Single paragraph, no preamble, no bullet points, no markdown.',
    'Explain why metals conduct electricity in one paragraph of about 150 words. Single paragraph, no preamble, no bullet points, no markdown.',
    'Explain how a rainbow forms in one paragraph of about 150 words. Single paragraph, no preamble, no bullet points, no markdown.',
    'Explain why hot water can freeze faster than cold water in one paragraph of about 150 words. Single paragraph, no preamble, no bullet points, no markdown.',
    'Explain what quantum tunnelling is in one paragraph of about 150 words. Single paragraph, no preamble, no bullet points, no markdown.',
    'Explain why the planets orbit in a plane in one paragraph of about 150 words. Single paragraph, no preamble, no bullet points, no markdown.',
    'Explain how sound travels through air in one paragraph of about 150 words. Single paragraph, no preamble, no bullet points, no markdown.',
    'Explain what dark matter might be in one paragraph of about 150 words. Single paragraph, no preamble, no bullet points, no markdown.',
    'Explain why the moon always shows the same face in one paragraph of about 150 words. Single paragraph, no preamble, no bullet points, no markdown.',
    'Explain how a laser works in one paragraph of about 150 words. Single paragraph, no preamble, no bullet points, no markdown.',
    'Explain why an airplane wing generates lift in one paragraph of about 150 words. Single paragraph, no preamble, no bullet points, no markdown.',
    'Explain what a phase transition is in one paragraph of about 150 words. Single paragraph, no preamble, no bullet points, no markdown.',
    'Explain why some materials are transparent in one paragraph of about 150 words. Single paragraph, no preamble, no bullet points, no markdown.',
    'Explain how nuclear fusion powers the sun in one paragraph of about 150 words. Single paragraph, no preamble, no bullet points, no markdown.',
    'Explain why friction heats things up in one paragraph of about 150 words. Single paragraph, no preamble, no bullet points, no markdown.',
    'Explain what a magnetic field is in one paragraph of about 150 words. Single paragraph, no preamble, no bullet points, no markdown.',
    'Explain why the sea is salty in one paragraph of about 150 words. Single paragraph, no preamble, no bullet points, no markdown.',
    'Explain how a thermometer measures temperature in one paragraph of about 150 words. Single paragraph, no preamble, no bullet points, no markdown.',
    'Explain why the sky turns red at sunset in one paragraph of about 150 words. Single paragraph, no preamble, no bullet points, no markdown.',
    'Explain what makes a wave a wave in one paragraph of about 150 words. Single paragraph, no preamble, no bullet points, no markdown.',
    'Explain why helium makes your voice higher in one paragraph of about 150 words. Single paragraph, no preamble, no bullet points, no markdown.',
    'Explain how electricity reaches a home in one paragraph of about 150 words. Single paragraph, no preamble, no bullet points, no markdown.',
    'Explain what the uncertainty principle says in one paragraph of about 150 words. Single paragraph, no preamble, no bullet points, no markdown.',
    'Explain why satellites stay in orbit in one paragraph of about 150 words. Single paragraph, no preamble, no bullet points, no markdown.',
    'Explain what a superconductor is in one paragraph of about 150 words. Single paragraph, no preamble, no bullet points, no markdown.',
    'Explain why light bends through glass in one paragraph of about 150 words. Single paragraph, no preamble, no bullet points, no markdown.',
    'Explain how an electric motor turns in one paragraph of about 150 words. Single paragraph, no preamble, no bullet points, no markdown.',
    'Explain what heat death means in one paragraph of about 150 words. Single paragraph, no preamble, no bullet points, no markdown.',
    'Explain why earthquakes happen in one paragraph of about 150 words. Single paragraph, no preamble, no bullet points, no markdown.',
    'Explain how a radio receives a signal in one paragraph of about 150 words. Single paragraph, no preamble, no bullet points, no markdown.',
]


@kbench.task(name="NTL passage generation",
             description="One 40-250 word paragraph per prompt, no markdown, no preamble.")
def generate(llm) -> None:
    import json
    for i, prompt in enumerate(PROMPTS):
        text = str(llm.prompt(prompt)).strip()
        kbench.assertions.assert_true(len(text) > 0, expectation=f"prompt {i} returned text")
        with open("passages.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({"prompt_id": i, "prompt": prompt, "text": text}) + "\n")


generate.run(kbench.llm)
