import sys
from dataclasses import dataclass
from time import perf_counter

import torch
from diffusers import StableDiffusionXLPipeline
from torch import Generator, cosine_similarity, Tensor

from os import urandom
from random import sample, shuffle

import nltk

nltk.download('words')
nltk.download('universal_tagset')
nltk.download('averaged_perceptron_tagger')

from nltk.corpus import words
from nltk import pos_tag


MODEL_DIRECTORY = "model"
SAMPLE_COUNT = 5


AVAILABLE_WORDS = [word for word, tag in pos_tag(words.words(), tagset='universal') if tag == "ADJ" or tag == "NOUN"]


def generate_random_prompt():
    sampled_words = sample(AVAILABLE_WORDS, k=min(len(AVAILABLE_WORDS), min(urandom(1)[0] % 32, 8)))
    shuffle(sampled_words)

    return ", ".join(sampled_words)


@dataclass
class CheckpointBenchmark:
    baseline_average: float
    average_time: float
    average_similarity: float
    failed: bool


@dataclass
class GenerationOutput:
    prompt: str
    seed: int
    output: Tensor
    generation_time: float


def generate(pipeline: StableDiffusionXLPipeline, prompt: str, seed: int):
    start = perf_counter()

    output = pipeline(
        prompt=prompt,
        generator=Generator(pipeline.device).manual_seed(seed),
        output_type="latent",
        num_inference_steps=20,
    ).images

    generation_time = perf_counter() - start

    return GenerationOutput(
        prompt=prompt,
        seed=seed,
        output=output,
        generation_time=generation_time,
    )


def compare_checkpoints():
    baseline_pipeline = StableDiffusionXLPipeline.from_pretrained(
        "stablediffusionapi/newdream-sdxl-20",
        torch_dtype=torch.float16,
    ).to("cuda")

    print("Generating baseline samples to compare")

    baseline_outputs: list[GenerationOutput] = [
        generate(
            baseline_pipeline,
            generate_random_prompt(),
            int.from_bytes(urandom(4), "little"),
        )
        for _ in range(SAMPLE_COUNT)
    ]

    del baseline_pipeline

    torch.cuda.empty_cache()

    baseline_average = sum([output.generation_time for output in baseline_outputs]) / len(baseline_outputs)

    average_time = float("inf")
    average_similarity = 1.0

    pipeline = StableDiffusionXLPipeline.from_pretrained(
        MODEL_DIRECTORY,
        torch_dtype=torch.float16,
        use_safetensors=True,
    ).to("cuda")

    i = 0

    # Take {SAMPLE_COUNT} samples, keeping track of how fast/accurate generations have been
    for i, baseline in enumerate(baseline_outputs):
        print(f"Sample {i}, prompt {baseline.prompt} and seed {baseline.seed}")

        generated = i
        remaining = SAMPLE_COUNT - generated

        generation = generate(
            pipeline,
            baseline.prompt,
            baseline.seed,
        )

        similarity = (cosine_similarity(
            baseline.output.flatten(),
            generation.output.flatten(),
            eps=1e-3,
            dim=0,
        ).item() * 0.5 + 0.5) ** 4

        print(
            f"Sample {i} generated "
            f"with generation time of {generation.generation_time} "
            f"and similarity {similarity}"
        )

        if generated:
            average_time = (average_time * generated + generation.generation_time) / (generated + 1)
        else:
            average_time = generation.generation_time

        average_similarity = (average_similarity * generated + similarity) / (generated + 1)

        if average_time < baseline_average * 1.0625:
            # So far, the average time is better than the baseline, so we can continue
            continue

        needed_time = (baseline_average * SAMPLE_COUNT - generated * average_time) / remaining

        if needed_time < average_time * 0.75:
            # Needs %33 faster than current performance to beat the baseline,
            # thus we shouldn't waste compute testing farther
            print("Too different from baseline, failing", file=sys.stderr)
            break

        if average_similarity < 0.85:
            # Deviating too much from original quality
            print("Too different from baseline, failing", file=sys.stderr)
            break

    print(
        f"Tested {i + 1} samples, "
        f"average similarity of {average_similarity}, "
        f"and speed of {average_time}"
    )


if __name__ == '__main__':
    compare_checkpoints()
