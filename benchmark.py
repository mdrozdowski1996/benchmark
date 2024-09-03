import sys
from dataclasses import dataclass
from os import urandom
from pathlib import Path
from random import sample, shuffle
from time import perf_counter

import nltk
import pynvml
import torch
from diffusers import StableDiffusionXLPipeline
from huggingface_hub import constants
from torch import Generator, cosine_similarity, Tensor

from vram_monitor import VRamMonitor

nltk.download('words')
nltk.download('universal_tagset')
nltk.download('averaged_perceptron_tagger')

from nltk.corpus import words
from nltk import pos_tag

BASELINE_REPOSITORY = "stablediffusionapi/newdream-sdxl-20"
MODEL_CACHE_DIR = Path("model-cache")
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
    baseline_size: int
    size: int
    baseline_vram_used: float
    vram_used: float
    baseline_watts_used: float
    watts_used: float
    failed: bool


@dataclass
class GenerationOutput:
    prompt: str
    seed: int
    output: Tensor
    generation_time: float
    vram_used: float
    watts_used: float


def calculate_score(baseline_average: float, model_average: float, similarity: float) -> float:
    return max(
        0.0,
        baseline_average - model_average
    ) * similarity


def get_baseline_size():
    baseline_dir = Path(constants.HF_HUB_CACHE) / f"models--{BASELINE_REPOSITORY.replace('/', '--')}"
    return sum(file.stat().st_size for file in baseline_dir.rglob("*"))


def get_model_size():
    return sum(file.stat().st_size for file in MODEL_CACHE_DIR.rglob("*"))


def get_joules(device: torch.device):
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(device.index)
    mj = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
    pynvml.nvmlShutdown()
    return mj / 1000.0  # convert mJ to J


def generate(pipeline: StableDiffusionXLPipeline, prompt: str, seed: int):
    start_joules = get_joules(pipeline.device)
    vram_monitor = VRamMonitor(pipeline.device)
    start = perf_counter()

    output = pipeline(
        prompt=prompt,
        generator=Generator(pipeline.device).manual_seed(seed),
        output_type="latent",
        num_inference_steps=20,
    ).images

    generation_time = perf_counter() - start
    joules_used = get_joules(pipeline.device) - start_joules
    watts_used = joules_used / generation_time
    vram_used = vram_monitor.complete()

    return GenerationOutput(
        prompt=prompt,
        seed=seed,
        output=output,
        generation_time=generation_time,
        vram_used=vram_used,
        watts_used=watts_used,
    )


def compare_checkpoints():
    baseline_pipeline = StableDiffusionXLPipeline.from_pretrained(
        BASELINE_REPOSITORY,
        torch_dtype=torch.float16,
        cache_dir=MODEL_CACHE_DIR,
    ).to("cuda")

    baseline_pipeline(prompt="")

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
    baseline_size = get_baseline_size()
    baseline_vram_used = sum([output.vram_used for output in baseline_outputs]) / len(baseline_outputs)
    baseline_watts_used = sum([output.watts_used for output in baseline_outputs]) / len(baseline_outputs)

    average_time = float("inf")
    average_similarity = 1.0

    pipeline = StableDiffusionXLPipeline.from_pretrained(
        MODEL_DIRECTORY,
        torch_dtype=torch.float16,
        use_safetensors=True,
    ).to("cuda")

    pipeline(prompt="")

    size = get_model_size()
    vram_used = 0.0
    watts_used = 0.0

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
            f"with generation time of {generation.generation_time}, "
            f"and similarity {similarity}, "
            f"and VRAM usage of {generation.vram_used}, "
            f"and watts usage of {generation.watts_used}."
        )

        if generated:
            average_time = (average_time * generated + generation.generation_time) / (generated + 1)
            vram_used = (baseline.vram_used * generated + generation.vram_used) / (generated + 1)
            watts_used = (baseline.watts_used * generated + generation.watts_used) / (generated + 1)
        else:
            average_time = generation.generation_time
            vram_used = generation.vram_used
            watts_used = generation.watts_used

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
        "Calculated baseline metrics "
        f"with a speed of {baseline_average}, "
        f"and model size of {baseline_size}, "
        f"and VRAM usage of {baseline_vram_used}, "
        f"and watts usage of {baseline_watts_used}."
    )

    print(
        f"Tested {i + 1} samples, "
        f"average similarity of {average_similarity}, "
        f"and speed of {average_time}, "
        f"and model size of {size}, "
        f"and VRAM usage of {vram_used}, "
        f"and watts usage of {watts_used}, "
        f"with a final score of {calculate_score(baseline_average, average_time, average_similarity)}."
    )


if __name__ == '__main__':
    compare_checkpoints()
