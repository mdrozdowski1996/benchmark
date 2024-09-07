import sys
from dataclasses import dataclass
from time import perf_counter

import torch

import oneflow as flow
from onediff.infer_compiler import oneflow_compile

from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel
from diffusers.models.attention_processor import Attention, AttnProcessor
from torch import Generator, cosine_similarity, Tensor

from os import urandom
from random import sample, shuffle
from peft import LoraConfig, get_peft_model, TaskType

import nltk
import argparse

nltk.download('words')
nltk.download('universal_tagset')
nltk.download('averaged_perceptron_tagger')

from nltk.corpus import words
from nltk import pos_tag
from txt2img_xl_pipeline import Txt2ImgXLPipeline
from utilities import TRT_LOGGER, add_arguments

from cuda import cudart


MODEL_DIRECTORY = "stablediffusionapi/newdream-sdxl-20"
SAMPLE_COUNT = 5
BASELINE_AVERAGE = 2.58


AVAILABLE_WORDS = [word for word, tag in pos_tag(words.words(), tagset='universal') if tag == "ADJ" or tag == "NOUN"]


def parseArgs():
    parser = argparse.ArgumentParser(description="Options for Stable Diffusion XL Txt2Img Demo", conflict_handler='resolve')
    parser = add_arguments(parser)
    parser.add_argument('--version', type=str, default="xl-1.0", choices=["xl-1.0"], help="Version of Stable Diffusion XL")
    parser.add_argument('--height', type=int, default=1024, help="Height of image to generate (must be multiple of 8)")
    parser.add_argument('--width', type=int, default=1024, help="Height of image to generate (must be multiple of 8)")

    parser.add_argument('--scheduler', type=str, default="DDIM", choices=["PNDM", "LMSD", "DPM", "DDIM", "EulerA"], help="Scheduler for diffusion process")

    parser.add_argument('--onnx-dir', default='onnx_xl_base', help="Directory for SDXL-Base ONNX models")
    
    parser.add_argument('--engine-dir', default='engine_xl_base', help="Directory for SDXL-Base TensorRT engines")

    return parser.parse_args()


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


def calculate_score(model_average: float, similarity: float) -> float:
    return max(
        0.0,
        BASELINE_AVERAGE - model_average
    ) * similarity


def generate(pipeline: StableDiffusionXLPipeline, prompt: str, seed: int):
    start = perf_counter()

    output = pipeline(
        prompt=prompt,
        generator=Generator(pipeline.device).manual_seed(seed),
        output_type="latent",
        num_inference_steps=20,
    ).images

    generation_time = perf_counter() - start

    #output[0].save("custom_output.png")
    print("generate", output.shape)

    return GenerationOutput(
        prompt=prompt,
        seed=seed,
        output=output,
        generation_time=generation_time,
    )

def generate_trt(demo_base, prompt: str, seed: int, warmup, verbose):
    prompt_l = []
    prompt_l.append(prompt)
    negative_prompt_l = [""]
    start = perf_counter()
    images, _ = demo_base.infer(prompt_l, negative_prompt_l, 1024, 1024, warmup=warmup, verbose=verbose, seed=args.seed, return_type="latents")
    generation_time = perf_counter() - start

    print("Trt", images[0].shape)

    return GenerationOutput(
        prompt=prompt,
        seed=seed,
        output=images[0],
        generation_time=generation_time,
    )

def create_pipeline():

    '''unet = UNet2DConditionModel.from_pretrained(
        "stablediffusionapi/newdream-sdxl-20",
        subfolder="unet",
        torch_dtype=torch.float16
    ).to("cuda")'''

    '''model_int8 = torch.ao.quantization.quantize_dynamic(
        unet,  # the original model
        {torch.nn.Linear, torch.nn.Conv2d},  # a set of layers to dynamically quantize
        dtype=torch.qint8)'''

   
    
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        "stablediffusionapi/newdream-sdxl-20",
        torch_dtype=torch.float16,
        use_safetensors=True
    ).to("cuda")

    pipeline.unet = oneflow_compile(pipeline.unet)


    #pipeline.enable_xformers_memory_efficient_attention()
    #pipeline.unet = torch.jit.trace(pipeline.unet, example_inputs=(torch.randn(1, 4, 64, 64).to("cuda"),))
    #pipeline.unet = torch.compile(pipeline.unet, mode="reduce-overhead", fullgraph=True)
    
    return pipeline

def generate_pipeline():


    def get_specific_layer_names(model):
        # Create a list to store the layer names
        layer_names = []
    
        # Recursively visit all modules and submodules
        for name, module in model.named_modules():
            #print(name)
            # Check if the module is an instance of the specified layers
            if isinstance(module, (torch.nn.Linear, torch.nn.Embedding, torch.nn.Conv2d)):
                # model name parsing 
                layer_names.append(name)
                #layer_names.append('.'.join(name.split('.')[4:]).split('.')[0])
    
        return layer_names
    
    

    unet = UNet2DConditionModel.from_pretrained(
        "stablediffusionapi/newdream-sdxl-20",
        subfolder="unet",
    )
    tar = list(set(get_specific_layer_names(unet)))
    print(tar)
    config = LoraConfig(r=16, target_modules=tar, inference_mode=True)
    # Train this Peft-wrapped UNet
    lora = get_peft_model(unet, config)

    lora.merge_and_unload()
    #lora.half()

    pipeline = StableDiffusionXLPipeline.from_pretrained(
        "stablediffusionapi/newdream-sdxl-20",
        unet=lora,
    )
    return pipeline
    '''start = perf_counter()

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
    )'''



def compare_checkpoints(args):
    baseline_pipeline = StableDiffusionXLPipeline.from_pretrained(
        "stablediffusionapi/newdream-sdxl-20",
        torch_dtype=torch.float16,
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

    average_time = float("inf")
    average_similarity = 1.0

    def init_pipeline(pipeline_class, refinfer, onnx_dir, engine_dir, args):
        # Initialize demo
        demo = pipeline_class(
            scheduler=args.scheduler,
            denoising_steps=20,
            output_dir=args.output_dir,
            version=args.version,
            hf_token=args.hf_token,
            verbose=args.verbose,
            nvtx_profile=args.nvtx_profile,
            max_batch_size=16,
            use_cuda_graph=args.use_cuda_graph,
            refiner=refinfer,
            framework_model_dir=args.framework_model_dir)

        # Load TensorRT engines and pytorch modules
        demo.loadEngines(engine_dir, args.framework_model_dir, onnx_dir, args.onnx_opset,
            opt_batch_size=1, opt_image_height=1024, opt_image_width=1024, \
            force_export=args.force_onnx_export, force_optimize=args.force_onnx_optimize, \
            force_build=args.force_engine_build,
            static_batch=args.build_static_batch, static_shape=not args.build_dynamic_shape, \
            enable_refit=args.build_enable_refit, enable_preview=args.build_preview_features, \
            enable_all_tactics=args.build_all_tactics, \
            timing_cache=args.timing_cache, onnx_refit_dir=args.onnx_refit_dir)
        return demo
    '''
    demo_base = init_pipeline(Txt2ImgXLPipeline, False, args.onnx_dir, args.engine_dir, args)
    max_device_memory = demo_base.calculateMaxDeviceMemory()
    print(max_device_memory)
    _, shared_device_memory = cudart.cudaMalloc(max_device_memory)
    
    demo_base.activateEngines(shared_device_memory)
    demo_base.loadResources(1024, 1024, 1, args.seed)

    '''
    

    #generate_trt(demo_base, prompt, seed, warmup, verbose)

    '''pipeline = StableDiffusionXLPipeline.from_pretrained(
        MODEL_DIRECTORY,
        torch_dtype=torch.float16,
        use_safetensors=True,
    ).to("cuda")

    pipeline(prompt="")'''
    
    i = 0
    pipe = create_pipeline()
    pipe(prompt="")

    
    with flow.autocast('cuda'):

        # Take {SAMPLE_COUNT} samples, keeping track of how fast/accurate generations have been
        for i, baseline in enumerate(baseline_outputs):
            print(f"Sample {i}, prompt {baseline.prompt} and seed {baseline.seed}")

            generated = i
            remaining = SAMPLE_COUNT - generated

            '''
            generation = generate_trt(
                demo_base,
                baseline.prompt,
                baseline.seed,
                warmup=False,
                verbose=False
            )'''
            generation = generate(
                pipe,
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
            f"with a final score of {calculate_score(average_time, average_similarity)}"
        )


if __name__ == '__main__':
    args = parseArgs()
    compare_checkpoints(args)
