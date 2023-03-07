import gc
import json
from dataclasses import dataclass
from typing import Optional

import streamlit as st
import torch
from diffusers import DiffusionPipeline, LMSDiscreteScheduler
from loguru import logger
from PIL.PngImagePlugin import PngInfo

from stablefusion import utils
from stablefusion.scripts.mixdiff.tiling import StableDiffusionTilingPipeline


example_prompt = """[
        "A charming house in the countryside, by jakub rozalski, sunset lighting, elegant, highly detailed, smooth, sharp focus, artstation, stunning masterpiece",
        "A dirt road in the countryside crossing pastures, by jakub rozalski, sunset lighting, elegant, highly detailed, smooth, sharp focus, artstation, stunning masterpiece",
        "An old and rusty giant robot lying on a dirt road, by jakub rozalski, dark sunset lighting, elegant, highly detailed, smooth, sharp focus, artstation, stunning masterpiece"
    ]"""

@dataclass
class DiffusionMixture:
    device: Optional[str] = None
    model: Optional[str] = None
    output_path: Optional[str] = None

    def __str__(self) -> str:
        return f"DiffusionMixture(model={self.model}, device={self.device}, output_path={self.output_path})"

    def __post_init__(self):
        scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)
        self.pipeline = StableDiffusionTilingPipeline.from_pretrained(
            self.model,
            scheduler=scheduler,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        )
        self.pipeline.to(self.device)
        self.pipeline.safety_checker = utils.no_safety_checker
        self._compatible_schedulers = self.pipeline.scheduler.compatibles
        self.scheduler_config = self.pipeline.scheduler.config
        self.compatible_schedulers = {scheduler.__name__: scheduler for scheduler in self._compatible_schedulers}

        if self.device == "mps":
            self.pipeline.enable_attention_slicing()
            # warmup
            prompt = "a photo of an astronaut riding a horse on mars"
            _ = self.pipeline(prompt, num_inference_steps=2)

    def _set_scheduler(self, scheduler_name):
        scheduler = self.compatible_schedulers[scheduler_name].from_config(self.scheduler_config)
        self.pipeline.scheduler = scheduler

    def generate_image(self, prompt, scheduler, image_size, guidance_scale, steps, seed):
        self._set_scheduler(scheduler)
        logger.info(self.pipeline.scheduler)
        if self.device == "mps":
            generator = torch.manual_seed(seed)
            num_images = 1
        else:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        output_images = self.pipeline(
            [prompt],
            seed=7178915308,
        ).images
        torch.cuda.empty_cache()
        gc.collect()
        metadata = {
            "prompt": prompt,
            "scheduler": scheduler,
            "image_size": image_size,
            "num_images": num_images,
            "guidance_scale": guidance_scale,
            "steps": steps,
            "seed": seed,
        }
        metadata = json.dumps(metadata)
        _metadata = PngInfo()
        _metadata.add_text("text2img", metadata)

        utils.save_images(
            images=output_images,
            module="text2img",
            metadata=metadata,
            output_path=self.output_path,
        )

        return output_images, _metadata

    def app(self):
        available_schedulers = list(self.compatible_schedulers.keys())
        if "EulerAncestralDiscreteScheduler" in available_schedulers:
            available_schedulers.insert(
                0, available_schedulers.pop(available_schedulers.index("EulerAncestralDiscreteScheduler"))
            )
        # with st.form(key="text2img"):
        prompt = st.text_area("Prompt", example_prompt, help="Prompt to guide image generation")
        prompt = eval(str(prompt))
        # sidebar options
        scheduler = st.sidebar.selectbox("Scheduler", available_schedulers, index=0, help="Scheduler(Sampler) to use for generation")
        image_height = st.sidebar.slider("Image height", 128, 1024, 512, 128, help="The height in pixels of the generated image.")
        image_width = st.sidebar.slider("Image width", 128, 1024, 512, 128, help="The width in pixels of the generated image.")
        guidance_scale = st.sidebar.slider("Guidance scale", 1.0, 40.0, 7.5, 0.5, help="Higher guidance scale encourages to generate images that are closely linked to the text `prompt`, usually at the expense of lower image quality.")
        num_images = st.sidebar.slider("Number of images per prompt", 1, 30, 1, 1, help="Number of images you want to generate. More images requires more time and uses more GPU memory.")
        steps = st.sidebar.slider("Steps", 1, 150, 50, 1, help="The number of denoising steps. More denoising steps usually lead to a higher quality image at the expense of slower inference.")

        seed_placeholder = st.sidebar.empty()
        seed = seed_placeholder.number_input("Seed", value=42, min_value=1, max_value=999999, step=1)
        random_seed = st.sidebar.button("Random seed")
        _seed = torch.randint(1, 999999, (1,)).item()
        if random_seed:
            seed = seed_placeholder.number_input("Seed", value=_seed, min_value=1, max_value=999999, step=1)

        sub_col, download_col = st.columns(2)
        with sub_col:
            submit = st.button("Generate")

        if submit:
            with st.spinner("Generating images..."):
                output_images, metadata = self.generate_image(
                    prompt=prompt,
                    scheduler=scheduler,
                    image_size=(image_height, image_width),
                    guidance_scale=guidance_scale,
                    steps=steps,
                    seed=seed,
                )

            utils.display_and_download_images(output_images, metadata, download_col)
