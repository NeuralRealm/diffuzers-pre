import gc
import json
from dataclasses import dataclass
from io import BytesIO
from typing import Optional, Union

import requests
import streamlit as st
import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from loguru import logger
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from stablefusion import utils
import cv2
from PIL import Image
import numpy as np


def canny_processor():

    image = np.array(image)

    low_threshold = 100
    high_threshold = 200

    image = cv2.Canny(image, low_threshold, high_threshold)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    canny_image = Image.fromarray(image)
    
    return canny_image


@dataclass
class Img2Img:
    model: Optional[str] = None
    device: Optional[str] = None
    output_path: Optional[str] = None
    controlnet_model: Optional[str] = None
    processer: Optional[str] = None

    def __str__(self) -> str:
        return f"BaseModel(model={self.model}, device={self.device}, ControlNet={self.controlnet}, processer={self.processer}, output_path={self.output_path})"
    
    
    def __post_init__(self):
        self.controlnet = ControlNetModel.from_pretrained(
                self.controlnet_model,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                use_auth_token=utils.use_auth_token(),
            )
        self.pipeline = StableDiffusionControlNetPipeline.from_pretrained(
            self.model,
            controlnet=self.controlnet,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            use_auth_token=utils.use_auth_token(),
        )

        self.pipeline.to(self.device)
        self.pipeline.safety_checker = utils.no_safety_checker
        self._compatible_schedulers = self.pipeline.scheduler.compatibles
        self.scheduler_config = self.pipeline.scheduler.config
        self.compatible_schedulers = {scheduler.__name__: scheduler for scheduler in self._compatible_schedulers}

        if self.device == "mps":
            self.pipeline.enable_attention_slicing()
            # warmup
            url = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg"
            response = requests.get(url)
            init_image = Image.open(BytesIO(response.content)).convert("RGB")
            init_image.thumbnail((768, 768))
            prompt = "A fantasy landscape, trending on artstation"
            _ = self.pipeline(
                prompt=prompt,
                image=init_image,
                strength=0.75,
                guidance_scale=7.5,
                num_inference_steps=2,
            )

    def _set_scheduler(self, scheduler_name):
        scheduler = self.compatible_schedulers[scheduler_name].from_config(self.scheduler_config)
        self.pipeline.scheduler = scheduler

    def generate_image(
        self, prompt, image, strength, negative_prompt, scheduler, num_images, guidance_scale, steps, seed
    ):
        self._set_scheduler(scheduler)
        logger.info(self.pipeline.scheduler)
        if self.device == "mps":
            generator = torch.manual_seed(seed)
            num_images = 1
        else:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        num_images = int(num_images)
        output_images = self.pipeline(
            prompt=prompt,
            image=image,
            strength=strength,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_images,
            generator=generator,
        ).images
        torch.cuda.empty_cache()
        gc.collect()
        metadata = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "scheduler": scheduler,
            "num_images": num_images,
            "guidance_scale": guidance_scale,
            "steps": steps,
            "seed": seed,
        }
        metadata = json.dumps(metadata)
        _metadata = PngInfo()
        _metadata.add_text("img2img", metadata)

        utils.save_images(
            images=output_images,
            module="controlnet",
            metadata=metadata,
            output_path=self.output_path,
        )
        return output_images, _metadata

    def app(self):
        available_schedulers = list(self.compatible_schedulers.keys())
        # if EulerAncestralDiscreteScheduler is available in available_schedulers, move it to the first position
        if "EulerAncestralDiscreteScheduler" in available_schedulers:
            available_schedulers.insert(
                0, available_schedulers.pop(available_schedulers.index("EulerAncestralDiscreteScheduler"))
            )

        input_image = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
        if input_image is not None:

            if self.processer == "Canny":
                processed_image = canny_processor()
                
            input_image = Image.open(processed_image)
            st.image(input_image, use_column_width=True)

        # with st.form(key="img2img"):
        col1, col2 = st.columns(2)
        with col1:
            prompt = st.text_area("Prompt", "", help="Prompt to guide image generation")
        with col2:
            negative_prompt = st.text_area("Negative Prompt", "", help="The prompt not to guide image generation. Write things that you dont want to see in the image.")

        scheduler = st.sidebar.selectbox("Scheduler", available_schedulers, index=0, help="Scheduler(Sampler) to use for generation")
        guidance_scale = st.sidebar.slider("Guidance scale", 1.0, 40.0, 7.5, 0.5, help="Higher guidance scale encourages to generate images that are closely linked to the text `prompt`, usually at the expense of lower image quality.")
        strength = st.sidebar.slider("Denoise Strength", 0.0, 1.0, 0.5, 0.05)
        num_images = st.sidebar.slider("Number of images per prompt", 1, 30, 1, 1, help="Number of images you want to generate. More images requires more time and uses more GPU memory.")
        steps = st.sidebar.slider("Steps", 1, 150, 50, 1, help="The number of denoising steps. More denoising steps usually lead to a higher quality image at the expense of slower inference.")
        seed_placeholder = st.sidebar.empty()
        seed = seed_placeholder.number_input("Seed", value=42, min_value=1, max_value=999999, step=1)
        random_seed = st.sidebar.button("Random seed")
        _seed = torch.randint(1, 999999, (1,)).item()
        if random_seed:
            seed = seed_placeholder.number_input("Seed", value=_seed, min_value=1, max_value=999999, step=1)
        # seed = st.sidebar.number_input("Seed", 1, 999999, 1, 1)
        sub_col, download_col = st.columns(2)
        with sub_col:
            submit = st.button("Generate")

        if submit:
            with st.spinner("Generating images..."):
                output_images, metadata = self.generate_image(
                    prompt=prompt,
                    image=input_image,
                    negative_prompt=negative_prompt,
                    scheduler=scheduler,
                    num_images=num_images,
                    guidance_scale=guidance_scale,
                    steps=steps,
                    seed=seed,
                    strength=strength,
                )

            utils.display_and_download_images(output_images, metadata, download_col)
