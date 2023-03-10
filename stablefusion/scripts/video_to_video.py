import cv2
import os
from typing import Optional
import gc
import json
from dataclasses import dataclass
from typing import Optional
from stablefusion import utils
import streamlit as st
import torch
from diffusers import DiffusionPipeline, StableDiffusionImg2ImgPipeline
from loguru import logger
from PIL.PngImagePlugin import PngInfo
import numpy as np
from PIL import Image

@dataclass
class VideoToVideo:
    device: Optional[str] = None
    model: Optional[str] = None
    output_path: Optional[str] = None

    def __str__(self) -> str:
        return f"Text2Image(model={self.model}, device={self.device}, output_path={self.output_path})"
    
    def __post_init__(self):
        self.pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
            self.model,
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
    
    def extract_video_to_image(self):
        
        image_dir = "{}/data/output/video_animations/images".format(utils.base_path())
        video_path = "{}/data/output/video_animations/uploaded_videos/uploaded_video.mp4".format(utils.base_path())
        
        
        video = cv2.VideoCapture(video_path)

        fps = video.get(cv2.CAP_PROP_FPS)
        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        # Loop through the frames and save them as images
        for i in range(frame_count):
            # Get the frame at the current position
            success, frame = video.read()
            if not success:
                break
            
            # Save the frame as an image
            cv2.imwrite(f"{image_dir}/images_{i}.jpg", frame)
            
            # Move to the next frame
            video.set(cv2.CAP_PROP_POS_FRAMES, (i+1))

        # Release the video object
        video.release()

        return fps


    def generate_ai_images(self, image, prompt, negative_prompt, scheduler, image_size, guidance_scale, steps, seed):
        
        image_dir = "{}/data/output/video_animations/images".format(utils.base_path())
        output_image_dir = "{}/data/output/video_animations/output_images/".format(utils.base_path())

        self._set_scheduler(scheduler)
        logger.info(self.pipeline.scheduler)
        if self.device == "mps":
            generator = torch.manual_seed(seed)
        else:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        image_files = [f for f in os.listdir(image_dir) if f.endswith(".jpg")]

        for image_file in image_files:

            image_path = os.path.join(image_dir, image_file)
            
            image = cv2.imread(image_path)
            resized_image = cv2.resize(image, (512, 512))
            rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb_image)

            output_image = self.pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image,
            width=image_size[1],
            height=image_size[0],
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            num_images_per_prompt=1,
            generator=generator,
            ).images[0]

            output_image.save("{}{}".format(output_image_dir, image_file))
            torch.cuda.empty_cache()
        
        return image_size[1], image_size[0]
    

    def images_to_video(fps, width, hight):

        # Set the path to the directory containing the image files
        image_dir = "{}/data/output/video_animations/output_images/".format(utils.base_path())

        # Get a list of all the image files in the directory
        image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]

        # Sort the list of image files by their numerical suffix
        image_files_sorted = sorted(image_files, key=lambda x: int(x.split('_')[1].split('.')[0]))

        # Define the codec and create a VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('output.mp4', fourcc, fps, (width, hight))

        # Iterate over each image file and write it to the video file
        for image_file in image_files_sorted:
            image_path = os.path.join(image_dir, image_file)
            image = cv2.imread(image_path)
            out.write(image)

        out.release()
    

    def app(self):
        available_schedulers = list(self.compatible_schedulers.keys())
        if "EulerAncestralDiscreteScheduler" in available_schedulers:
            available_schedulers.insert(
                0, available_schedulers.pop(available_schedulers.index("EulerAncestralDiscreteScheduler"))
            )
        # with st.form(key="text2img"):
        video_upload = st.file_uploader(label="Upload your video: ", type=["mp4"])
        if video_upload is not None:

            video_bytes = video_upload.read()

            with open("{}/data/output/video_animations/uploaded_videos/uploaded_video.mp4".format(utils.base_path()), "wb") as f:
                f.write(video_bytes)
                
            st.video(video_bytes, format='video/mp4')


        col1, col2 = st.columns(2)
        with col1:
            prompt = st.text_area("Prompt", "Blue elephant", help="Prompt to guide image generation")
        with col2:
            negative_prompt = st.text_area("Negative Prompt", "", help="The prompt not to guide image generation. Write things that you dont want to see in the image.")
        # sidebar options
        scheduler = st.sidebar.selectbox("Scheduler", available_schedulers, index=0, help="Scheduler(Sampler) to use for generation")
        image_height = st.sidebar.slider("Image height", 128, 1024, 512, 128, help="The height in pixels of the generated image.")
        image_width = st.sidebar.slider("Image width", 128, 1024, 512, 128, help="The width in pixels of the generated image.")
        guidance_scale = st.sidebar.slider("Guidance scale", 1.0, 40.0, 7.5, 0.5, help="Higher guidance scale encourages to generate images that are closely linked to the text `prompt`, usually at the expense of lower image quality.")
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
            with st.spinner("Generating Video..."):
                
                fps = self.extract_video_to_image()

                width, height = self.generate_ai_images(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    scheduler=scheduler,
                    image_size=(image_height, image_width),
                    guidance_scale=guidance_scale,
                    steps=steps,
                    seed=seed,
                )

                self.images_to_video(width=width, hight=height, fps=fps)
