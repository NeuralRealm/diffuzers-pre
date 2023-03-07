from diffusers import LMSDiscreteScheduler
from stablefusion.scripts.mixdiff.tiling import StableDiffusionTilingPipeline
import streamlit as st
# Creater scheduler and model (similar to StableDiffusionPipeline)

# Mixture of Diffusers generation

def mixer_app():
    scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)
    pipeline = StableDiffusionTilingPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", scheduler=scheduler).to("cuda:0")

    image = pipeline(
        prompt=[[
            "A charming house in the countryside, by jakub rozalski, sunset lighting, elegant, highly detailed, smooth, sharp focus, artstation, stunning masterpiece",
            "A dirt road in the countryside crossing pastures, by jakub rozalski, sunset lighting, elegant, highly detailed, smooth, sharp focus, artstation, stunning masterpiece",
            "An old and rusty giant robot lying on a dirt road, by jakub rozalski, dark sunset lighting, elegant, highly detailed, smooth, sharp focus, artstation, stunning masterpiece"
        ]],
        tile_height=640,
        tile_width=640,
        tile_row_overlap=0,
        tile_col_overlap=256,
        guidance_scale=8,
        seed=7178915308,
        num_inference_steps=50,
    )["sample"][0]

    st.image(image)