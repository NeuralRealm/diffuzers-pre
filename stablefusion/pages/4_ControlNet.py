import streamlit as st

from stablefusion import utils
from stablefusion.scripts.controlnet import Controlnet
from stablefusion.Home import read_model_list

control_net_model_list = ["lllyasviel/sd-controlnet-canny",
                          "lllyasviel/sd-controlnet-hed",
                          "lllyasviel/sd-controlnet-normal",
                          "lllyasviel/sd-controlnet-scribble",
                          "lllyasviel/sd-controlnet-depth",
                          "lllyasviel/sd-controlnet-mlsd",
                          "lllyasviel/sd-controlnet-openpose",
                          "lllyasviel/sd-controlnet-seg"
                          ]

processer_list = [  "Canny",
                    "Hed",
                    "Normal",
                    "Scribble",
                    "Depth",
                    "Mlsd",
                    "OpenPose",
                    "Seg"
                ]


import streamlit as st

from stablefusion import utils
from stablefusion.scripts.gfp_gan import GFPGAN
from stablefusion.scripts.image_info import ImageInfo
from stablefusion.scripts.interrogator import ImageInterrogator
from stablefusion.scripts.upscaler import Upscaler
from stablefusion.scripts.model_adding import ModelAdding
from stablefusion.scripts.model_removing import ModelRemoving


def app():
    utils.create_base_page()
    with st.form("inpainting_model_form"):
        base_model = st.selectbox(
            "Base model For your ControlNet: ",
            options=read_model_list()
        )
        controlnet_model = st.selectbox(label="Choose Your ControlNet: ", options=control_net_model_list)
        processer = st.selectbox(label="Choose Your Processer: ", options=processer_list)
        submit = st.form_submit_button("Load model")
    if submit:
        st.session_state.inpainting_model = base_model
        with st.spinner("Loading model..."):
            inpainting = Controlnet(
                model=base_model,
                device=st.session_state.device,
                output_path=st.session_state.output_path,
                controlnet_model=controlnet_model,
                processer = processer
            )
            st.session_state.inpainting = inpainting
    if "inpainting" in st.session_state:
        st.write(f"Current model: {st.session_state.inpainting}")
        st.session_state.inpainting.app()


if __name__ == "__main__":
    app()