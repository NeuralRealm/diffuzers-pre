import streamlit as st

from stablefusion import utils
from stablefusion.scripts.diffusion_mixture import DiffusionMixture
from stablefusion.Home import read_model_list


def app():
    utils.create_base_page()
    with st.form("diffusion_mixture_form"):
        model = st.selectbox(
            "Which model do you want to use for Diffusion Mixture?",
            options=read_model_list()
        )
        submit = st.form_submit_button("Load model")
    if submit:
        with st.spinner("Loading model..."):
            diffusion_mixture = DiffusionMixture(
                model=model,
                device=st.session_state.device,
                output_path=st.session_state.output_path,
            )
            st.session_state.diffusion_mixture = diffusion_mixture
    if "diffusion_mixture" in st.session_state:
        st.write(f"Current model: {st.session_state.diffusion_mixture}")
        st.session_state.diffusion_mixture.app()


if __name__ == "__main__":
    app()