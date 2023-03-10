import streamlit as st

from stablefusion import utils
from stablefusion.scripts.video_to_video import VideoToVideo
from stablefusion.Home import read_model_list

def app():
    utils.create_base_page()
    with st.form("inpainting_model_form"):
        model = st.selectbox(
            "Which model do you want to use for inpainting?",
            options=read_model_list()
        )
        submit = st.form_submit_button("Load model")
    if submit:
        st.session_state.inpainting_model = model
        with st.spinner("Loading model..."):
            X2Video = VideoToVideo(
                model=model,
                device=st.session_state.device,
                output_path=st.session_state.output_path,
            )
            st.session_state.X2Video = X2Video
    if "X2Video" in st.session_state:
        st.write(f"Current model: {st.session_state.X2Video}")
        st.session_state.X2Video.app()


if __name__ == "__main__":
    app()