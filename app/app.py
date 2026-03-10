import streamlit as st
import requests

st.title("Hindi Character Recognition")

uploaded_file = st.file_uploader(
    "Upload Hindi Character Image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:

    # Show uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    # Send image to FastAPI
    response = requests.post(
        "http://127.0.0.1:8000/predict",
        files={"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
    )

    if response.status_code == 200:

        result = response.json()

        # Show predicted Hindi character
        st.markdown(
            f"<h1 style='text-align:center; font-size:120px'>{result['character']}</h1>",
            unsafe_allow_html=True
        )

        st.success(f"Class: {result['class']}")
        st.write("Confidence:", round(result["confidence"] * 100, 2), "%")

    else:
        st.error("API request failed")