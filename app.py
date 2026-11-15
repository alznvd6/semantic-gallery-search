"""Streamlit UI for semantic photo search.

Run with:
    streamlit run app.py

TODO (Phase 3): polish the layout, add error states,
maybe support image-to-image search later.
"""
from pathlib import Path

import streamlit as st

from src.config import Config
from src import search as search_mod


st.set_page_config(page_title="Semantic Photo Search", layout="wide")
st.title("Semantic Photo Search")

cfg = Config()

query = st.text_input("Describe the photo you are looking for:")
k = st.slider("Number of results", 1, 30, cfg.top_k)

if query:
    with st.spinner("Searching..."):
        try:
            results = search_mod.search(query, cfg, k=k)
        except NotImplementedError:
            st.warning("Search is not implemented yet (Phase 2).")
            results = []

    if results:
        cols = st.columns(3)
        for i, (path, score) in enumerate(results):
            with cols[i % 3]:
                st.image(
                    path,
                    caption=f"{Path(path).name} — {score:.3f}",
                    use_container_width=True,
                )
    else:
        st.info("No matching photos found.")