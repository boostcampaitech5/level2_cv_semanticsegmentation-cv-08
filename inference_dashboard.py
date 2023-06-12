import os

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch
from streamlit import session_state as state
from torch.utils.data import DataLoader

import augmentations
from datasets import XRayInferenceDataset
from runner import test
from utils import CLASSES, decode_rle_to_mask, label2rgb, read_json


def load_local_model():
    with st.spinner("Loading model..."):
        state.trained_model = torch.load(state.config.inference_model_dir)
    st.success("Model loaded successfully!")


def load_dataset():
    tf = getattr(augmentations, state.config.test_augmentations.name)(
        **state.config.test_augmentations.parameters
    )
    test_dataset = XRayInferenceDataset(state.config, transforms=tf)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=state.config.test_batch_size,
        shuffle=False,
        num_workers=state.config.test_num_workers,
        drop_last=False,
    )
    state.test_loader = test_loader


def predict_all():
    with st.spinner("Running predictions..."):
        rles, filename_and_class = test(state.config, state.trained_model, state.test_loader)
        classes, filename = zip(*[x.split("_") for x in filename_and_class])
        image_name = [os.path.basename(f) for f in filename]
        df = pd.DataFrame(
            {
                "image_name": image_name,
                "class": classes,
                "rle": rles,
            }
        )
        state.df = df

        state.prediction_cache = {
            "rles": rles,
            "filename_and_class": filename_and_class,
        }


def visualise():
    for i in range(state.page - 1 * state.batch_size, state.page * state.batch_size):
        preds = []
        for rle in state.prediction_cache["rles"][
            i * len(CLASSES) : i * len(CLASSES) + len(CLASSES)
        ]:
            pred = decode_rle_to_mask(rle, height=2048, width=2048)
            preds.append(pred)
        if not preds:
            continue
        preds = np.stack(preds, 0)
        class_fname = state.prediction_cache["filename_and_class"][i * len(CLASSES)]
        image = cv2.imread(os.path.join(state.config.test_image_root, class_fname.split("_")[1]))
        seg_map = label2rgb(preds)
        image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_AREA)
        seg_map = cv2.resize(seg_map, (512, 512), interpolation=cv2.INTER_AREA)
        overaid_image = cv2.addWeighted(image, 0.4, seg_map, 0.5, 0)
        class_fname.split("_")[1]

        col1, col2, col3 = st.columns([0.3, 0.3, 0.3])

        with col1:
            st.markdown('<p style="text-align: center;">Original</p>', unsafe_allow_html=True)
            st.image(image, use_column_width="auto")
            st.caption(class_fname.split("_")[1])

        with col2:
            st.markdown(
                '<p style="text-align: center;">Segmentation Map</p>', unsafe_allow_html=True
            )
            st.image(seg_map, use_column_width="auto")

        with col3:
            st.markdown('<p style="text-align: center;">Overlay</p>', unsafe_allow_html=True)
            st.image(overaid_image, use_column_width="auto")


if __name__ == "__main__":
    st.set_page_config(page_title="Segmentation Inference Dashboard", layout="wide")
    if "prediction_cache" not in state:
        state.prediction_cache = {}
    if "curr_config_path" not in state:
        state.curr_config_path = {}
    st.header("Segmentation Inference Dashboard")

    st.text_input("Config Path", key="new_config_path")
    if not state.curr_config_path:
        state.curr_config_path = state.new_config_path
    elif state.curr_config_path != state.new_config_path:
        state.prediction_cache.clear()
        if "df" in state:
            state.df = {}
        state.curr_config_path = state.new_config_path
    state.num_batches = 30
    state.batch_size = 10
    state.page = st.selectbox("Page", range(1, state.num_batches + 1))
    if st.button(label="Download csv"):
        if "df" in state:
            state.df.to_csv(os.path.join(state.config.model_dir, "output.csv"), index=False)
    # print(state.prediction_cache)

    if len(state.curr_config_path) > 0:
        state.config = read_json(state.curr_config_path)
        if not state.prediction_cache:
            load_local_model()
            load_dataset()
            predict_all()
        visualise()
