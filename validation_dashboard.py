import os

import cv2
import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F
from streamlit import session_state as state

import augmentations
from datasets import XRayDatasetV2
from utils import label2rgb, read_json


def load_local_model():
    with st.spinner("Loading model..."):
        state.trained_model = torch.load(state.config.inference_model_dir)
    st.success("Model loaded successfully!")


def load_dataset():
    tf = getattr(augmentations, state.config.valid.augmentations.name)(
        **state.config.valid.augmentations.parameters
    )
    valid_dataset = XRayDatasetV2(state.config, is_train=False, transforms=tf)
    state.valid_dataset = valid_dataset
    state.fnames = valid_dataset.fnames


def predict(idx, thr=0.5):
    img, label = state.valid_dataset[idx]
    img_name = state.fnames[idx]
    state.trained_model.eval()
    with st.spinner("Running prediction..."):
        with torch.no_grad():
            img = torch.unsqueeze(img, 0).cuda()
            output = state.trained_model(img)

            # restore original size
            output = F.interpolate(output, size=(2048, 2048), mode="bilinear")
            output = torch.sigmoid(output)
            output = (output > thr).detach().cpu().numpy()[0]

            label = torch.unsqueeze(label, 0)
            label = F.interpolate(label, size=(2048, 2048), mode="bilinear")

            false_positive = np.where(label - output < 0, 1, 0)[0]
            false_negative = np.where(label - output > 0, 1, 0)[0]

        state.prediction_cache[idx] = {
            "pred": output,
            "filename": img_name,
            "label": label[0],
            "false_positive": false_positive,
            "false_negative": false_negative,
        }


def visualize(idx):
    col1, col2, col3, col4, col5 = st.columns([0.2, 0.2, 0.2, 0.2, 0.2])

    img_fname = state.prediction_cache[idx]["filename"]
    image = cv2.imread(os.path.join(state.config.image_dir, f"{img_fname}.png"))
    image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_AREA)

    gt = state.prediction_cache[idx]["label"]
    gt = label2rgb(gt)
    gt = cv2.resize(gt, (512, 512), interpolation=cv2.INTER_AREA)

    seg_map = state.prediction_cache[idx]["pred"]
    seg_map = label2rgb(seg_map)
    seg_map = cv2.resize(seg_map, (512, 512), interpolation=cv2.INTER_AREA)

    false_positive = state.prediction_cache[idx]["false_positive"]
    false_positive = label2rgb(false_positive)
    false_positive = cv2.resize(false_positive, (512, 512), interpolation=cv2.INTER_AREA)

    false_negative = state.prediction_cache[idx]["false_negative"]
    false_negative = label2rgb(false_negative)
    false_negative = cv2.resize(false_negative, (512, 512), interpolation=cv2.INTER_AREA)

    with col1:
        st.markdown('<p style="text-align: center;">Input</p>', unsafe_allow_html=True)
        st.image(image, use_column_width="auto")
        st.caption(img_fname)

    with col2:
        st.markdown('<p style="text-align: center;">Ground Truth</p>', unsafe_allow_html=True)
        st.image(gt, use_column_width="auto")

    with col3:
        st.markdown('<p style="text-align: center;">Prediction</p>', unsafe_allow_html=True)
        st.image(seg_map, use_column_width="auto")

    with col4:
        st.markdown('<p style="text-align: center;">False Positive</p>', unsafe_allow_html=True)
        st.image(false_positive, use_column_width="auto")

    with col5:
        st.markdown('<p style="text-align: center;">False Negative</p>', unsafe_allow_html=True)
        st.image(false_negative, use_column_width="auto")


if __name__ == "__main__":
    st.set_page_config(page_title="Segmentation Inference Dashboard", layout="wide")

    if "prediction_cache" not in state:
        state.prediction_cache = {}

    if "curr_config_path" not in state:
        state.curr_config_path = {}

    st.header("Validation Results")

    st.text_input("Config Path", key="new_config_path")

    if not state.curr_config_path:
        state.curr_config_path = state.new_config_path
    elif state.curr_config_path != state.new_config_path:
        state.prediction_cache.clear()
        state.curr_config_path = state.new_config_path

    if len(state.curr_config_path) > 0:
        state.config = read_json(state.curr_config_path)
        if not state.prediction_cache:
            load_local_model()
            load_dataset()

        fname_list = [f"{i}: {f}" for i, f in enumerate(state.fnames)]
        idx_str = st.selectbox("Choose image", fname_list)
        idx = int(idx_str.split(":")[0])

        if idx not in state.prediction_cache.keys():
            predict(idx)

        visualize(idx)
