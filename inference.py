import torch
#################################### For Image ####################################
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import numpy as np
from skimage import measure
from sam3.visualization_utils import draw_box_on_image, normalize_bbox, plot_results
# Load the model

img_dir = "/nas/yu/dataset/rf20-vl/defect-detection/valid"
out_dir = "/nas/yu/code/sam3/output/defect-detection/valid/"
os.makedirs(out_dir, exist_ok=True)
model = build_sam3_image_model(
    checkpoint_path="/nas/yu/code/sam3/run/checkpoints/checkpoint_converted.pt",
    device="cuda" if torch.cuda.is_available() else "cpu",
    bpe_path="/nas/yu/code/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz",
    enable_segmentation=True,
    load_from_HF=False,
)
processor = Sam3Processor(model, confidence_threshold=0.0)

if os.path.isdir(img_dir):
    for img_name in os.listdir(img_dir):
        if not img_name.endswith((".jpg", ".jpeg", ".png")):
            continue
        image_path = os.path.join(img_dir, img_name)
        image = Image.open(image_path)
        inference_state = processor.set_image(image)
        # Prompt the model with text
        output = processor.set_text_prompt(state=inference_state, prompt="")
        img0 = Image.open(image_path)
        plot_results(img0, inference_state)
else:
    img_path = img_dir
    img_name = os.path.basename(img_path)
    image = Image.open(img_path)
    inference_state = processor.set_image(image)
    # Prompt the model with text
    output = processor.set_text_prompt(state=inference_state, prompt="")
    img0 = Image.open(img_path)
    plot_results(img0, inference_state)