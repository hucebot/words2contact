import os
import warnings
from typing import List

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    CLIPSegForImageSegmentation,
    AutoModelForCausalLM,
    Owlv2ForObjectDetection,
)
from groundingdino.util.inference import Model
from geom_utils import BoundingBox

# Set matplotlib to interactive and suppress warnings
import matplotlib
matplotlib.rcParams["interactive"] = True
warnings.filterwarnings("ignore")

# Constants
WEIGHTS_NAME = "groundingdino_swint_ogc.pth"
CONFIG_PATH = "config/GroundingDINO_SwinT_OGC.py"
WEIGHTS_PATH = os.path.join("models/VLMs/GroundingDINO", WEIGHTS_NAME)
BOX_THRESHOLD = 0.2
TEXT_THRESHOLD = 0.2
TEXT_PROMPT = ["windex"]

# Ensure image path and cache directories exist
IMAGE_PATH = (
    "dataset/images/53e2f90e25d2d125215b2c7f1612f54959ef89b779c33a40e341e72d62b45760.png"
)

class Yello:
    """
    A class for performing Open Vocabulary Object Detection using different Vision-Language Models (VLMs).
    """

    def __init__(self, vlm: str = "CLIPSeg", debug: bool = False, device: str = "cuda"):
        assert vlm in ["CLIPSeg", "GroundingDINO", "Owlv2", "Florence-2"], (
            f"Invalid VLM: {vlm}. Supported VLMs: CLIPSeg, GroundingDINO, Owlv2, Florence-2"
        )
        self.vlm = vlm
        self.debug = debug
        self.device = device
        self.cache_dir = "./models/VLMs/"

        if vlm == "GroundingDINO":
            self.load_groundingdino()
        elif vlm == "Florence-2":
            self.load_florence2()
        elif vlm == "CLIPSeg":
            self.load_clipseg()
        elif vlm == "Owlv2":
            self.load_owlv2()

    def load_clipseg(self):
        """Load the CLIPSeg model."""
        if self.debug:
            print("Loading CLIPSeg Model...")
        self.processor = AutoProcessor.from_pretrained(
            "CIDAS/clipseg-rd64-refined", cache_dir=self.cache_dir
        )
        self.model = CLIPSegForImageSegmentation.from_pretrained(
            "CIDAS/clipseg-rd64-refined", cache_dir=self.cache_dir
        ).to(self.device)
        if self.debug:
            print("CLIPSeg Model loaded successfully.")

    def load_groundingdino(self):
        """Load the GroundingDINO model."""
        if self.debug:
            print("Loading GroundingDINO Model...")
        self.model = Model(CONFIG_PATH, WEIGHTS_PATH, device=self.device)
        if self.debug:
            print("GroundingDINO Model loaded successfully.")

    def load_owlv2(self):
        """Load the Owlv2 model."""
        if self.debug:
            print("Loading Owlv2 Model...")
        self.processor = AutoProcessor.from_pretrained("google/owlv2-base-patch16")
        self.model = Owlv2ForObjectDetection.from_pretrained(
            "google/owlv2-base-patch16", cache_dir=self.cache_dir
        ).to(self.device)
        if self.debug:
            print("Owlv2 Model loaded successfully.")

    def load_florence2(self):
        """Load the Florence-2 model."""
        if self.debug:
            print("Loading Florence-2 Model...")
        model_name = "microsoft/Florence-2-large"
        self.processor = AutoProcessor.from_pretrained(
            model_name, trust_remote_code=True, cache_dir=self.cache_dir
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, trust_remote_code=True
        ).to(self.device).eval()
        if self.debug:
            print("Florence-2 Model loaded successfully.")

    def predict(self, img: np.array, objects: List[str]) -> List[BoundingBox]:
        """
        Perform prediction using the selected VLM.
        Args:
            img (np.array): Input image in NumPy array format.
            objects (List[str]): List of objects to detect.

        Returns:
            List[BoundingBox]: List of detected bounding boxes.
        """
        if self.vlm == "CLIPSeg":
            return self.predict_clipseg(img, objects)
        elif self.vlm == "GroundingDINO":
            return self.predict_groundingdino(img, objects)
        elif self.vlm == "Owlv2":
            return self.predict_owlv2(img, objects)
        elif self.vlm == "Florence-2":
            return self.predict_florence2(img, objects)

    def predict_clipseg(self, img: np.array, objects: List[str]) -> List[BoundingBox]:
        """
        Predict using the CLIPSeg model.
        """
        img_pil = Image.fromarray(img)
        inputs = self.processor(
            text=objects, images=[img_pil] * len(objects), padding=True, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Handle output dimensions
        predictions = (
            outputs.logits.detach().unsqueeze(0)
            if outputs.logits.dim() == 3
            else outputs.logits.detach().unsqueeze(1)
        )

        bbs = []
        for i, obj in enumerate(objects):
            seg_heatmap = torch.sigmoid(predictions[i][0])
            seg_heatmap_resized = self.resize_segmentation(seg_heatmap, img_pil.size)

            # Extract bounding boxes
            bbs.append(self.extract_bounding_box(seg_heatmap_resized, obj))

        return bbs

    def predict_groundingdino(self, img: np.array, objects: List[str]) -> List[BoundingBox]:
        """
        Predict using the GroundingDINO model.
        """
        detections = self.model.predict_with_classes(img, objects, BOX_THRESHOLD, TEXT_THRESHOLD)

        bbs = []
        seen_classes = set()

        for i in range(detections.xyxy.shape[0]):
            class_id = detections.class_id[i]
            if class_id in seen_classes:
                continue
            seen_classes.add(class_id)

            try:
                bb = BoundingBox(
                    x=detections.xyxy[i][0],
                    y=detections.xyxy[i][1],
                    width=detections.xyxy[i][2] - detections.xyxy[i][0],
                    height=detections.xyxy[i][3] - detections.xyxy[i][1],
                    class_name=objects[class_id],
                )
            except IndexError:
                bb = BoundingBox(
                    x=detections.xyxy[i][0],
                    y=detections.xyxy[i][1],
                    width=detections.xyxy[i][2] - detections.xyxy[i][0],
                    height=detections.xyxy[i][3] - detections.xyxy[i][1],
                    class_name="Unknown",
                )
            bbs.append(bb)

        return bbs

    def predict_owlv2(self, img: np.array, objects: List[str]) -> List[BoundingBox]:
        """
        Predict using the Owlv2 model.
        """
        img_pil = Image.fromarray(img)
        objects_formatted = [[obj] for obj in objects]
        inputs = self.processor(text=objects_formatted, images=img_pil, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Post-process outputs to adjust bounding box sizes
        results = self.processor.post_process_object_detection(
            outputs=outputs,
            target_sizes=torch.Tensor([img_pil.size[::-1]])
        )

        # Extract unique detections
        bbs, seen_objects = [], {}
        for i, box in enumerate(results[0]["boxes"]):
            label = objects[results[0]["labels"][i]][0]
            score = results[0]["scores"][i].item()
            if label in seen_objects and seen_objects[label][1] >= score:
                continue
            seen_objects[label] = (box, score)

        for label, (box, score) in seen_objects.items():
            x1, y1, x2, y2 = box.cpu().numpy()
            bbs.append(
                BoundingBox(
                    x=x1,
                    y=y1,
                    width=x2 - x1,
                    height=y2 - y1,
                    class_name=label
                )
            )

        return bbs

    def predict_florence2(self, img: np.array, objects: List[str]) -> List[BoundingBox]:
        """
        Predict using the Florence-2 model.
        """
        img_pil = Image.fromarray(img)
        task_prompt = "<OPEN_VOCABULARY_DETECTION>"
        inputs = self.processor(
            text=f"{task_prompt}{objects[0]}",
            images=img_pil,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                early_stopping=True,
                num_beams=3,
            )
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            parsed_answer = self.processor.post_process_generation(
                generated_text,
                task=task_prompt,
                image_size=(img_pil.width, img_pil.height),
            )

        bbs = []
        parsed_answer = parsed_answer["<OPEN_VOCABULARY_DETECTION>"]
        for bbox, label in zip(parsed_answer["bboxes"], parsed_answer["bboxes_labels"]):
            x1, y1, x2, y2 = bbox
            bbs.append(
                BoundingBox(
                    x=x1,
                    y=y1,
                    width=x2 - x1,
                    height=y2 - y1,
                    class_name=label
                )
            )

        return bbs

    def resize_segmentation(self, heatmap, original_size):
        """
        Resize the segmentation heatmap to the original image size.
        """
        scale_factor_width = original_size[0] / heatmap.shape[1]
        scale_factor_height = original_size[1] / heatmap.shape[0]
        resized = cv2.resize(heatmap.cpu().numpy(), original_size)
        return resized * scale_factor_width, scale_factor_height

    def extract_bounding_box(self, heatmap, class_name):
        """
        Extract bounding box from heatmap using contours.
        """
        gray = (heatmap * 255).astype(np.uint8)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Select largest contour as the bounding box
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        return BoundingBox(x, y, w, h, class_name=class_name)

