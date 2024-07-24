from geom_utils import Point, BoundingBox
import matplotlib
from transformers.utils.constants import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD
from groundingdino.util.inference import Model
from transformers import AutoProcessor, CLIPSegForImageSegmentation, AutoModelForCausalLM, Owlv2ForObjectDetection
import os
import time
import warnings
import numpy as np
import torch
import cv2
from PIL import Image
from typing import List
import matplotlib.pyplot as plt
import sys
from os.path import dirname

matplotlib.rcParams['interactive'] == True
warnings.filterwarnings("ignore")

WEIGHTS_NAME = "groundingdino_swint_ogc.pth"
CONFIG_PATH = "scripts/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
IMAGE_PATH = "dataset/images/53e2f90e25d2d125215b2c7f1612f54959ef89b779c33a40e341e72d62b45760.png"
BOX_TRESHOLD = 0.2
TEXT_TRESHOLD = 0.2
TEXT_PROMPT = ["windex"]

WEIGHTS_PATH = os.path.join("models/VLMs/GroundingDINO", WEIGHTS_NAME)


# The Class tha contains everything needed to perform Open Vocabulary Object Detection
class Yello:
    def __init__(self, vlm: str = "CLIPSeg", debug: bool = False, device: str = "cuda"):
        assert vlm in ["CLIPSeg", "GroundingDINO", "Owlv2", "Florence-2"]
        self.device = device
        self.vlm = vlm
        self.debug = debug
        self.cache_dir = "./models/VLMs/"
        if self.vlm == "GroundingDINO":
            self.load_groundingdino()
        elif self.vlm == "Florence-2":
            self.load_florence2()
        elif self.vlm == "CLIPSeg":
            self.load_clipseg()
        elif self.vlm == "Owlv2":
            self.load_owlv2()

    def load_clipseg(self):
        if self.debug:
            print("Loading CLIPSeg Model")
        self.processor = AutoProcessor.from_pretrained(
            "CIDAS/clipseg-rd64-refined", cache_dir="./models/VLMs/")
        self.model = CLIPSegForImageSegmentation.from_pretrained(
            "CIDAS/clipseg-rd64-refined", cache_dir="./models/VLMs/").to(self.device)
        if self.debug:
            print("Model loaded successfully")
        return

    def load_groundingdino(self):
        if self.debug:
            print("Loading GroundingDINO Model")
        self.model = Model(CONFIG_PATH, WEIGHTS_PATH, device=self.device)
        if self.debug:
            print("Model loaded successfully")
        return

    def load_owlv2(self):
        if self.debug:
            print("Loading Owlv2 Model")

        self.processor = AutoProcessor.from_pretrained("google/owlv2-base-patch16")
        self.model = Owlv2ForObjectDetection.from_pretrained(
            "google/owlv2-base-patch16", cache_dir="./models/VLMs").to("cuda")

        if self.debug:
            print("Model loaded successfully")
        return

    def load_florence2(self):
        model_name = 'microsoft/Florence-2-large'
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True, chache_dir=self.cache_dir)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).eval()
        self.model.to(self.device)

    def predict(self, img: np.array, objects: List[str]) -> List[BoundingBox]:
        if self.vlm == "CLIPSeg":
            return self.predict_clipseg(img, objects)
        elif self.vlm == "GroundingDINO":
            return self.predict_groundingdino(img, objects)
        elif self.vlm == "Owlv2":
            return self.predict_owlv2(img, objects)
        elif self.vlm == "Florence-2":
            return self.predict_florence2(img, objects)

    def predict_clipseg(self, img: np.array, objects) -> List[BoundingBox]:
        # first let's convert the image to a PIL Image
        img = Image.fromarray(img)
        num_objects = len(objects)
        inputs = self.processor(text=objects, images=[img] * num_objects,
                                padding=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)

        # check if single object predction and handle predictions accordingly
        if outputs.logits.dim() == 3:
            predictions = outputs.logits.detach().unsqueeze(1)
        else:
            predictions = outputs.logits.detach().unsqueeze(0)

        bbs = []

        for i in range(num_objects):
            seg_heatmap = torch.sigmoid(predictions[i][0])
            scale_factor_height = img.size[1] / seg_heatmap.shape[1]
            scale_factor_width = img.size[0] / seg_heatmap.shape[0]

            # let's get the k max for each heatmap
            k = 1
            max_values = np.argpartition(seg_heatmap.cpu().numpy(), -10, axis=None)[-10:]
            idxs = np.unravel_index(max_values, seg_heatmap.shape)
            max_positions = [Point(x, y) for x, y in zip(idxs[1], idxs[0])]

            # now let's get the bounding box of the object
            seg_heatmap_cp = seg_heatmap.cpu().numpy()
            seg_heatmap_cp = (seg_heatmap_cp - seg_heatmap_cp.min()) / \
                (seg_heatmap_cp.max() - seg_heatmap_cp.min()) * 255
            gray = seg_heatmap_cp.astype(np.uint8)
            thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = cnts[0] if len(cnts) == 2 else cnts[1]
            bb = []
            max_points_cnt = []
            for c in cnts:
                x, y, w, h = cv2.boundingRect(c)

                # count the number of max points in the bounding box
                count = 0
                for point in max_positions:
                    if x <= point.x <= x+w and y <= point.y <= y+h:
                        count += 1
                max_points_cnt.append(count)
            # now let's use the max_points cnt to get the bounding box with the most max points
            winner = max_points_cnt.index(max(max_points_cnt))
            x, y, w, h = cv2.boundingRect(cnts[winner])
            bb = BoundingBox(x, y, w, h, class_name=objects[i])
            bb.scale(scale_factor_width, scale_factor_height)

            bbs.append(bb)

        return bbs

    def predict_groundingdino(self, img: np.array, objects):

        detections = self.model.predict_with_classes(img, objects, BOX_TRESHOLD, TEXT_TRESHOLD)

        bbs = []
        seen_classes = []
        for i in range(detections.xyxy.shape[0]):

            if detections.class_id[i] in seen_classes:
                continue
            else:
                seen_classes.append(detections.class_id[i])

            try:
                bb = BoundingBox(detections.xyxy[i][0], detections.xyxy[i][1], detections.xyxy[i][2] -
                                 detections.xyxy[i][0], detections.xyxy[i][3] - detections.xyxy[i][1], class_name=objects[detections.class_id[i]])
            except:
                bb = BoundingBox(detections.xyxy[i][0], detections.xyxy[i][1], detections.xyxy[i][2] -
                                 detections.xyxy[i][0], detections.xyxy[i][3] - detections.xyxy[i][1], class_name="Unknown")
            bbs.append(bb)

        return bbs

    def predict_owlv2(self, img: np.array, objects):
        img = Image.fromarray(img)
        objects = [[obj] for obj in objects]
        inputs = self.processor(text=objects, images=img, return_tensors="pt").to("cuda")

        with torch.no_grad():
            outputs = self.model(**inputs)

        # we need to do some preperation to scale the bounding boxes

        def get_preprocessed_image(pixel_values):
            pixel_values = pixel_values.squeeze().numpy()
            unnormalized_image = (pixel_values * np.array(OPENAI_CLIP_STD)
                                  [:, None, None]) + np.array(OPENAI_CLIP_MEAN)[:, None, None]
            unnormalized_image = (unnormalized_image * 255).astype(np.uint8)
            unnormalized_image = np.moveaxis(unnormalized_image, 0, -1)
            unnormalized_image = Image.fromarray(unnormalized_image)
            return unnormalized_image

        unnormalized_image = get_preprocessed_image(inputs.pixel_values.to("cpu"))
        target_sizes = torch.Tensor([unnormalized_image.size[::-1]])

        results = self.processor.post_process_object_detection(
            outputs=outputs, target_sizes=target_sizes, threshold=0.35)

        unnormalized_image = np.array(unnormalized_image)
        mask = unnormalized_image[:, :, :] != [127, 127, 127]
        indices = np.argwhere(mask)
        arr = unnormalized_image[indices[:, 0].min():indices[:, 0].max(), indices[:, 1].min():indices[:, 1].max()]
        scale_factor_width = img.size[0] / arr.shape[1]
        scale_factor_height = img.size[1] / arr.shape[0]
        arr = cv2.resize(arr, (img.size[0], img.size[1]))

        bbs = []
        already_seen = []
        scores_seen = []
        boxes = []

        for i in range(len(results[0]["boxes"])):
            box = results[0]["boxes"][i].to("cpu").numpy()
            score = results[0]["scores"][i]
            label = objects[results[0]["labels"][i]][0]
            if label in already_seen:
                idx = already_seen.index(label)
                if scores_seen[idx] < score:
                    scores_seen[idx] = score
                    boxes[idx] = box

            else:
                already_seen.append(label)
                scores_seen.append(score)
                boxes.append(box)

        for i in range(len(boxes)):
            box = boxes[i]
            w = box[2] - box[0]
            h = box[1] - box[3]
            x = box[0]
            y = box[1] - h
            bb = BoundingBox(x, y, w, h, class_name=already_seen[i])
            bb.scale(scale_factor_width, scale_factor_height)
            bbs.append(bb)
        return bbs

    def predict_florence2(self, img: np.array, objects):
        img = Image.fromarray(img)
        task_prompt = '<OPEN_VOCABULARY_DETECTION>'
        prompt = f"{task_prompt}{objects[0]}"
        inputs = self.processor(text=prompt, images=img, return_tensors="pt").to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                early_stopping=False,
                do_sample=False,
                num_beams=3,
            )

            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
            parsed_answer = self.processor.post_process_generation(
                generated_text,
                task=task_prompt,
                image_size=(img.width, img.height)
            )

        bbs = []
        labels = []
        parsed_answer = parsed_answer['<OPEN_VOCABULARY_DETECTION>']
        for bbox, label in zip(parsed_answer["bboxes"], parsed_answer["bboxes_labels"]):
            if label in labels:
                continue
            x1, y1, x2, y2 = bbox
            bb = BoundingBox(x1, y1, x2 - x1, y2 - y1, class_name=label)
            bbs.append(bb)

        return bbs
