from .geom_utils import Point
from transformers import (
    AutoProcessor,
    CLIPSegForImageSegmentation,
    GroupViTModel,
    AutoModelForCausalLM,
)
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, InterpolationMode

import numpy as np
import torch
import cv2
from PIL import Image, ImageDraw
from .CLIP_Surgery import clip as clip_surgery

BICUBIC = InterpolationMode.BICUBIC


def get_max(heatmap):
    """
    Retrieve the point with the maximum value from a heatmap.
    """
    point = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    return Point(point[1], point[0])


class Saygment:
    """
    A class to handle open-vocabulary segmentation using various vision-language models (VLMs).
    """

    def __init__(self, vlm: str = "CLIPSeg", debug: bool = True, device: str = "cuda:0", cache_dir: str = "./models/VLMs/"):
        """
        Initialize the Saygment class with the specified VLM.

        Args:
            vlm (str): The vision-language model to use. Options: "CLIPSeg", "CLIP_Surgery", "GroupViT", "Florence-2".
            debug (bool): Whether to enable debug prints.
            device (str): The device to use for computation.
            cache_dir (str): The directory to cache the models.
        """
        assert vlm in ["CLIPSeg", "CLIP_Surgery", "GroupViT", "Florence-2"], "Invalid VLM specified."
        self.device = device
        self.vlm = vlm
        self.debug = debug
        self.cache_dir = cache_dir

        # Load the appropriate model
        self._load_model()

    def _load_model(self):
        """
        Load the model and processor based on the specified VLM.
        """
        loaders = {
            "CLIPSeg": self._load_clipseg,
            "CLIP_Surgery": self._load_clipsurgery,
            "GroupViT": self._load_groupvit,
            "Florence-2": self._load_florence2,
        }
        loaders[self.vlm]()

    def _load_clipseg(self):
        if self.debug:
            print("Loading CLIPSeg model...")
        self.processor = AutoProcessor.from_pretrained(
            "CIDAS/clipseg-rd64-refined", cache_dir=self.cache_dir
        )
        self.model = CLIPSegForImageSegmentation.from_pretrained(
            "CIDAS/clipseg-rd64-refined", cache_dir=self.cache_dir
        ).to(self.device)
        if self.debug:
            print("CLIPSeg model loaded successfully.")

    def _load_clipsurgery(self):
        if self.debug:
            print("Loading CLIP Surgery model...")
        self.model, _ = clip_surgery.load("CS-ViT-B/16", device=self.device, download_root=self.cache_dir)
        self.processor = Compose([
            Resize((512, 512), interpolation=BICUBIC),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
        if self.debug:
            print("CLIP Surgery model loaded successfully.")

    def _load_groupvit(self):
        if self.debug:
            print("Loading GroupViT model...")
        self.model = GroupViTModel.from_pretrained("nvidia/groupvit-gcc-yfcc", cache_dir=self.cache_dir)
        self.processor = AutoProcessor.from_pretrained(
            "nvidia/groupvit-gcc-yfcc", cache_dir=self.cache_dir
        ).to(self.device)
        if self.debug:
            print("GroupViT model loaded successfully.")

    def _load_florence2(self):
        if self.debug:
            print("Loading Florence-2 model...")
        model_name = "microsoft/Florence-2-large"
        self.processor = AutoProcessor.from_pretrained(
            model_name, trust_remote_code=True, cache_dir=self.cache_dir
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, trust_remote_code=True
        ).eval().to(self.device)
        if self.debug:
            print("Florence-2 model loaded successfully.")

    def predict_clipseg(self, img: np.array, objects) -> Point:
        """
        Perform segmentation using the CLIPSeg model.
        """
        img_pil = Image.fromarray(img)
        num_objects = len(objects)
        inputs = self.processor(
            text=objects, images=[img_pil] * num_objects, padding=True, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        predictions = outputs.logits.detach()
        predictions = predictions.unsqueeze(1) if predictions.dim() == 3 else predictions.unsqueeze(0)

        for i in range(num_objects):
            seg_heatmap = torch.sigmoid(predictions[i][0])
            seg_heatmap_resized = cv2.resize(seg_heatmap.cpu().numpy(), img_pil.size)
            self.heatmap = seg_heatmap_resized
            return get_max(self.heatmap), self.heatmap

    def predict_clipsurgery(self, img: np.array, objects) -> Point:
        """
        Perform segmentation using the CLIP Surgery model.
        """
        cv2_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img_pil = Image.fromarray(img)
        image = self.processor(img_pil).unsqueeze(0).to(self.device)

        with torch.no_grad():
            image_features = self.model.encode_image(image)
            image_features = image_features / image_features.norm(dim=1, keepdim=True)
            text_features = clip_surgery.encode_text_with_prompt_ensemble(self.model, objects, self.device)
            redundant_features = clip_surgery.encode_text_with_prompt_ensemble(self.model, [""], self.device)

            similarity = clip_surgery.clip_feature_surgery(image_features, text_features, redundant_features)
            similarity_map = clip_surgery.get_similarity_map(similarity[:, 1:, :], cv2_img.shape[:2])

            for vis in similarity_map.cpu().numpy():
                self.heatmap = vis
                return get_max(self.heatmap), self.heatmap

    def predict_groupvit(self, img: np.array, objects) -> Point:
        """
        Perform segmentation using the GroupViT model.
        """
        img_pil = Image.fromarray(img)
        inputs = self.processor(text=objects, images=img_pil, padding=True, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs, output_segmentation=True)

        logits = torch.nn.functional.interpolate(
            outputs.segmentation_logits.detach().cpu(),
            size=img_pil.size[::-1],  # (height, width)
            mode="bilinear",
            align_corners=False,
        )

        self.heatmap = logits.squeeze().numpy()
        return get_max(self.heatmap), self.heatmap

    def predict_florence2(self, img: np.array, objects) -> Point:
        """
        Perform segmentation using the Florence-2 model.
        """
        if len(objects) != 1:
            raise ValueError("Florence-2 model only supports single object detection.")

        img_pil = Image.fromarray(img)
        task_prompt = "<REFERRING_EXPRESSION_SEGMENTATION>"
        prompt = f"{task_prompt}{objects[0]}"
        inputs = self.processor(text=prompt, images=img_pil, return_tensors="pt").to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                num_beams=3,
            )
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
            parsed_answer = self.processor.post_process_generation(
                generated_text, task=task_prompt, image_size=img_pil.size
            )

        black = Image.new("RGB", img_pil.size, (0, 0, 0))
        mask = ImageDraw.Draw(black)
        for polygons, label in zip(parsed_answer["polygons"], parsed_answer["labels"]):
            for polygon in polygons:
                mask.polygon(
                    (np.array(polygon).reshape(-1, 2) * 1).reshape(-1).tolist(), outline="white", fill="white"
                )

        mask = np.array(mask._image.convert("L")) / 255
        self.heatmap = mask
        return get_max(self.heatmap), self.heatmap

    def predict(self, img: np.array, objects) -> Point:
        """
        Perform segmentation using the specified VLM.
        """
        methods = {
            "CLIPSeg": self.predict_clipseg,
            "CLIP_Surgery": self.predict_clipsurgery,
            "GroupViT": self.predict_groupvit,
            "Florence-2": self.predict_florence2,
        }
        return methods[self.vlm](img, objects)
