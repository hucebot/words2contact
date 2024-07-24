from geom_utils import Point
from transformers import AutoProcessor, CLIPSegForImageSegmentation, GroupViTModel, AutoModelForCausalLM
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torchvision.transforms import InterpolationMode

import numpy as np
import torch
import cv2
from PIL import Image, ImageDraw
import CLIP_Surgery.clip as clip_surgery


BICUBIC = InterpolationMode.BICUBIC


def get_max(heatmap):
    # get the maximum point from the heatmap
    point = np.unravel_index(np.argmax(heatmap), heatmap.shape)

    return Point(point[1], point[0])

# The Class tha contains everything needed to perform Open Vocabulary Segmentation


class Saygment:
    def __init__(self, vlm: str = "CLIPSeg", debug: bool = True, device: str = "cuda:0", cache_dir: str = "./models/VLMs/"):
        # vlm can either be CLIPSeg or GroundingDINO
        assert vlm in ["CLIPSeg", "CLIP_Surgery", "GroupViT", "Florence-2"]
        self.device = device
        self.vlm = vlm
        self.debug = debug
        self.cache_dir = cache_dir
        if self.vlm == "CLIPSeg":
            self.load_clipseg()
        elif self.vlm == "CLIP_Surgery":
            self.load_clipsurgery()
        elif self.vlm == "GroupViT":
            self.load_groupvit()
        elif self.vlm == "Florence-2":
            self.load_florence2()

    def load_clipseg(self):
        if self.debug:
            print("Loading CLIPSeg Model")
        self.processor = AutoProcessor.from_pretrained(
            "CIDAS/clipseg-rd64-refined", cache_dir=self.cache_dir)
        self.model = CLIPSegForImageSegmentation.from_pretrained(
            "CIDAS/clipseg-rd64-refined", cache_dir=self.cache_dir).to(self.device)
        if self.debug:
            print("Model loaded successfully")
        return

    def load_clipsurgery(self):
        if self.debug:
            print("Loading CLIP Surgery Model")
            self.model, _ = clip_surgery.load("CS-ViT-B/16", device=self.device, download_root=self.cache_dir)
            self.processor = Compose([Resize((512, 512), interpolation=BICUBIC), ToTensor(),
                                      Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])
        if self.debug:
            print("Model loaded successfully")
        return

    def load_groupvit(self):
        if self.debug:
            print("Loading GroupViT Model")
            self.model = GroupViTModel.from_pretrained("nvidia/groupvit-gcc-yfcc", cache_dir=self.cache_dir)
            self.processor = AutoProcessor.from_pretrained(
                "nvidia/groupvit-gcc-yfcc", cache_dir=self.cache_dir, device=self.device)
        if self.debug:
            print("Model loaded successfully")
        return

    def load_florence2(self):
        model_name = 'microsoft/Florence-2-large'
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True, chache_dir=self.cache_dir)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).eval()
        self.model.to(self.device)

    def predict_clipseg(self, img: np.array, objects) -> Point:

        img = Image.fromarray(img)
        print(img.size)

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

        for i in range(num_objects):
            seg_heatmap = torch.sigmoid(predictions[i][0])

            seg_heatmap = cv2.resize(seg_heatmap.cpu().numpy(), (img.size[0], img.size[1]))
            self.heatmap = seg_heatmap

            return get_max(self.heatmap), seg_heatmap

    def predict_clipsurgery(self, img: np.array, objects) -> Point:
        # first let's convert the image to a PIL Image
        cv2_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = Image.fromarray(img)
        image = self.processor(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            image_features = self.model.encode_image(image)
            image_features = image_features / image_features.norm(dim=1, keepdim=True)

            text_features = clip_surgery.encode_text_with_prompt_ensemble(self.model, objects, self.device)

            redundant_features = clip_surgery.encode_text_with_prompt_ensemble(self.model, [""], self.device)

            similarity = clip_surgery.clip_feature_surgery(image_features, text_features, redundant_features)
            similarity_map = clip_surgery.get_similarity_map(similarity[:, 1:, :], cv2_img.shape[:2])

            for b in range(similarity_map.shape[0]):
                for n in range(similarity_map.shape[-1]):
                    vis = (similarity_map[b, :, :, n].cpu().numpy())

                    self.heatmap = vis

                    return get_max(self.heatmap), self.heatmap

    def predict_groupvit(self, img: np.array, objects) -> Point:
        img = Image.fromarray(img)
        inputs = self.processor(text=objects, images=img, padding=True, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs, output_segmentation=True)

        logits = outputs.segmentation_logits

        logits = torch.nn.functional.interpolate(logits.detach().cpu(),
                                                 size=img.size[::-1],  # (height, width)
                                                 mode='bilinear',
                                                 align_corners=False)

        print(logits.shape)

        self.heatmap = logits.squeeze(0).squeeze(0).numpy()

        return get_max(self.heatmap), self.heatmap

    def predict_florence2(self, img: np.array, objects) -> Point:
        img = Image.fromarray(img)

        task_prompt = '<REFERRING_EXPRESSION_SEGMENTATION>'

        if len(objects) != 1:
            raise ValueError("Florence-2 model only supports single object detection")

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

        print(parsed_answer)
        prediction = parsed_answer['<REFERRING_EXPRESSION_SEGMENTATION>']
        # temporary black image that is the same size as the input image
        black = Image.new('RGB', (img.width, img.height), (0, 0, 0))
        mask = ImageDraw.Draw(black)
        scale = 1

        for polygons, label in zip(prediction['polygons'], prediction['labels']):
            color = 'white'
            fill_color = color

            for _polygon in polygons:
                _polygon = np.array(_polygon).reshape(-1, 2)
                if len(_polygon) < 3:
                    print('Invalid polygon:', _polygon)
                    continue

                _polygon = (_polygon * scale).reshape(-1).tolist()

                # Draw the polygon

                mask.polygon(_polygon, outline=color, fill=fill_color)

        # convert rgb to grayscale (0, 1)
        mask = mask._image.convert('L')
        mask = np.array(mask)
        self.heatmap = mask / 255

        return get_max(self.heatmap), self.heatmap

    def predict(self, img: np.array, objects) -> Point:
        if self.vlm == "CLIPSeg":
            return self.predict_clipseg(img, objects)
        elif self.vlm == "CLIP_Surgery":
            return self.predict_clipsurgery(img, objects)
        elif self.vlm == "GroupViT":
            return self.predict_groupvit(img, objects)
        elif self.vlm == "Florence-2":
            return self.predict_florence2(img, objects)
