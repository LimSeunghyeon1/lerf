import argparse
import os
import sys

import numpy as np
import json
import torch
from PIL import Image

# sys.path.append(os.path.join('/home/tidy/Grounded-Segment-Anything', "GroundingDINO"))
# sys.path.append(os.path.join('/home/tidy/Grounded-Segment-Anything', "segment_anything"))


# Grounding DINO
import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap


# segment anything
from segment_anything import (
    sam_model_registry,
    sam_hq_model_registry,
    SamPredictor
)
import cv2
import numpy as np
import matplotlib.pyplot as plt




class GroundedSAM:
    def __init__(self, cfg):
        # cfg
        config_file = cfg.config  # change the path of the model config file
        grounded_checkpoint = cfg.grounded_checkpoint  # change the path of the model
        sam_version = cfg.sam_version
        sam_checkpoint = cfg.sam_checkpoint
        sam_hq_checkpoint = cfg.sam_hq_checkpoint
        use_sam_hq = cfg.use_sam_hq
        # image_path = cfg.input_image
        # text_prompt = cfg.text_prompt
        output_dir = cfg.output_dir
        self.box_threshold = cfg.box_threshold
        self.text_threshold = cfg.text_threshold
        self.device = cfg.device
        # load model
        self.model = GroundedSAM.load_model(config_file, grounded_checkpoint, device=self.device)
        if use_sam_hq:
            self.predictor = SamPredictor(sam_hq_model_registry[sam_version](checkpoint=sam_hq_checkpoint).to(self.device))
        else:
            self.predictor = SamPredictor(sam_model_registry[sam_version](checkpoint=sam_checkpoint).to(self.device))
            
    def inference_from_path(self, image_path, text_prompt, output_path, vis=False):
        assert type(image_path) == str
        assert type(text_prompt) == str
        image_pil, image = GroundedSAM.load_image(image_path)
        
        #run grounding dino
        boxes_filt, pred_phrases = GroundedSAM.get_grounding_output(self.model, image, text_prompt, self.box_threshold,\
            self.text_threshold, device=self.device)
        image = cv2.imread(image_path)

        if len(pred_phrases) == 0:
            return False   
        self.predictor.set_image(image)
        size = image_pil.size
        H, W = size[1], size[0]
        for i in range(boxes_filt.size(0)):
            boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
            boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
            boxes_filt[i][2:] += boxes_filt[i][:2]

        boxes_filt = boxes_filt.cpu()
        transformed_boxes = self.predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2]).to(self.device)
        
        masks, _, _ = self.predictor.predict_torch(
            point_coords = None,
            point_labels = None,
            boxes = transformed_boxes.to(self.device),
            multimask_output = False,
        )
        
        masks = masks.cpu().numpy()[0]
        
        '''
        if several masks are found, return all of them
        '''
        image_resize = cv2.resize(image, (masks[0].shape[1], masks[0].shape[0]))
        for i, mask in enumerate(masks):
            if i != 0:
                break
            mask = np.expand_dims(mask, axis=-1)
            masked_img = image_resize * mask
        
            if vis:
                # cv2.imwrite(f"masked_img_{pred_phrases[i]}.png", masked_img)
                assert output_path[-4:] == '.png' or output_path[-4:] == '.jpg'
                cv2.imwrite(output_path, masked_img)
            
        # return masks, masked_img
        return True
    
    @staticmethod
    def load_image(image_path):
        # load image
        image_pil = Image.open(image_path).convert("RGB")  # load image

        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image, _ = transform(image_pil, None)  # 3, h, w
        return image_pil, image

    @staticmethod
    def load_model(model_config_path, model_checkpoint_path, device):
        args = SLConfig.fromfile(model_config_path)
        args.device = device
        model = build_model(args)
        checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
        load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        print(load_res)
        _ = model.eval()
        return model

    @staticmethod
    def get_grounding_output(model, image, caption, box_threshold, text_threshold, with_logits=True, device="cpu"):
        caption = caption.lower()
        caption = caption.strip()
        if not caption.endswith("."):
            caption = caption + "."
        model = model.to(device)
        image = image.to(device)
        with torch.no_grad():
            outputs = model(image[None], captions=[caption])
        logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
        boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
        logits.shape[0]

        # filter output
        logits_filt = logits.clone()
        boxes_filt = boxes.clone()
        filt_mask = logits_filt.max(dim=1)[0] > box_threshold
        logits_filt = logits_filt[filt_mask]  # num_filt, 256
        boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
        logits_filt.shape[0]

        # get phrase
        tokenlizer = model.tokenizer
        tokenized = tokenlizer(caption)
        # build pred
        pred_phrases = []
        for logit, box in zip(logits_filt, boxes_filt):
            pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
            if with_logits:
                pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
            else:
                pred_phrases.append(pred_phrase)

        return boxes_filt, pred_phrases

    @staticmethod
    def show_mask(mask, ax, random_color=False):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            color = np.array([30/255, 144/255, 255/255, 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)


    @staticmethod
    def show_box(box, ax, label):
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))
        ax.text(x0, y0, label)


    @staticmethod
    def save_mask_data(output_dir, mask_list, box_list, label_list):
        value = 0  # 0 for background

        mask_img = torch.zeros(mask_list.shape[-2:])
        for idx, mask in enumerate(mask_list):
            mask_img[mask.cpu().numpy()[0] == True] = value + idx + 1
        plt.figure(figsize=(10, 10))
        plt.imshow(mask_img.numpy())
        plt.axis('off')
        plt.savefig(os.path.join(output_dir, 'mask.jpg'), bbox_inches="tight", dpi=300, pad_inches=0.0)

        json_data = [{
            'value': value,
            'label': 'background'
        }]
        for label, box in zip(label_list, box_list):
            value += 1
            name, logit = label.split('(')
            logit = logit[:-1] # the last is ')'
            json_data.append({
                'value': value,
                'label': name,
                'logit': float(logit),
                'box': box.numpy().tolist(),
            })
        with open(os.path.join(output_dir, 'mask.json'), 'w') as f:
            json.dump(json_data, f)
        
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Grounded-Segment-Anything Demo", add_help=True)
    parser.add_argument("--config", type=str, required=True, help="path to config file")
    parser.add_argument(
        "--grounded_checkpoint", type=str, required=True, help="path to checkpoint file"
    )
    parser.add_argument(
        "--sam_version", type=str, default="vit_h", required=False, help="SAM ViT version: vit_b / vit_l / vit_h"
    )
    parser.add_argument(
        "--sam_checkpoint", type=str, required=False, help="path to sam checkpoint file"
    )
    parser.add_argument(
        "--sam_hq_checkpoint", type=str, default=None, help="path to sam-hq checkpoint file"
    )
    parser.add_argument(
        "--use_sam_hq", action="store_true", help="using sam-hq for prediction"
    )
    parser.add_argument("--input_image", type=str, required=True, help="path to image file")
    parser.add_argument("--text_prompt", type=str, required=True, help="text prompt")
    parser.add_argument(
        "--output_dir", "-o", type=str, default="outputs", required=True, help="output directory"
    )

    parser.add_argument("--box_threshold", type=float, default=0.3, help="box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.25, help="text threshold")

    parser.add_argument("--device", type=str, default="cpu", help="running on cpu only!, default=False")
    args = parser.parse_args()
    
    base_path = '/home/tidy/lerf/lerf/data/images'
    out_path = '/home/tidy/lerf/lerf/data/images_masked'
    gsam = GroundedSAM(args)
    # do for all image data
    for dirpath, dirname, filenames in os.walk(base_path):
        for filename in filenames:
            if filename[-4:] != '.jpg': continue
            
            input_image = os.path.join(dirpath, filename)
            output_path = os.path.join(out_path, filename)
            isnotempty = gsam.inference_from_path(input_image, args.text_prompt, output_path, vis=True)
            if not isnotempty:
                print(filename)