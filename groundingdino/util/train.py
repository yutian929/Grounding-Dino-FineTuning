from typing import Tuple, List, Dict

import cv2
import numpy as np
import supervision as sv
import torch
from PIL import Image
from torchvision.ops import box_convert
from torchvision.ops import box_iou, generalized_box_iou ,sigmoid_focal_loss
import torch.nn.functional as F
import bisect
from torch.utils.data import Dataset, DataLoader

import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util.misc import clean_state_dict
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import get_phrases_from_posmap

# ----------------------------------------------------------------------------------------------------------------------
# OLD API
# ----------------------------------------------------------------------------------------------------------------------


def focal_loss(logits, targets, alpha=0.25, gamma=2, eps=1e-7):
    logits = logits.clamp(min=-50, max=50)  # Clamp logits
    return sigmoid_focal_loss(logits,targets,reduction="mean")


def preprocess_caption(caption: str) -> str:
    result = caption.lower().strip()
    if result.endswith("."):
        return result
    return result + "."


def load_model(model_config_path: str, model_checkpoint_path: str, device: str = "cuda"):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    model.eval()
    return model


def load_image(image_path: str) -> Tuple[np.array, torch.Tensor]:
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image_source = Image.open(image_path).convert("RGB")
    image = np.asarray(image_source)
    image_transformed, _ = transform(image_source, None)
    return image, image_transformed


def train_image(model,
                image_source,
                image: torch.Tensor,
                caption_objects: list,
                box_target: list,
                device: str = "cuda"):

    def get_object_positions(tokenized, caption_objects):
        positions_dict = {}
        for obj_name in caption_objects:
            obj_token = tokenizer(obj_name + ".")['input_ids']
            start_pos = next((i for i, _ in enumerate(tokenized['input_ids']) if 
                             tokenized['input_ids'][i:i+len(obj_token)-2] == obj_token[1:-1]), None)
            if start_pos is not None:
                positions_dict[obj_name] = [start_pos, start_pos + len(obj_token) - 2]
        return positions_dict

    # Tokenization and object position extraction
    tokenizer = model.tokenizer
    caption = preprocess_caption(caption=".".join(set(caption_objects)))
    #print(f"Caption is {caption}")
    tokenized = tokenizer(caption)
    object_positions = get_object_positions(tokenized, caption_objects)

    # Move model and input to the device
    model = model.to(device)
    image = image.to(device)

    outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"][0]
    boxes = outputs["pred_boxes"][0]

    # Bounding box losses
    h, w, _ = image_source.shape
    boxes = boxes * torch.Tensor([w, h, w, h]).to(device)
    box_predicted = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy")
    box_target = torch.tensor(box_target).to(device)
    ious = generalized_box_iou(box_target, box_predicted)
    maxvals, maxidx = torch.max(ious, dim=1)
    selected_preds = box_predicted.gather(0, maxidx.unsqueeze(-1).repeat(1, box_predicted.size(1)))
    regression_loss = F.smooth_l1_loss(box_target, selected_preds)
    iou_loss = 1.0 - maxvals.mean()
    reg_loss = iou_loss + regression_loss

    # Logit losses
    selected_logits = logits.gather(0, maxidx.unsqueeze(-1).repeat(1, logits.size(1)))
    targets_logits_list = []
    for obj_name, logit in zip(caption_objects, selected_logits):
        target = torch.zeros_like(logit).to(device)
        start, end = object_positions[obj_name]
        target[start:end] = 1.0
        targets_logits_list.append(target)

   
    targets_logits = torch.stack(targets_logits_list, dim=0)
    cls_loss = focal_loss(selected_logits, targets_logits)
    #print(f"Output keys are {outputs.keys()}")
    print(f"Regression and Classification loss are {reg_loss} and {cls_loss}")

    # Total loss
    delta_factor=0.01
    total_loss = cls_loss + delta_factor*reg_loss  

    return total_loss


def train_batch(model,
                image_sources,
                images: torch.Tensor,
                caption_objects_batch: List[list],
                box_targets_batch: List[list],
                batch_size: int = 8,
                device: str = "cuda"):
    """
    Train model with a batch of images and annotations
    
    Args:
        model: The GroundingDINO model
        image_sources: List of source images (numpy arrays)
        images: Batch of images as tensor [batch_size, C, H, W]
        caption_objects_batch: List of lists containing caption objects for each image
        box_targets_batch: List of lists containing target boxes for each image
        batch_size: Batch size
        device: Device to run the model on
    
    Returns:
        Total loss for the batch
    """
    # Move model and input to device
    model = model.to(device)
    images = images.to(device)
    
    tokenizer = model.tokenizer
    total_loss = 0.0
    
    # Process each image in the batch
    all_outputs = None
    all_captions = []
    
    # Prepare captions for all images in batch
    for caption_objects in caption_objects_batch:
        caption = preprocess_caption(caption=".".join(set(caption_objects)))
        all_captions.append(caption)
    
    # Forward pass for the whole batch
    outputs = model(images, captions=all_captions)
    
    # Process each image in the batch
    for idx in range(batch_size):
        # Skip if we've run out of actual data
        if idx >= len(caption_objects_batch):
            break
        
        caption_objects = caption_objects_batch[idx]
        box_target = box_targets_batch[idx]
        image_source = image_sources[idx]
        
        # Process outputs for this sample
        logits = outputs["pred_logits"][idx]
        boxes = outputs["pred_boxes"][idx]
        
        # Get object positions for this caption
        caption = all_captions[idx]
        tokenized = tokenizer(caption)
        
        def get_object_positions(tokenized, caption_objects):
            positions_dict = {}
            for obj_name in caption_objects:
                obj_token = tokenizer(obj_name + ".")['input_ids']
                start_pos = next((i for i, _ in enumerate(tokenized['input_ids']) if 
                                tokenized['input_ids'][i:i+len(obj_token)-2] == obj_token[1:-1]), None)
                if start_pos is not None:
                    positions_dict[obj_name] = [start_pos, start_pos + len(obj_token) - 2]
            return positions_dict
        
        object_positions = get_object_positions(tokenized, caption_objects)
        
        # Bounding box losses
        h, w, _ = image_source.shape
        boxes = boxes * torch.Tensor([w, h, w, h]).to(device)
        box_predicted = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy")
        box_target_tensor = torch.tensor(box_target).to(device)
        ious = generalized_box_iou(box_target_tensor, box_predicted)
        maxvals, maxidx = torch.max(ious, dim=1)
        selected_preds = box_predicted.gather(0, maxidx.unsqueeze(-1).repeat(1, box_predicted.size(1)))
        regression_loss = F.smooth_l1_loss(box_target_tensor, selected_preds)
        iou_loss = 1.0 - maxvals.mean()
        reg_loss = iou_loss + regression_loss

        # Logit losses
        selected_logits = logits.gather(0, maxidx.unsqueeze(-1).repeat(1, logits.size(1)))
        targets_logits_list = []
        for obj_name, logit in zip(caption_objects, selected_logits):
            target = torch.zeros_like(logit).to(device)
            if obj_name in object_positions:  # Handle case where object position not found
                start, end = object_positions[obj_name]
                target[start:end] = 1.0
            targets_logits_list.append(target)

        if targets_logits_list:  # Check if list is not empty
            targets_logits = torch.stack(targets_logits_list, dim=0)
            cls_loss = focal_loss(selected_logits, targets_logits)
        else:
            cls_loss = torch.tensor(0.0, device=device)

        # Total loss for this sample
        delta_factor = 0.01
        sample_loss = cls_loss + delta_factor * reg_loss
        total_loss += sample_loss
        
        print(f"Sample {idx} - Regression loss: {reg_loss.item()}, Classification loss: {cls_loss.item()}")
    
    # Average loss over actual batch size
    return total_loss / len(caption_objects_batch)


class GroundingDINODataset(Dataset):
    """Dataset for Grounding DINO training"""
    
    def __init__(self, ann_dict, transform=None):
        self.image_paths = list(ann_dict.keys())
        self.annotations = ann_dict
        self.transform = transform if transform else self._default_transform()
        
    def _default_transform(self):
        return T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image_source = Image.open(image_path).convert("RGB")
        image = np.asarray(image_source)
        transformed_image, _ = self.transform(image_source, None)
        
        boxes = self.annotations[image_path]['boxes']
        captions = self.annotations[image_path]['captions']
        
        return {
            'image_source': image,
            'image': transformed_image,
            'caption_objects': captions,
            'box_target': boxes,
            'image_path': image_path
        }


def collate_fn(batch):
    """Collate function for DataLoader"""
    image_sources = [item['image_source'] for item in batch]
    images = torch.stack([item['image'] for item in batch])
    caption_objects_batch = [item['caption_objects'] for item in batch]
    box_targets_batch = [item['box_target'] for item in batch] 
    image_paths = [item['image_path'] for item in batch]
    
    return {
        'image_sources': image_sources,
        'images': images,
        'caption_objects_batch': caption_objects_batch,
        'box_targets_batch': box_targets_batch,
        'image_paths': image_paths
    }


def annotate(image_source: np.ndarray, boxes: torch.Tensor, logits: torch.Tensor, phrases: List[str]) -> np.ndarray:
    h, w, _ = image_source.shape
    boxes = boxes * torch.Tensor([w, h, w, h])
    xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
    detections = sv.Detections(xyxy=xyxy)

    labels = [
        f"{phrase} {logit:.2f}"
        for phrase, logit
        in zip(phrases, logits)
    ]

    box_annotator = sv.BoxAnnotator()
    annotated_frame = cv2.cvtColor(image_source, cv2.COLOR_RGB2BGR)
    annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
    return annotated_frame


# ----------------------------------------------------------------------------------------------------------------------
# NEW API
# ----------------------------------------------------------------------------------------------------------------------


class Model:

    def __init__(
        self,
        model_config_path: str,
        model_checkpoint_path: str,
        device: str = "cuda"
    ):
        self.model = load_model(
            model_config_path=model_config_path,
            model_checkpoint_path=model_checkpoint_path,
            device=device
        ).to(device)
        self.device = device

    def predict_with_caption(
        self,
        image: np.ndarray,
        caption: str,
        box_threshold: float = 0.35,
        text_threshold: float = 0.25
    ) -> Tuple[sv.Detections, List[str]]:
        """
        import cv2

        image = cv2.imread(IMAGE_PATH)

        model = Model(model_config_path=CONFIG_PATH, model_checkpoint_path=WEIGHTS_PATH)
        detections, labels = model.predict_with_caption(
            image=image,
            caption=caption,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD
        )

        import supervision as sv

        box_annotator = sv.BoxAnnotator()
        annotated_image = box_annotator.annotate(scene=image, detections=detections, labels=labels)
        """
        processed_image = Model.preprocess_image(image_bgr=image).to(self.device)
        boxes, logits, phrases = predict(
            model=self.model,
            image=processed_image,
            caption=caption,
            box_threshold=box_threshold,
            text_threshold=text_threshold, 
            device=self.device)
        source_h, source_w, _ = image.shape
        detections = Model.post_process_result(
            source_h=source_h,
            source_w=source_w,
            boxes=boxes,
            logits=logits)
        return detections, phrases

    def predict_with_classes(
        self,
        image: np.ndarray,
        classes: List[str],
        box_threshold: float,
        text_threshold: float
    ) -> sv.Detections:
        """
        import cv2

        image = cv2.imread(IMAGE_PATH)

        model = Model(model_config_path=CONFIG_PATH, model_checkpoint_path=WEIGHTS_PATH)
        detections = model.predict_with_classes(
            image=image,
            classes=CLASSES,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD
        )


        import supervision as sv

        box_annotator = sv.BoxAnnotator()
        annotated_image = box_annotator.annotate(scene=image, detections=detections)
        """
        caption = ". ".join(classes)
        processed_image = Model.preprocess_image(image_bgr=image).to(self.device)
        boxes, logits, phrases = predict(
            model=self.model,
            image=processed_image,
            caption=caption,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device=self.device)
        source_h, source_w, _ = image.shape
        detections = Model.post_process_result(
            source_h=source_h,
            source_w=source_w,
            boxes=boxes,
            logits=logits)
        class_id = Model.phrases2classes(phrases=phrases, classes=classes)
        detections.class_id = class_id
        return detections

    @staticmethod
    def preprocess_image(image_bgr: np.ndarray) -> torch.Tensor:
        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image_pillow = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        image_transformed, _ = transform(image_pillow, None)
        return image_transformed

    @staticmethod
    def post_process_result(
            source_h: int,
            source_w: int,
            boxes: torch.Tensor,
            logits: torch.Tensor
    ) -> sv.Detections:
        boxes = boxes * torch.Tensor([source_w, source_h, source_w, source_h])
        xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
        confidence = logits.numpy()
        return sv.Detections(xyxy=xyxy, confidence=confidence)

    @staticmethod
    def phrases2classes(phrases: List[str], classes: List[str]) -> np.ndarray:
        class_ids = []
        for phrase in phrases:
            for class_ in classes:
                if class_ in phrase:
                    class_ids.append(classes.index(class_))
                    break
            else:
                class_ids.append(None)
        return np.array(class_ids)
