from typing import List
import folder_paths
import os
import cv2
import numpy as np
import supervision as sv
import torch
from tqdm import tqdm
from ultralytics import YOLOWorld

from .utils.efficient_sam import load, inference_with_boxes
from .utils.video import generate_file_name, calculate_end_frame_index, create_directory

current_directory = os.path.dirname(os.path.abspath(__file__))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BOUNDING_BOX_ANNOTATOR = sv.BoundingBoxAnnotator()
MASK_ANNOTATOR = sv.MaskAnnotator()
LABEL_ANNOTATOR = sv.LabelAnnotator()

def process_categories(categories: str) -> List[str]:
    return [category.strip() for category in categories.split(',')]

def annotate_image(
    input_image: np.ndarray,
    detections: sv.Detections,
    categories: List[str],
    with_confidence: bool = False,
    thickness: int = 2,
    text_thickness: int = 2,
    text_scale: float = 1.0,
) -> np.ndarray:
    labels = [
        (
            f"{categories[class_id]}: {confidence:.3f}"
            if with_confidence
            else f"{categories[class_id]}"
        )
        for class_id, confidence in
        zip(detections.class_id, detections.confidence)
    ]
    output_image = input_image.copy()
    output_image = MASK_ANNOTATOR.annotate(output_image, detections)
    output_image = BOUNDING_BOX_ANNOTATOR.annotate(output_image, detections)
    output_image = LABEL_ANNOTATOR.annotate(output_image, detections, labels=labels)
    return output_image


class Yoloworld_ModelLoader_Zho:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "yolo_world_model": (["yolov8s-world", "yolov8m-world", "yolov8l-world"], ),
            }
        }

    RETURN_TYPES = ("YOLOWORLDMODEL",)
    RETURN_NAMES = ("yolo_world_model",)
    FUNCTION = "load_yolo_world_model"
    CATEGORY = "🔎YOLOWORLD_ESAM"
  
    def load_yolo_world_model(self, yolo_world_model):
        # Добавляем суффикс .pt к имени модели
        YOLO_WORLD_MODEL = YOLOWorld(model_id=f"{yolo_world_model}.pt")
        return (YOLO_WORLD_MODEL,)
        

class ESAM_ModelLoader_Zho:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "device": (["CUDA", "CPU"], ),
            }
        }

    RETURN_TYPES = ("ESAMMODEL",)
    RETURN_NAMES = ("esam_model",)
    FUNCTION = "load_esam_model"
    CATEGORY = "🔎YOLOWORLD_ESAM"
  
    def load_esam_model(self, device):
        model_name = "efficient_sam_s_gpu.jit" if device == "CUDA" else "efficient_sam_s_cpu.jit"
        model_path = os.path.join(current_directory, model_name)
        EFFICIENT_SAM_MODEL = torch.jit.load(model_path)
        return (EFFICIENT_SAM_MODEL,)

class Yoloworld_ESAM_Zho:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "yolo_world_model": ("YOLOWORLDMODEL",),
                "esam_model": ("ESAMMODEL",),
                "image": ("IMAGE",),
                "categories": ("STRING", {"default": "person, bicycle, car", "multiline": True}),
                "confidence_threshold": ("FLOAT", {"default": 0.3, "min": 0, "max": 1, "step": 0.01}),
                "iou_threshold": ("FLOAT", {"default": 0.5, "min": 0, "max": 1, "step": 0.01}),
                "box_thickness": ("INT", {"default": 2, "min": 1, "max": 5}),
                "text_thickness": ("INT", {"default": 1, "min": 1, "max": 5}),
                "text_scale": ("FLOAT", {"default": 0.5, "min": 0, "max": 1, "step": 0.01}),
                "with_confidence": ("BOOLEAN", {"default": True}),
                "with_class_agnostic_nms": ("BOOLEAN", {"default": False}),
                "with_segmentation": ("BOOLEAN", {"default": True}),
                "mask_combined": ("BOOLEAN", {"default": True}),
                "mask_extracted": ("BOOLEAN", {"default": False}),
                "mask_extracted_index": ("INT", {"default": 0, "min": 0, "max": 1000}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", )
    RETURN_NAMES = ("annotated_image", "masks")
    FUNCTION = "yoloworld_esam_image"
    CATEGORY = "🔎YOLOWORLD_ESAM"
                       
    def yoloworld_esam_image(self, yolo_world_model, esam_model, image, categories, 
                           confidence_threshold, iou_threshold, box_thickness,
                           text_thickness, text_scale, with_segmentation,
                           mask_combined, with_confidence, with_class_agnostic_nms,
                           mask_extracted, mask_extracted_index):
        # Преобразование входного тензора в numpy array
        image_np = image.cpu().numpy() * 255
        image_np = image_np.astype(np.uint8)
        
        categories = process_categories(categories)
        processed_images = []
        processed_masks = []
        
        # Обработка каждого изображения в батче
        for batch_idx in range(image_np.shape[0]):
            img = image_np[batch_idx]
            img_rgb = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            # Выполнение предсказания YOLO-World
            yolo_world_model.set_classes(categories)
            results = yolo_world_model.predict(img_rgb, conf=confidence_threshold)
            
            # Конвертация результатов в формат supervision
            detections = sv.Detections.from_ultralytics(results[0])
            detections = detections.with_nms(
                class_agnostic=with_class_agnostic_nms,
                threshold=iou_threshold
            )
            
            # Генерация масок с помощью EfficientSAM
            if with_segmentation and detections.xyxy.shape[0] > 0:
                detections.mask = inference_with_boxes(
                    image=img_rgb,
                    xyxy=detections.xyxy,
                    model=esam_model,
                    device=DEVICE
                )
                
                # Обработка масок
                if mask_combined:
                    combined_mask = np.zeros(img.shape[:2], dtype=np.uint8)
                    for mask in detections.mask:
                        combined_mask = np.logical_or(combined_mask, mask)
                    processed_masks.append(torch.from_numpy(combined_mask.astype(np.float32)))
                elif mask_extracted:
                    idx = min(mask_extracted_index, len(detections.mask)-1)
                    processed_masks.append(torch.from_numpy(detections.mask[idx].astype(np.float32)))
                else:
                    processed_masks.extend([torch.from_numpy(m.astype(np.float32)) for m in detections.mask])
            
            # Аннотирование изображения
            annotated_image = annotate_image(
                input_image=img_rgb,
                detections=detections,
                categories=categories,
                with_confidence=with_confidence,
                thickness=box_thickness,
                text_thickness=text_thickness,
                text_scale=text_scale,
            )
            
            # Конвертация обратно в тензор
            annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
            processed_images.append(torch.from_numpy(annotated_image).float() / 255.0)
        
        # Сборка выходных тензоров
        images_out = torch.stack(processed_images, dim=0)
        masks_out = torch.stack(processed_masks, dim=0) if processed_masks else torch.zeros(0)
        
        return (images_out, masks_out)

NODE_CLASS_MAPPINGS = {
    "Yoloworld_ModelLoader_Zho": Yoloworld_ModelLoader_Zho,
    "ESAM_ModelLoader_Zho": ESAM_ModelLoader_Zho,
    "Yoloworld_ESAM_Zho": Yoloworld_ESAM_Zho,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Yoloworld_ModelLoader_Zho": "🔎YOLO-World Model Loader",
    "ESAM_ModelLoader_Zho": "🔎EfficientSAM Model Loader",
    "Yoloworld_ESAM_Zho": "🔎YOLO-World + EfficientSAM",
}
