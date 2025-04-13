import json
import csv
import os
from pathlib import Path
import argparse

def convert_coco_to_csv(coco_json_path, csv_output_path):
    """
    Convert COCO format JSON annotations to a simplified CSV format with the following columns:
    label_name,bbox_x,bbox_y,bbox_width,bbox_height,image_name,image_width,image_height
    
    Args:
        coco_json_path: Path to the COCO JSON annotation file
        csv_output_path: Path to save the output CSV file
    """
    # Load COCO JSON file
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)

    # Extract categories, images, and annotations
    categories = {category['id']: category['name'] for category in coco_data['categories']}
    
    # Create a mapping from image_id to image details
    image_map = {}
    for image in coco_data['images']:
        image_id = image['id']
        image_map[image_id] = {
            'file_name': image['file_name'],
            'width': image['width'],
            'height': image['height']
        }

    # Prepare CSV rows
    csv_rows = []
    
    # Process annotations
    for annotation in coco_data['annotations']:
        image_id = annotation['image_id']
        image_info = image_map[image_id]
        
        # Get category name
        category_id = annotation['category_id']
        label_name = categories.get(category_id, f"unknown_{category_id}")
        
        # Get bbox values - COCO format is [x, y, width, height]
        bbox = annotation['bbox']
        bbox_x = int(bbox[0])
        bbox_y = int(bbox[1])
        bbox_width = int(bbox[2])
        bbox_height = int(bbox[3])
        
        # Create row for CSV
        row = {
            'label_name': label_name,
            'bbox_x': bbox_x,
            'bbox_y': bbox_y,
            'bbox_width': bbox_width,
            'bbox_height': bbox_height,
            'image_name': image_info['file_name'],
            'image_width': image_info['width'],
            'image_height': image_info['height']
        }
        csv_rows.append(row)

    # Write to CSV file
    with open(csv_output_path, 'w', newline='') as csvfile:
        fieldnames = ['label_name', 'bbox_x', 'bbox_y', 'bbox_width', 'bbox_height', 
                     'image_name', 'image_width', 'image_height']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    
    print(f"Conversion complete! Converted {len(csv_rows)} annotations from {len(image_map)} images.")
    print(f"Output saved to: {csv_output_path}")


def main():
    parser = argparse.ArgumentParser(description='Convert COCO JSON annotations to CSV format for Grounding DINO fine-tuning')
    parser.add_argument('--coco_json', type=str, default='multimodal-data/annotation/_annotations.coco.json',
                       help='Path to COCO JSON annotation file')
    parser.add_argument('--output_csv', type=str, default='multimodal-data/annotation/annotation.csv',
                       help='Path to save the output CSV file')

    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    
    convert_coco_to_csv(args.coco_json, args.output_csv)


if __name__ == "__main__":
    main()
