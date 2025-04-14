import json
import csv
import os
from pathlib import Path
import argparse
import cv2

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
    
    # rewrite with indent=4
    with open(coco_json_path, 'w') as f:
        json.dump(coco_data, f, indent=4)

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


def check(ann_file, images_dir="multimodal-data/images"):
    with open(ann_file) as file_obj:
        ann_reader= csv.DictReader(file_obj)  
        # Iterate over each row in the csv file
        # using reader object
        # read the first image
        row_0 = next(ann_reader)
        img_n_0 = row_0['image_name']
        x1_list = [int(row_0['bbox_x'])]
        y1_list = [int(row_0['bbox_y'])]
        x2_list = [int(row_0['bbox_x'])+int(row_0['bbox_width'])]
        y2_list = [int(row_0['bbox_y'])+int(row_0['bbox_height'])]
        label_list = [row_0['label_name']]
        for row in ann_reader:
            #print(row)
            img_n=row['image_name']
            if img_n != img_n_0:
                break
            x1_list.append(int(row['bbox_x']))
            y1_list.append(int(row['bbox_y']))
            x2_list.append(x1_list[-1] + int(row['bbox_width']))
            y2_list.append(y1_list[-1] + int(row['bbox_height']))
            label_list.append(row['label_name'])
        # draw the first image
        img_0_path = os.path.join(images_dir, img_n_0)
        img_0 = cv2.imread(img_0_path)
        for i in range(len(label_list)):
            x1 = int(x1_list[i])
            y1 = int(y1_list[i])
            x2 = int(x2_list[i])
            y2 = int(y2_list[i])
            label = label_list[i]
            cv2.rectangle(img_0, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img_0, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.imwrite(f"check_{img_n_0}", img_0)
    
def compare(ann_file, example_ann_file):
    check(ann_file)
    check(example_ann_file, images_dir="multimodal-data-example/images")
    print(f"Comparison between {ann_file} and {example_ann_file} completed.")

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

    # Compare with example annotations
    example_ann_file = 'multimodal-data-example/annotation/annotation.csv'
    compare(args.output_csv, example_ann_file)


if __name__ == "__main__":
    main()
    
