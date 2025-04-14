from groundingdino.util.train import load_model, load_image, train_image, annotate, train_batch, GroundingDINODataset, collate_fn
import cv2
import os
import json
import csv
import torch
from collections import defaultdict
import torch.optim as optim
from torch.utils.data import DataLoader

# Model
model = load_model("groundingdino/config/GroundingDINO_SwinT_OGC.py", "weights/groundingdino_swint_ogc.pth")

# Dataset paths
images_files=sorted(os.listdir("multimodal-data/images"))
ann_file="multimodal-data/annotation/annotation.csv"

def draw_box_with_label(image, output_path, coordinates, label, color=(0, 0, 255), thickness=2, font_scale=0.5):
    """
    Draw a box and a label on an image using OpenCV.

    Parameters:
    - image (numpyarray): input image.
    - output_path (str): Path to save the image with the box and label.
    - coordinates (tuple): A tuple (x1, y1, x2, y2) indicating the top-left and bottom-right corners of the box.
    - label (str): The label text to be drawn next to the box.
    - color (tuple, optional): Color of the box and label in BGR format. Default is red (0, 0, 255).
    - thickness (int, optional): Thickness of the box's border. Default is 2 pixels.
    - font_scale (float, optional): Font scale for the label. Default is 0.5.
    """
    
    # Draw the rectangle
    cv2.rectangle(image, (coordinates[0], coordinates[1]), (coordinates[2], coordinates[3]), color, thickness)
    
    # Define a position for the label (just above the top-left corner of the rectangle)
    label_position = (coordinates[0], coordinates[1]-10)
    
    # Draw the label
    cv2.putText(image, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)
    
    # Save the modified image
    cv2.imwrite(output_path, image)

def draw_box_with_label(image, output_path, coordinates, label, color=(0, 0, 255), thickness=2, font_scale=0.5):
    """
    Draw a box and a label on an image using OpenCV.

    Parameters:
    - image (str):  Input image.
    - output_path (str): Path to save the image with the box and label.
    - coordinates (tuple): A tuple (x1, y1, x2, y2) indicating the top-left and bottom-right corners of the box.
    - label (str): The label text to be drawn next to the box.
    - color (tuple, optional): Color of the box and label in BGR format. Default is red (0, 0, 255).
    - thickness (int, optional): Thickness of the box's border. Default is 2 pixels.
    - font_scale (float, optional): Font scale for the label. Default is 0.5.
    """
    
    # Draw the rectangle
    cv2.rectangle(image, (coordinates[0], coordinates[1]), (coordinates[2], coordinates[3]), color, thickness)
    
    # Define a position for the label (just above the top-left corner of the rectangle)
    label_position = (coordinates[0], coordinates[1]-10)
    
    # Draw the label
    cv2.putText(image, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)
    
    # Save the modified image
    cv2.imwrite(output_path, image)

def read_dataset(ann_file):
    ann_Dict= defaultdict(lambda: defaultdict(list))
    with open(ann_file) as file_obj:
        ann_reader= csv.DictReader(file_obj)  
        # Iterate over each row in the csv file
        # using reader object
        for row in ann_reader:
            #print(row)
            img_n=os.path.join("multimodal-data/images",row['image_name'])
            x1=int(row['bbox_x'])
            y1=int(row['bbox_y'])
            x2=x1+int(row['bbox_width'])
            y2=y1+int(row['bbox_height'])
            label=row['label_name']
            ann_Dict[img_n]['boxes'].append([x1,y1,x2,y2])
            ann_Dict[img_n]['captions'].append(label)
    return ann_Dict


def train(model, ann_file, epochs=1, save_path='weights/model_weights', save_epoch=50, batch_size=8):
    # Read Dataset
    ann_Dict = read_dataset(ann_file)
    
    # Create dataset and dataloader
    dataset = GroundingDINODataset(ann_Dict)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size,
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True
    )
    
    # Add optimizer with weight decay for regularization
    optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)
    
    # Add learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=10, verbose=True
    )
    
    # Early stopping parameters
    best_loss = float('inf')
    patience = 15
    patience_counter = 0
    
    # Ensure the model is in training mode
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        batch_count = 0
        
        for batch_idx, batch in enumerate(dataloader):
            # Zero the gradients
            optimizer.zero_grad()
            
            # Get the batch data
            image_sources = batch['image_sources']
            images = batch['images']
            caption_objects_batch = batch['caption_objects_batch']
            box_targets_batch = batch['box_targets_batch']
            
            # Call the batch training function
            loss = train_batch(
                model=model,
                image_sources=image_sources,
                images=images,
                caption_objects_batch=caption_objects_batch,
                box_targets_batch=box_targets_batch,
                batch_size=len(image_sources)
            )
            
            # Backpropagate and optimize
            loss.backward()
            optimizer.step()
            
            # Accumulate the loss
            total_loss += loss.item()
            batch_count += 1
            
            print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx+1}/{len(dataloader)}, Loss: {loss.item()}")

        avg_loss = total_loss / batch_count
        print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss}")
        
        # Learning rate scheduling
        scheduler.step(avg_loss)
        
        # Early stopping check
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            # Save best model
            checkpoint = {
                'model': model.state_dict(),
                'epoch': epoch,
                'optimizer': optimizer.state_dict(),
                'loss': avg_loss,
                'best_loss': best_loss
            }
            torch.save(checkpoint, f"{save_path}_best.pth")
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break
            
        if (epoch % save_epoch) == 0:
            checkpoint = {
                'model': model.state_dict(),
                'epoch': epoch,
                'optimizer': optimizer.state_dict(),
                'loss': avg_loss
            }
            torch.save(checkpoint, f"{save_path}{epoch}.pth")


if __name__=="__main__":
    train(
        model=model,
        ann_file=ann_file,
        epochs=101,
        save_path='weights/fine_tuning_weights/',
        save_epoch=10,
        batch_size=8  # Added batch size parameter
    )
