import cv2
import numpy as np
from ultralytics import YOLO
import os

class CombinedBagPeopleDetector:
    """
    A combined detector class for baggage and person detection using YOLOv8.
    
    This class integrates a custom-trained baggage detection model with a 
    pre-trained COCO model for person detection, providing comprehensive 
    object detection capabilities.
    """
    
    def __init__(self, bag_model_path, use_pretrained_people=True):
        """
        Initialize combined detector
        
        Args:
            bag_model_path (str): Path to your trained bag detection model
            use_pretrained_people (bool): Use pre-trained COCO model for people detection
        """
        print("🔧 Loading models...")
        
        # Load your trained bag detection model
        self.bag_model = YOLO(bag_model_path)
        
        # Load pre-trained COCO model for people detection
        if use_pretrained_people:
            self.people_model = YOLO('yolov8x.pt')  # Pre-trained COCO model
        else:
            self.people_model = YOLO(bag_model_path)  # Fallback to same model
        
        # Bag class names (11 classes)
        self.bag_classes = [
            'Backpack', 'Handbag', 'Suitcase', 'Trash bag', 'Paper bag', 
            'Hand bag', 'Gunny bag', 'Carry bag', 'Big handbag', 'Box bag', 'Kattapai'
        ]
        
        print("✅ Both models loaded successfully!")
    
    def detect_combined(self, frame, bag_conf=0.5, people_conf=0.5):
        """
        Run detection with both models and combine results
        
        Args:
            frame: Input video frame
            bag_conf (float): Confidence threshold for bag detection
            people_conf (float): Confidence threshold for people detection
            
        Returns:
            list: List of combined detections
        """
        
        # Detect bags using your trained model
        bag_results = self.bag_model(frame, conf=bag_conf, verbose=False)
        bag_detections = self._process_bag_results(bag_results)
        
        # Detect people using pre-trained COCO model
        people_results = self.people_model(frame, conf=people_conf, verbose=False)
        people_detections = self._process_people_results(people_results)
        
        # Combine all detections
        all_detections = bag_detections + people_detections
        
        # Apply Non-Maximum Suppression to remove overlapping detections
        final_detections = self._apply_nms(all_detections)
        
        return final_detections
    
    def _process_bag_results(self, results):
        """Process bag detection results"""
        detections = []
        
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    bbox = box.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
                    
                    if class_id < len(self.bag_classes):
                        detections.append({
                            'bbox': bbox,
                            'confidence': confidence,
                            'class_id': class_id,
                            'class_name': self.bag_classes[class_id],
                            'type': 'bag'
                        })
        
        return detections
    
    def _process_people_results(self, results):
        """Process people detection results"""
        detections = []
        
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    bbox = box.xyxy[0].cpu().numpy()
                    
                    # Only keep person class (class 0 in COCO)
                    if class_id == 0:
                        detections.append({
                            'bbox': bbox,
                            'confidence': confidence,
                            'class_id': 11,  # Assign as class 11 for consistency
                            'class_name': 'person',
                            'type': 'person'
                        })
        
        return detections
    
    def _apply_nms(self, detections, iou_threshold=0.5):
        """Apply Non-Maximum Suppression to remove overlapping detections"""
        if not detections:
            return []
        
        # Extract boxes and scores
        boxes = np.array([det['bbox'] for det in detections])
        scores = np.array([det['confidence'] for det in detections])
        
        # Convert to format required by OpenCV NMS
        boxes_list = boxes.tolist()
        scores_list = scores.tolist()
        
        # Apply NMS
        indices = cv2.dnn.NMSBoxes(
            boxes_list, 
            scores_list, 
            score_threshold=0.3, 
            nms_threshold=iou_threshold
        )
        
        if len(indices) > 0:
            indices = indices.flatten()
            return [detections[i] for i in indices]
        else:
            return detections
    
    def plot_detections(self, frame, detections):
        """
        Draw bounding boxes and labels on frame
        
        Args:
            frame: Input frame
            detections: List of detection dictionaries
            
        Returns:
            frame: Annotated frame with bounding boxes and labels
        """
        
        # Colors for different types
        colors = {
            'bag': (0, 255, 0),      # Green for bags
            'person': (0, 0, 255)    # Red for people
        }
        
        for det in detections:
            x1, y1, x2, y2 = map(int, det['bbox'])
            confidence = det['confidence']
            class_name = det['class_name']
            obj_type = det['type']
            
            # Choose color
            color = colors.get(obj_type, (255, 255, 255))
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            cv2.putText(frame, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return frame
    
    def get_detection_summary(self, detections):
        """
        Get summary statistics of detections
        
        Args:
            detections: List of detection dictionaries
            
        Returns:
            dict: Summary statistics
        """
        bag_count = sum(1 for det in detections if det['type'] == 'bag')
        person_count = sum(1 for det in detections if det['type'] == 'person')
        
        bag_classes_detected = set(det['class_name'] for det in detections if det['type'] == 'bag')
        
        return {
            'total_objects': len(detections),
            'bags_detected': bag_count,
            'people_detected': person_count,
            'unique_bag_types': list(bag_classes_detected)
        }