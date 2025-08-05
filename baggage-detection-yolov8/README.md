# Advanced Baggage and Person Detection System Using YOLOv8

## Overview
This project implements a comprehensive baggage and person detection system using YOLOv8 architecture. The system combines a custom-trained model for 11 distinct bag classes with a pre-trained COCO model for person detection, providing robust real-time object detection capabilities for security and surveillance applications.

## Features
- **Dual-Model Architecture**: Custom baggage detection model + Pre-trained person detection
- **11 Bag Classes**: Backpack, Handbag, Suitcase, Trash bag, Paper bag, Hand bag, Gunny bag, Carry bag, Big handbag, Box bag, Kattapai
- **Real-time Processing**: Video stream processing with real-time detection
- **Non-Maximum Suppression**: Optimized detection accuracy by removing overlapping predictions
- **Automated Visualization**: Bounding box and label generation with confidence scores

## Model Architecture
- **Baggage Detection**: Custom YOLOv8l model fine-tuned on baggage dataset
- **Person Detection**: Pre-trained YOLOv8x COCO model (Class 0 - Person)
- **Training Details**: 150 epochs, batch size 8, image size 640x640
- **Data Augmentation**: Mosaic, Mixup, Copy-Paste, HSV adjustments

## Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended)

### Setup
```bash
# Clone the repository
git clone https://github.com/deepan2003/baggage-detection-yolov8.git
cd baggage-detection-yolov8

# Install dependencies
pip install -r requirements.txt

# Download pre-trained COCO model (will be downloaded automatically on first run)
```

## Usage

### Basic Detection
```python
from src.detector import CombinedBagPeopleDetector

# Initialize detector
detector = CombinedBagPeopleDetector(
    bag_model_path='models/best.pt',
    use_pretrained_people=True
)

# Run detection on video
python src/main.py
```

### Configuration
Update the following paths in `src/main.py`:
- `bag_model_path`: Path to your trained model
- `video_path`: Input video file
- `output_path`: Output video save location

## Dataset Structure
```
baggage_dataset/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
└── data.yaml
```

## Model Performance
- **Baggage Detection**: Custom trained model with 11 classes
- **Person Detection**: Pre-trained COCO model (70%+ accuracy)
- **Combined Processing**: NMS applied for optimized results
- **Confidence Thresholds**: Configurable (default: 0.5 for both models)

## Class Labels
```
0: Backpack       6: Gunny bag
1: Handbag        7: Carry bag  
2: Suitcase       8: Big handbag
3: Trash bag      9: Box bag
4: Paper bag     10: Kattapai
5: Hand bag      11: Person (from COCO)
```

## Results
The system successfully detects and classifies various types of baggage while simultaneously identifying persons in real-time video streams. The dual-model approach ensures high accuracy for both baggage and person detection tasks.

## File Structure
```
baggage-detection-yolov8/
├── README.md
├── requirements.txt
├── models/
│   ├── best.pt
│   └── model_info.txt
├── src/
│   ├── main.py
│   ├── detector.py
│   └── utils.py
├── data/
│   ├── sample_images/
│   └── class_names.txt
└── results/
    ├── sample_output.mp4
    └── detection_examples/
```

## Contributing
Feel free to open issues and submit pull requests for improvements.

## License
This project is licensed under the MIT License.

## Contact
**Deepan K S**
- Email: deepanks01@gmail.com
- LinkedIn: [linkedin.com/in/deepan-ks](https://linkedin.com/in/deepan-ks)
- GitHub: [github.com/deepan2003](https://github.com/deepan2003)

## Acknowledgments
- Ultralytics YOLOv8 for the base architecture
- COCO dataset for pre-trained person detection model