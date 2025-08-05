#!/usr/bin/env python3
"""
Advanced Baggage and Person Detection System
Main script for running combined baggage and person detection on video files.

Author: Deepan K S
Email: deepanks01@gmail.com
"""

import cv2
import argparse
import os
from detector import CombinedBagPeopleDetector

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Baggage and Person Detection System')
    parser.add_argument('--bag_model', type=str, default='../models/best.pt',
                       help='Path to trained baggage detection model')
    parser.add_argument('--input_video', type=str, required=True,
                       help='Path to input video file')
    parser.add_argument('--output_video', type=str, default='output_detection.mp4',
                       help='Path to save output video')
    parser.add_argument('--bag_conf', type=float, default=0.5,
                       help='Confidence threshold for bag detection (0.0-1.0)')
    parser.add_argument('--people_conf', type=float, default=0.5,
                       help='Confidence threshold for people detection (0.0-1.0)')
    parser.add_argument('--save_video', action='store_true',
                       help='Save output video with detections')
    parser.add_argument('--display', action='store_true',
                       help='Display real-time detection (requires display)')
    
    return parser.parse_args()

def process_video(detector, video_path, output_path=None, bag_conf=0.5, 
                 people_conf=0.5, save_video=False, display=False):
    """
    Process video file with detection
    
    Args:
        detector: CombinedBagPeopleDetector instance
        video_path: Path to input video
        output_path: Path to save output video
        bag_conf: Confidence threshold for bags
        people_conf: Confidence threshold for people
        save_video: Whether to save output video
        display: Whether to display real-time results
    """
    
    # Load video
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"❌ Failed to open video: {video_path}")
        return False
    
    # Video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"📹 Video Info: {frame_width}x{frame_height} @ {fps:.2f}fps, {total_frames} frames")
    
    # Setup output video writer
    out = None
    if save_video and output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        print(f"💾 Output will be saved to: {output_path}")
    
    frame_count = 0
    detection_stats = {'bags': 0, 'people': 0, 'total_frames': 0}
    
    print("🎬 Starting video processing...")
    
    try:
        while True:
            success, frame = cap.read()
            if not success:
                break
            
            # Run combined detection
            detections = detector.detect_combined(frame, bag_conf=bag_conf, people_conf=people_conf)
            
            # Get detection summary for this frame
            summary = detector.get_detection_summary(detections)
            detection_stats['bags'] += summary['bags_detected']
            detection_stats['people'] += summary['people_detected']
            detection_stats['total_frames'] += 1
            
            # Draw detections on frame
            annotated_frame = detector.plot_detections(frame.copy(), detections)
            
            # Save frame if requested
            if save_video and out:
                out.write(annotated_frame)
            
            # Display frame if requested
            if display:
                cv2.imshow("Baggage & Person Detection", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\n⏹️ Processing stopped by user")
                    break
            
            frame_count += 1
            
            # Progress update
            if frame_count % 100 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"📊 Progress: {frame_count}/{total_frames} frames ({progress:.1f}%)")
    
    except KeyboardInterrupt:
        print("\n⏹️ Processing interrupted by user")
    
    finally:
        # Cleanup
        cap.release()
        if out:
            out.release()
        if display:
            cv2.destroyAllWindows()
    
    # Print final statistics
    print(f"\n✅ Processing complete!")
    print(f"🎥 Total frames processed: {frame_count}")
    print(f"🎒 Total bags detected: {detection_stats['bags']}")
    print(f"👥 Total people detected: {detection_stats['people']}")
    print(f"📈 Average detections per frame: {(detection_stats['bags'] + detection_stats['people']) / max(frame_count, 1):.2f}")
    
    if save_video and output_path:
        print(f"💾 Output saved at: {output_path}")
    
    return True

def main():
    """Main function"""
    args = parse_arguments()
    
    # Validate input file
    if not os.path.exists(args.input_video):
        print(f"❌ Error: Input video file not found: {args.input_video}")
        return
    
    # Validate model file
    if not os.path.exists(args.bag_model):
        print(f"❌ Error: Bag detection model not found: {args.bag_model}")
        return
    
    print("🚀 Initializing Advanced Baggage and Person Detection System")
    print(f"🎒 Bag Model: {args.bag_model}")
    print(f"📹 Input Video: {args.input_video}")
    print(f"🔍 Confidence Thresholds - Bags: {args.bag_conf}, People: {args.people_conf}")
    
    # Initialize detector
    try:
        detector = CombinedBagPeopleDetector(
            bag_model_path=args.bag_model,
            use_pretrained_people=True
        )
    except Exception as e:
        print(f"❌ Error initializing detector: {e}")
        return
    
    # Process video
    success = process_video(
        detector=detector,
        video_path=args.input_video,
        output_path=args.output_video if args.save_video else None,
        bag_conf=args.bag_conf,
        people_conf=args.people_conf,
        save_video=args.save_video,
        display=args.display
    )
    
    if success:
        print("\n🎉 Detection completed successfully!")
    else:
        print("\n❌ Detection failed!")

if __name__ == "__main__":
    main()