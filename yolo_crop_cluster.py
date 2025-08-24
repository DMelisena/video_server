#!/usr/bin/env python3
"""
YOLO Detection Cropper and Clusterer
Runs YOLO inference, crops detected objects, and clusters similar detections
"""

import argparse
import cv2
import torch
import numpy as np
from pathlib import Path
import sys
import json
import shutil
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
import pickle

from ultralytics.nn.tasks import DetectionModel # Import the specific class
torch.serialization.add_safe_globals([DetectionModel])

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    print("Warning: ultralytics not found. Install with: pip install ultralytics")

try:
    import tensorflow as tf
    from tensorflow.keras.applications import ResNet50
    from tensorflow.keras.applications.resnet50 import preprocess_input
    from tensorflow.keras.preprocessing import image
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("Warning: tensorflow not found. Feature extraction will be limited.")

def setup_args():
    """Setup command line arguments"""
    parser = argparse.ArgumentParser(description='Crop YOLO detections and cluster similar objects')
    parser.add_argument('--model', '-m', type=str, required=True,
                        help='Path to YOLO .pt model file')
    parser.add_argument('--source', '-s', type=str, required=True,
                        help='Source: image file, video file, or directory')
    parser.add_argument('--output', '-o', type=str, default='detection_results',
                        help='Output directory (default: detection_results)')
    parser.add_argument('--conf', '-c', type=float, default=0.25,
                        help='Confidence threshold (default: 0.25)')
    parser.add_argument('--iou', type=float, default=0.45,
                        help='IoU threshold for NMS (default: 0.45)')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to run on: cpu, cuda, mps, or auto')
    parser.add_argument('--cluster-method', choices=['kmeans', 'dbscan'], default='kmeans',
                        help='Clustering method (default: kmeans)')
    parser.add_argument('--n-clusters', type=int, default=5,
                        help='Number of clusters for k-means (default: 5)')
    parser.add_argument('--min-crop-size', type=int, default=32,
                        help='Minimum crop size in pixels (default: 32)')
    parser.add_argument('--padding', type=int, default=10,
                        help='Padding around bounding box in pixels (default: 10)')
    parser.add_argument('--feature-method', choices=['basic', 'resnet'], default='basic',
                        help='Feature extraction method (default: basic)')
    parser.add_argument('--save-features', action='store_true',
                        help='Save extracted features for later analysis')
    parser.add_argument('--cleanup-clusters', action='store_true',
                        help='Remove all cluster folders after processing')
    
    return parser.parse_args()

def get_device(device_arg):
    """Determine the best device to use"""
    if device_arg == 'auto':
        if torch.cuda.is_available():
            return 'cuda'
        elif torch.backends.mps.is_available():
            return 'mps'
        else:
            return 'cpu'
    return device_arg

class DetectionProcessor:
    def __init__(self, args):
        self.args = args
        self.output_dir = Path(args.output)
        self.crops_dir = self.output_dir / 'crops'
        self.clusters_dir = self.output_dir / 'clusters'
        self.analysis_dir = self.output_dir / 'analysis'
        
        # Create directories
        for dir_path in [self.output_dir, self.crops_dir, self.clusters_dir, self.analysis_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        self.detections = []
        self.features = []
        self.crop_paths = []
        self.class_names = {}
        
        # Load feature extractor if using ResNet
        if args.feature_method == 'resnet' and TENSORFLOW_AVAILABLE:
            print("Loading ResNet50 feature extractor...")
            self.feature_extractor = ResNet50(weights='imagenet', include_top=False, pooling='avg')
        else:
            self.feature_extractor = None

    def extract_basic_features(self, crop):
        """Extract basic image features"""
        # Convert to different color spaces
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
        
        # Color statistics
        bgr_mean = np.mean(crop, axis=(0, 1))
        bgr_std = np.std(crop, axis=(0, 1))
        hsv_mean = np.mean(hsv, axis=(0, 1))
        hsv_std = np.std(hsv, axis=(0, 1))
        
        # Texture features (using simple gradient)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_mag = np.sqrt(grad_x**2 + grad_y**2)
        texture_mean = np.mean(gradient_mag)
        texture_std = np.std(gradient_mag)
        
        # Shape features
        height, width = crop.shape[:2]
        aspect_ratio = width / height if height > 0 else 1
        area = height * width
        
        # Combine all features
        features = np.concatenate([
            bgr_mean, bgr_std,
            hsv_mean, hsv_std,
            [texture_mean, texture_std, aspect_ratio, np.log(area + 1)]
        ])
        
        return features

    def extract_resnet_features(self, crop):
        """Extract features using ResNet50"""
        if self.feature_extractor is None:
            return self.extract_basic_features(crop)
        
        # Preprocess image for ResNet
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        crop_resized = cv2.resize(crop_rgb, (224, 224))
        crop_array = image.img_to_array(crop_resized)
        crop_array = np.expand_dims(crop_array, axis=0)
        crop_array = preprocess_input(crop_array)
        
        # Extract features
        features = self.feature_extractor.predict(crop_array, verbose=0)
        return features.flatten()

    def process_detections(self, model, device):
        """Run inference and process detections"""
        print(f"Processing source: {self.args.source}")
        
        # Run YOLO inference
        results = model(
            source=self.args.source,
            conf=self.args.conf,
            iou=self.args.iou,
            save=False,
            verbose=True,
            device=device
        )
        
        detection_count = 0
        
        for result in results:
            if result.boxes is None:
                continue
                
            # Get original image
            orig_img = result.orig_img
            if orig_img is None:
                continue
            
            # Get class names
            if hasattr(result, 'names'):
                self.class_names.update(result.names)
            
            # Process each detection
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)
            
            for i, (box, conf, class_id) in enumerate(zip(boxes, confidences, class_ids)):
                x1, y1, x2, y2 = map(int, box)
                
                # Add padding
                h, w = orig_img.shape[:2]
                x1 = max(0, x1 - self.args.padding)
                y1 = max(0, y1 - self.args.padding)
                x2 = min(w, x2 + self.args.padding)
                y2 = min(h, y2 + self.args.padding)
                
                # Check minimum size
                crop_w, crop_h = x2 - x1, y2 - y1
                if crop_w < self.args.min_crop_size or crop_h < self.args.min_crop_size:
                    continue
                
                # Crop detection
                crop = orig_img[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                
                # Save crop
                class_name = self.class_names.get(class_id, f"class_{class_id}")
                crop_filename = f"{class_name}_{detection_count:06d}_conf{conf:.2f}.jpg"
                crop_path = self.crops_dir / crop_filename
                cv2.imwrite(str(crop_path), crop)
                
                # Extract features
                if self.args.feature_method == 'resnet':
                    features = self.extract_resnet_features(crop)
                else:
                    features = self.extract_basic_features(crop)
                
                # Store detection info
                detection_info = {
                    'crop_path': str(crop_path),
                    'class_id': int(class_id),
                    'class_name': class_name,
                    'confidence': float(conf),
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'crop_size': [crop_w, crop_h],
                    'source_image': str(result.path) if hasattr(result, 'path') else 'unknown'
                }
                
                self.detections.append(detection_info)
                self.features.append(features)
                self.crop_paths.append(crop_path)
                detection_count += 1
                
                if detection_count % 100 == 0:
                    print(f"Processed {detection_count} detections...")
        
        print(f"Total detections processed: {detection_count}")
        return detection_count

    def cluster_detections(self):
        """Cluster the detections based on extracted features"""
        if len(self.features) == 0:
            print("No detections to cluster!")
            return
        
        print(f"Clustering {len(self.features)} detections...")
        
        # Prepare features
        features_array = np.array(self.features)
        
        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features_array)
        
        # Apply PCA if features are high-dimensional
        if features_scaled.shape[1] > 50:
            print("Applying PCA for dimensionality reduction...")
            pca = PCA(n_components=50, random_state=42)
            features_scaled = pca.fit_transform(features_scaled)
            
            # Save PCA info
            pca_info = {
                'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
                'n_components': 50
            }
            with open(self.analysis_dir / 'pca_info.json', 'w') as f:
                json.dump(pca_info, f, indent=2)
        
        # Perform clustering
        if self.args.cluster_method == 'kmeans':
            n_clusters = min(self.args.n_clusters, len(features_scaled))
            clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        else:  # dbscan
            clusterer = DBSCAN(eps=0.5, min_samples=3)
        
        cluster_labels = clusterer.fit_predict(features_scaled)
        
        # Save clustering results
        clustering_results = {
            'method': self.args.cluster_method,
            'n_detections': len(self.detections),
            'n_clusters': len(np.unique(cluster_labels)),
            'cluster_labels': cluster_labels.tolist()
        }
        
        with open(self.analysis_dir / 'clustering_results.json', 'w') as f:
            json.dump(clustering_results, f, indent=2)
        
        # Organize crops by cluster
        self.organize_by_clusters(cluster_labels)
        
        # Generate analysis
        self.generate_analysis(features_scaled, cluster_labels, scaler)
        
        return cluster_labels

    def organize_by_clusters(self, cluster_labels):
        """Organize cropped images by cluster"""
        print("Organizing crops by clusters...")
        
        # Create cluster directories
        cluster_dirs = {}
        for label in np.unique(cluster_labels):
            if label == -1:  # DBSCAN noise
                cluster_dir = self.clusters_dir / 'noise'
            else:
                cluster_dir = self.clusters_dir / f'cluster_{label:02d}'
            cluster_dir.mkdir(exist_ok=True)
            cluster_dirs[label] = cluster_dir
        
        # Copy crops to cluster directories
        cluster_stats = defaultdict(list)
        
        for detection, crop_path, cluster_label in zip(self.detections, self.crop_paths, cluster_labels):
            dest_path = cluster_dirs[cluster_label] / crop_path.name
            shutil.copy2(crop_path, dest_path)
            cluster_stats[cluster_label].append(detection)
        
        # Generate cluster summaries
        cluster_summary = {}
        for cluster_label, detections in cluster_stats.items():
            classes = [d['class_name'] for d in detections]
            confidences = [d['confidence'] for d in detections]
            
            cluster_summary[str(cluster_label)] = {
                'count': len(detections),
                'dominant_classes': dict(Counter(classes).most_common(5)),
                'avg_confidence': float(np.mean(confidences)),
                'confidence_std': float(np.std(confidences)),
                'size_stats': {
                    'avg_width': float(np.mean([d['crop_size'][0] for d in detections])),
                    'avg_height': float(np.mean([d['crop_size'][1] for d in detections]))
                }
            }
        
        with open(self.analysis_dir / 'cluster_summary.json', 'w') as f:
            json.dump(cluster_summary, f, indent=2)
        
        print(f"Organized into {len(cluster_dirs)} clusters")

    def cleanup_clusters(self):
        """Remove all cluster folders and their contents"""
        removed_count = 0
        print("Removing all cluster folders...")
        
        # Remove the main clusters directory if it exists
        if self.clusters_dir.exists():
            try:
                shutil.rmtree(self.clusters_dir)
                removed_count += 1
                print(f"✅ Main clusters directory removed: {self.clusters_dir}")
            except Exception as e:
                print(f"❌ Error removing main clusters directory: {e}")
        
        # Also remove any directories in the output directory that start with 'cluster'
        if self.output_dir.exists():
            for item in self.output_dir.iterdir():
                if item.is_dir() and item.name.startswith('cluster'):
                    try:
                        shutil.rmtree(item)
                        removed_count += 1
                        print(f"✅ Removed cluster directory: {item}")
                    except Exception as e:
                        print(f"❌ Error removing {item}: {e}")
        
        if removed_count == 0:
            print("No cluster folders found to remove")
        else:
            print(f"✅ Successfully removed {removed_count} cluster folder(s)")

    def generate_analysis(self, features_scaled, cluster_labels, scaler):
        """Generate analysis plots and reports"""
        print("Generating analysis...")
        
        # Cluster distribution plot
        plt.figure(figsize=(10, 6))
        unique_labels, counts = np.unique(cluster_labels, return_counts=True)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        
        bars = plt.bar(range(len(unique_labels)), counts, color=colors)
        plt.xlabel('Cluster')
        plt.ylabel('Number of Detections')
        plt.title('Distribution of Detections Across Clusters')
        plt.xticks(range(len(unique_labels)), 
                  [f'Noise' if label == -1 else f'Cluster {label}' for label in unique_labels],
                  rotation=45)
        
        # Add count labels on bars
        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    str(count), ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.analysis_dir / 'cluster_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Class distribution per cluster
        if len(self.class_names) > 1:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            cluster_class_matrix = defaultdict(lambda: defaultdict(int))
            for detection, cluster_label in zip(self.detections, cluster_labels):
                cluster_class_matrix[cluster_label][detection['class_name']] += 1
            
            # Create matrix for heatmap
            all_classes = list(set(d['class_name'] for d in self.detections))
            matrix_data = []
            cluster_names = []
            
            for cluster_label in sorted(cluster_class_matrix.keys()):
                if cluster_label == -1:
                    cluster_names.append('Noise')
                else:
                    cluster_names.append(f'Cluster {cluster_label}')
                
                row = [cluster_class_matrix[cluster_label][class_name] for class_name in all_classes]
                matrix_data.append(row)
            
            matrix_data = np.array(matrix_data)
            
            sns.heatmap(matrix_data, 
                       xticklabels=all_classes,
                       yticklabels=cluster_names,
                       annot=True, 
                       fmt='d', 
                       cmap='YlOrRd')
            plt.title('Class Distribution Across Clusters')
            plt.xlabel('Classes')
            plt.ylabel('Clusters')
            plt.tight_layout()
            plt.savefig(self.analysis_dir / 'class_cluster_heatmap.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 2D visualization using first two principal components
        if features_scaled.shape[1] >= 2:
            pca_2d = PCA(n_components=2, random_state=42)
            features_2d = pca_2d.fit_transform(features_scaled)
            
            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], 
                                c=cluster_labels, cmap='tab10', alpha=0.6)
            plt.xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]:.1%} variance)')
            plt.ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]:.1%} variance)')
            plt.title('2D Visualization of Clusters')
            plt.colorbar(scatter)
            plt.tight_layout()
            plt.savefig(self.analysis_dir / 'cluster_2d_visualization.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # Save features if requested
        if self.args.save_features:
            features_data = {
                'features': features_scaled.tolist(),
                'cluster_labels': cluster_labels.tolist(),
                'detection_info': self.detections,
                'scaler_params': {
                    'mean': scaler.mean_.tolist(),
                    'scale': scaler.scale_.tolist()
                }
            }
            
            with open(self.analysis_dir / 'features_data.pkl', 'wb') as f:
                pickle.dump(features_data, f)
        
        print(f"Analysis saved to: {self.analysis_dir}")

def main():
    """Main function"""
    args = setup_args()
    
    if not ULTRALYTICS_AVAILABLE:
        print("Error: ultralytics is required for this script")
        print("Install with: pip install ultralytics")
        sys.exit(1)
    
    # Check if model file exists
    if not Path(args.model).exists():
        print(f"Error: Model file not found: {args.model}")
        sys.exit(1)
    
    # Check if source exists
    source_path = Path(args.source)
    if not (source_path.exists() or args.source.isdigit()):
        print(f"Error: Source not found: {args.source}")
        sys.exit(1)
    
    try:
        # Initialize processor
        processor = DetectionProcessor(args)
        
        # Load YOLO model
        print(f"Loading YOLO model: {args.model}")
        device = get_device(args.device)
        model = YOLO(args.model)
        model.to(device)
        print(f"Using device: {device}")
        
        # Process detections
        detection_count = processor.process_detections(model, device)
        
        if detection_count == 0:
            print("No detections found!")
            sys.exit(1)
        
        # Cluster detections
        cluster_labels = processor.cluster_detections()
        
        # Cleanup clusters if requested
        if args.cleanup_clusters:
            processor.cleanup_clusters()
        
        print(f"\n🎉 Processing completed successfully!")
        print(f"📊 Results saved to: {processor.output_dir}")
        print(f"🖼️  Cropped detections: {detection_count}")
        print(f"🔄 Clusters created: {len(np.unique(cluster_labels))}")
        if not args.cleanup_clusters:
            print(f"📁 Organized crops in: {processor.clusters_dir}")
        else:
            print(f"🗑️  Cluster folders removed (as requested)")
        print(f"📈 Analysis reports in: {processor.analysis_dir}")
        
    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
