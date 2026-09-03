"""
Image quality assessment module for DR-EarlyXAI.
"""
import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import argparse

class QualityChecker:
    def __init__(self, blur_threshold=100.0, dark_threshold=40.0, bright_threshold=220.0, overall_threshold=50.0):
        self.blur_threshold = blur_threshold
        self.dark_threshold = dark_threshold
        self.bright_threshold = bright_threshold
        self.overall_threshold = overall_threshold

    def check_single(self, image_path: str) -> dict:
        """
        Evaluate image quality for a single image.
        
        Args:
            image_path: Path to the image file.
            
        Returns:
            Dictionary containing quality metrics and acceptance flag.
        """
        try:
            img_pil = Image.open(image_path).convert('RGB')
            img_np = np.array(img_pil)
            img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            
            # 1. Blur Detection (Laplacian variance)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            blur_score = float(laplacian_var)
            
            # 2. Exposure Assessment (Mean brightness)
            mean_brightness = float(np.mean(gray))
            
            # 3. Noise Estimation (Median Absolute Deviation of Laplacian)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            median = np.median(laplacian)
            mad = np.median(np.abs(laplacian - median))
            noise_score = float(mad)
            
            # 4. Overall Quality Score
            blur_norm = min(1.0, blur_score / 500.0)
            brightness_norm = 1.0 - abs(mean_brightness - 128) / 128.0
            noise_norm = max(0.0, 1.0 - (noise_score / 50.0))
            
            overall_quality = (blur_norm * 0.5 + brightness_norm * 0.3 + noise_norm * 0.2) * 100
            
            messages = []
            is_acceptable = True
            
            if blur_score < self.blur_threshold:
                messages.append("Blurry")
                is_acceptable = False
            
            if mean_brightness < self.dark_threshold:
                messages.append("Underexposed")
                is_acceptable = False
            elif mean_brightness > self.bright_threshold:
                messages.append("Overexposed")
                is_acceptable = False
                
            if overall_quality < self.overall_threshold:
                messages.append("Poor overall quality")
                is_acceptable = False
                
            if not messages:
                messages.append("Good")
                
            return {
                "file": os.path.basename(image_path),
                "blur_score": blur_score,
                "mean_brightness": mean_brightness,
                "noise_score": noise_score,
                "overall_quality": overall_quality,
                "is_acceptable": is_acceptable,
                "details": ", ".join(messages)
            }
            
        except Exception as e:
            return {
                "file": os.path.basename(image_path),
                "blur_score": 0.0,
                "mean_brightness": 0.0,
                "noise_score": 0.0,
                "overall_quality": 0.0,
                "is_acceptable": False,
                "details": f"Error: {str(e)}"
            }

    def check_batch(self, image_dir: str, output_csv: str = None, limit: int = None) -> pd.DataFrame:
        """
        Process a batch of images in a directory.
        
        Args:
            image_dir: Directory containing images.
            output_csv: Path to save the results.
            limit: Maximum number of images to process.
            
        Returns:
            DataFrame with quality assessment results.
        """
        results = []
        image_extensions = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}
        
        count = 0
        for root, _, files in os.walk(image_dir):
            for file in files:
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    path = os.path.join(root, file)
                    res = self.check_single(path)
                    results.append(res)
                    count += 1
                    if limit and count >= limit:
                        break
            if limit and count >= limit:
                break
                
        df = pd.DataFrame(results)
        
        if output_csv:
            os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
            df.to_csv(output_csv, index=False)
            
        return df

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Image Quality Check")
    parser.add_argument("--image_dir", type=str, default="test_images", help="Input directory")
    parser.add_argument("--limit", type=int, default=None, help="Max images to process")
    args = parser.parse_args()
    
    # All output files go under outputs/ directory
    output_csv = os.path.abspath(os.path.join("outputs", "quality", "quality_report.csv"))
    
    checker = QualityChecker()
    print(f"Running quality check on {args.image_dir}...")
    df = checker.check_batch(args.image_dir, output_csv=output_csv, limit=args.limit)
    
    print(f"Processed {len(df)} images.")
    if len(df) > 0:
        print(f"Acceptable: {df['is_acceptable'].sum()}/{len(df)}")
        print(f"Mean Quality Score: {df['overall_quality'].mean():.2f}")
        print(f"Results saved to {output_csv}")
