"""
Fundus image preprocessing module for DR-EarlyXAI.
"""
import os
import cv2
import numpy as np
from PIL import Image
import argparse

class FundusPreprocessor:
    def __init__(self, clip_limit=2.0, grid_size=(8, 8)):
        self.clip_limit = clip_limit
        self.grid_size = grid_size
        self.clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.grid_size)

    def crop_retinal_circle(self, img: np.ndarray) -> np.ndarray:
        """
        Detect the circular fundus region and crop to its bounding box.
        
        Args:
            img: Input image as numpy array (RGB).
            
        Returns:
            Cropped image.
        """
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img
            
        # Threshold to find the bright circle against dark background
        _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return img
            
        # Find the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Compute bounding box of the minimum enclosing circle
        (x, y), radius = cv2.minEnclosingCircle(largest_contour)
        
        center = (int(x), int(y))
        r = int(radius)
        
        # Bounding box
        x1 = max(0, center[0] - r)
        y1 = max(0, center[1] - r)
        x2 = min(img.shape[1], center[0] + r)
        y2 = min(img.shape[0], center[1] + r)
        
        # Create a mask for the circle
        mask = np.zeros_like(gray)
        cv2.circle(mask, center, r, (255,), -1)
        
        # Apply mask
        if len(img.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
            
        img_masked = cv2.bitwise_and(img, mask)
        
        # Crop to bounding box (Black Border Removal)
        img_cropped = img_masked[y1:y2, x1:x2]
        
        return img_cropped

    def normalize_illumination(self, img: np.ndarray) -> np.ndarray:
        """
        Apply CLAHE on the L channel of LAB color space.
        
        Args:
            img: Input image as numpy array (RGB).
            
        Returns:
            Illumination normalized image.
        """
        if len(img.shape) != 3:
            return img
            
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        l_clahe = self.clahe.apply(l)
        
        lab_clahe = cv2.merge((l_clahe, a, b))
        img_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)
        
        return img_clahe

    def preprocess(self, image_path: str, target_size: int = 512) -> Image.Image:
        """
        Run the full preprocessing pipeline on a single image.
        
        Args:
            image_path: Path to the image file.
            target_size: Output image dimension.
            
        Returns:
            Preprocessed PIL Image.
        """
        # Load image
        img_pil = Image.open(image_path).convert('RGB')
        img = np.array(img_pil)
        
        # 1 & 2. Retinal Circle Cropping & Black Border Removal
        img = self.crop_retinal_circle(img)
        
        # 3. Illumination Normalization
        img = self.normalize_illumination(img)
        
        # 4. Resize
        img = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_AREA)
        
        return Image.fromarray(img)

    def preprocess_batch(self, input_dir: str, output_dir: str, target_size: int = 512, limit: int = None):
        """
        Process a batch of images.
        
        Args:
            input_dir: Input directory containing images.
            output_dir: Output directory to save preprocessed images.
            target_size: Target image dimension.
            limit: Maximum number of images to process.
            
        Returns:
            Number of successfully processed images.
        """
        os.makedirs(output_dir, exist_ok=True)
        image_extensions = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}
        
        count = 0
        for root, _, files in os.walk(input_dir):
            for file in files:
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    in_path = os.path.join(root, file)
                    out_path = os.path.join(output_dir, file)
                    
                    try:
                        img_processed = self.preprocess(in_path, target_size=target_size)
                        img_processed.save(out_path)
                        count += 1
                        
                        if limit and count >= limit:
                            break
                    except Exception as e:
                        print(f"Error processing {in_path}: {e}")
                        
            if limit and count >= limit:
                break
                
        return count

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Fundus Image Preprocessing")
    parser.add_argument("--input_dir", type=str, default="test_images", help="Input directory")
    parser.add_argument("--output_dir", type=str, default=os.path.join("outputs", "preprocessed"), help="Output directory")
    parser.add_argument("--limit", type=int, default=None, help="Max images to process")
    parser.add_argument("--target_size", type=int, default=512, help="Target resize dimension")
    args = parser.parse_args()
    
    preprocessor = FundusPreprocessor()
    print(f"Preprocessing images from {args.input_dir} to {args.output_dir}...")
    
    processed_count = preprocessor.preprocess_batch(
        args.input_dir, 
        args.output_dir, 
        target_size=args.target_size,
        limit=args.limit
    )
    
    print(f"Successfully preprocessed {processed_count} images.")
