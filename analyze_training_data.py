import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from collections import defaultdict
import random

def analyze_training_data():
    """Analyze the training data to understand patterns and potential issues."""
    
    data_dir = "Data"
    letters = [chr(i) for i in range(ord('A'), ord('Z')+1)]
    
    # Statistics for each letter
    letter_stats = {}
    
    print("="*60)
    print("TRAINING DATA ANALYSIS")
    print("="*60)
    
    for letter in letters:
        letter_dir = os.path.join(data_dir, letter)
        if not os.path.exists(letter_dir):
            print(f"❌ No data folder for letter {letter}")
            continue
            
        # Get all image files
        image_files = [f for f in os.listdir(letter_dir) if f.endswith('.jpg')]
        
        if not image_files:
            print(f"❌ No images found for letter {letter}")
            continue
        
        print(f"\n📁 Letter {letter}: {len(image_files)} images")
        
        # Analyze a sample of images
        sample_size = min(5, len(image_files))
        sample_files = random.sample(image_files, sample_size)
        
        # Image analysis
        image_sizes = []
        aspect_ratios = []
        brightness_values = []
        contrast_values = []
        
        for img_file in sample_files:
            img_path = os.path.join(letter_dir, img_file)
            img = cv2.imread(img_path)
            
            if img is not None:
                # Basic image properties
                height, width = img.shape[:2]
                image_sizes.append((width, height))
                aspect_ratios.append(width / height)
                
                # Convert to grayscale for analysis
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
                # Brightness (mean pixel value)
                brightness = np.mean(gray)
                brightness_values.append(brightness)
                
                # Contrast (standard deviation)
                contrast = np.std(gray)
                contrast_values.append(contrast)
        
        # Calculate statistics
        if image_sizes:
            avg_width = np.mean([size[0] for size in image_sizes])
            avg_height = np.mean([size[1] for size in image_sizes])
            avg_aspect = np.mean(aspect_ratios)
            avg_brightness = np.mean(brightness_values)
            avg_contrast = np.mean(contrast_values)
            
            letter_stats[letter] = {
                'count': len(image_files),
                'avg_width': avg_width,
                'avg_height': avg_height,
                'avg_aspect_ratio': avg_aspect,
                'avg_brightness': avg_brightness,
                'avg_contrast': avg_contrast,
                'sample_files': sample_files
            }
            
            print(f"   📊 Average size: {avg_width:.1f}x{avg_height:.1f}")
            print(f"   📐 Average aspect ratio: {avg_aspect:.2f}")
            print(f"   💡 Average brightness: {avg_brightness:.1f}")
            print(f"   🎨 Average contrast: {avg_contrast:.1f}")
    
    # Find potential issues
    print("\n" + "="*60)
    print("POTENTIAL DATA ISSUES")
    print("="*60)
    
    # Check for letters with very few images
    low_count_letters = [(letter, stats['count']) for letter, stats in letter_stats.items() if stats['count'] < 50]
    if low_count_letters:
        print(f"\n⚠️  Letters with low image count (<50):")
        for letter, count in low_count_letters:
            print(f"   {letter}: {count} images")
    
    # Check for unusual aspect ratios
    aspect_issues = []
    for letter, stats in letter_stats.items():
        if stats['avg_aspect_ratio'] < 0.5 or stats['avg_aspect_ratio'] > 2.0:
            aspect_issues.append((letter, stats['avg_aspect_ratio']))
    
    if aspect_issues:
        print(f"\n⚠️  Letters with unusual aspect ratios:")
        for letter, aspect in aspect_issues:
            print(f"   {letter}: {aspect:.2f}")
    
    # Check for brightness/contrast issues
    brightness_issues = []
    contrast_issues = []
    
    for letter, stats in letter_stats.items():
        if stats['avg_brightness'] < 100:  # Too dark
            brightness_issues.append((letter, stats['avg_brightness'], 'dark'))
        elif stats['avg_brightness'] > 200:  # Too bright
            brightness_issues.append((letter, stats['avg_brightness'], 'bright'))
        
        if stats['avg_contrast'] < 30:  # Low contrast
            contrast_issues.append((letter, stats['avg_contrast']))
    
    if brightness_issues:
        print(f"\n⚠️  Letters with brightness issues:")
        for letter, brightness, issue_type in brightness_issues:
            print(f"   {letter}: {brightness:.1f} ({issue_type})")
    
    if contrast_issues:
        print(f"\n⚠️  Letters with low contrast:")
        for letter, contrast in contrast_issues:
            print(f"   {letter}: {contrast:.1f}")
    
    # Visual analysis
    print("\n" + "="*60)
    print("VISUAL ANALYSIS")
    print("="*60)
    
    # Show sample images for problematic letters
    problematic_letters = ['D', 'N', 'F', 'J', 'K', 'V', 'P']
    
    for letter in problematic_letters:
        if letter in letter_stats:
            print(f"\n🔍 Sample images for letter {letter}:")
            sample_files = letter_stats[letter]['sample_files']
            
            for i, img_file in enumerate(sample_files[:3]):
                img_path = os.path.join(data_dir, letter, img_file)
                img = cv2.imread(img_path)
                
                if img is not None:
                    # Resize for display
                    display_img = cv2.resize(img, (150, 150))
                    
                    # Save sample image for analysis
                    sample_dir = "sample_analysis"
                    os.makedirs(sample_dir, exist_ok=True)
                    sample_path = os.path.join(sample_dir, f"{letter}_sample_{i+1}.jpg")
                    cv2.imwrite(sample_path, display_img)
                    
                    print(f"   Sample {i+1}: {img_file} -> saved as {sample_path}")
    
    return letter_stats

def create_improved_training_data():
    """Create improved training data with better preprocessing."""
    
    print("\n" + "="*60)
    print("CREATING IMPROVED TRAINING DATA")
    print("="*60)
    
    data_dir = "Data"
    improved_dir = "Improved_Data"
    os.makedirs(improved_dir, exist_ok=True)
    
    letters = [chr(i) for i in range(ord('A'), ord('Z')+1)]
    
    for letter in letters:
        letter_dir = os.path.join(data_dir, letter)
        improved_letter_dir = os.path.join(improved_dir, letter)
        os.makedirs(improved_letter_dir, exist_ok=True)
        
        if not os.path.exists(letter_dir):
            continue
            
        image_files = [f for f in os.listdir(letter_dir) if f.endswith('.jpg')]
        
        print(f"\n🔄 Processing letter {letter}: {len(image_files)} images")
        
        processed_count = 0
        for img_file in image_files:
            img_path = os.path.join(letter_dir, img_file)
            img = cv2.imread(img_path)
            
            if img is not None:
                # Apply preprocessing improvements
                processed_img = preprocess_image(img)
                
                if processed_img is not None:
                    # Save improved image
                    improved_path = os.path.join(improved_letter_dir, img_file)
                    cv2.imwrite(improved_path, processed_img)
                    processed_count += 1
        
        print(f"   ✅ Processed {processed_count}/{len(image_files)} images")

def preprocess_image(img):
    """Apply improved preprocessing to an image."""
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply histogram equalization for better contrast
        equalized = cv2.equalizeHist(gray)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(equalized, (3, 3), 0)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 11, 2)
        
        # Convert back to BGR for consistency
        processed = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        
        return processed
        
    except Exception as e:
        print(f"Error preprocessing image: {str(e)}")
        return None

if __name__ == "__main__":
    # Analyze current training data
    letter_stats = analyze_training_data()
    
    # Create improved training data
    create_improved_training_data()
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print("Check the 'sample_analysis' folder for sample images")
    print("Check the 'Improved_Data' folder for preprocessed training data") 