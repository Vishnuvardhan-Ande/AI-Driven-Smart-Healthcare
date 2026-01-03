"""
Comprehensive training script that trains all improved models and optimizes fusion.

This script:
1. Trains improved image model with focal loss and better augmentation
2. Trains improved clinical model with multiple algorithms
3. Optimizes fusion weights and thresholds
4. Evaluates the complete system

Usage:
    python src/train_all_improved.py
"""

import os
import sys
import subprocess

def run_script(script_path, description):
    """Run a Python script and handle errors."""
    print("\n" + "=" * 60)
    print(f"STEP: {description}")
    print("=" * 60)
    
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            check=True,
            capture_output=False
        )
        print(f"\n✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error in {description}")
        print(f"Return code: {e.returncode}")
        return False
    except Exception as e:
        print(f"\n❌ Unexpected error in {description}: {e}")
        return False


def main():
    print("=" * 60)
    print("COMPREHENSIVE MODEL TRAINING PIPELINE")
    print("=" * 60)
    print("\nThis will train:")
    print("1. Improved image model (DenseNet121 with focal loss)")
    print("2. Improved clinical model (multiple algorithms)")
    print("3. Optimized fusion configuration")
    print("\nNote: This may take several hours depending on your hardware.")
    
    response = input("\nContinue? (y/n): ")
    if response.lower() != 'y':
        print("Training cancelled.")
        return
    
    # Step 1: Train improved image model
    success1 = run_script(
        "src/train_image_improved.py",
        "Training Improved Image Model"
    )
    
    if not success1:
        print("\n⚠️  Image model training failed. Continuing with existing model...")
    
    # Step 2: Train improved clinical model
    success2 = run_script(
        "src/train_clinical_improved.py",
        "Training Improved Clinical Model"
    )
    
    if not success2:
        print("\n⚠️  Clinical model training failed. Please check the errors above.")
        return
    
    # Step 3: Optimize fusion
    success3 = run_script(
        "src/fusion_tuning_improved.py",
        "Optimizing Fusion Configuration"
    )
    
    if not success3:
        print("\n⚠️  Fusion optimization failed. Using default fusion weights...")
    
    # Final summary
    print("\n" + "=" * 60)
    print("TRAINING PIPELINE COMPLETE")
    print("=" * 60)
    
    if success1 and success2 and success3:
        print("\n✅ All models trained and optimized successfully!")
        print("\nNext steps:")
        print("1. Evaluate the system: python src/system_eval.py")
        print("2. Test the Flask app: python src/app.py")
    else:
        print("\n⚠️  Some steps failed. Please review the errors above.")
        print("You can run individual scripts to debug:")
        print("  - python src/train_image_improved.py")
        print("  - python src/train_clinical_improved.py")
        print("  - python src/fusion_tuning_improved.py")


if __name__ == "__main__":
    main()

