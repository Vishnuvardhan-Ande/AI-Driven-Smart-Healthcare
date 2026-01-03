"""
Interactive step-by-step training script.

This allows you to run training steps individually instead of all at once.
You can pause between steps and choose which one to run next.

Usage:
    python src/train_step_by_step.py
"""

import os
import sys
import subprocess

def print_menu():
    """Display the menu of available training steps."""
    print("\n" + "=" * 60)
    print("STEP-BY-STEP TRAINING MENU")
    print("=" * 60)
    print("\nAvailable steps:")
    print("1. Train Improved Image Model (DenseNet121 with focal loss)")
    print("   - This may take 1-3 hours depending on your hardware")
    print("   - Trains on chest X-ray images")
    print()
    print("2. Train Improved Clinical Model (Multiple algorithms)")
    print("   - This may take 30-60 minutes")
    print("   - Trains on clinical data (XGBoost, LightGBM, CatBoost)")
    print()
    print("3. Optimize Fusion Configuration")
    print("   - This may take 15-30 minutes")
    print("   - Requires both image and clinical models to be trained")
    print()
    print("4. Run All Steps (Sequential)")
    print("   - Runs steps 1, 2, and 3 in sequence")
    print()
    print("0. Exit")
    print("=" * 60)

def run_script(script_path, description):
    """Run a Python script and handle errors."""
    print("\n" + "=" * 60)
    print(f"RUNNING: {description}")
    print("=" * 60)
    print(f"Script: {script_path}")
    print("\nYou can stop this at any time with Ctrl+C")
    print("=" * 60 + "\n")
    
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            check=True,
            capture_output=False
        )
        print("\n" + "=" * 60)
        print(f"✅ {description} completed successfully!")
        print("=" * 60)
        return True
    except KeyboardInterrupt:
        print("\n\n⚠️  Process interrupted by user (Ctrl+C)")
        print("Progress may have been saved depending on the script.")
        return False
    except subprocess.CalledProcessError as e:
        print("\n" + "=" * 60)
        print(f"❌ Error in {description}")
        print(f"Return code: {e.returncode}")
        print("=" * 60)
        return False
    except Exception as e:
        print("\n" + "=" * 60)
        print(f"❌ Unexpected error in {description}: {e}")
        print("=" * 60)
        return False

def check_prerequisites(step):
    """Check if prerequisites are met for a step."""
    if step == 3:  # Fusion optimization requires models
        image_model = "models/dense_best.h5"
        clinical_model = "models/clinical_best.pkl"
        
        missing = []
        if not os.path.exists(image_model):
            missing.append(f"Image model: {image_model}")
        if not os.path.exists(clinical_model):
            missing.append(f"Clinical model: {clinical_model}")
        
        if missing:
            print("\n⚠️  Prerequisites not met:")
            for item in missing:
                print(f"   - {item}")
            print("\nPlease run the required training steps first.")
            return False
    
    return True

def main():
    print("=" * 60)
    print("HEALTHCARE AI - STEP-BY-STEP TRAINING")
    print("=" * 60)
    print("\nThis tool allows you to run training steps individually.")
    print("You can pause between steps and resume later.\n")
    
    completed_steps = {
        1: False,  # Image model
        2: False,  # Clinical model
        3: False   # Fusion optimization
    }
    
    while True:
        print_menu()
        
        # Show progress
        if any(completed_steps.values()):
            print("\nCompleted steps:")
            if completed_steps[1]:
                print("  ✅ Step 1: Image Model")
            if completed_steps[2]:
                print("  ✅ Step 2: Clinical Model")
            if completed_steps[3]:
                print("  ✅ Step 3: Fusion Optimization")
        
        try:
            choice = input("\nSelect a step to run (0-4): ").strip()
            
            if choice == "0":
                print("\nExiting. You can resume training later by running this script again.")
                break
            
            elif choice == "1":
                if not check_prerequisites(1):
                    continue
                success = run_script(
                    "src/train_image_improved.py",
                    "Training Improved Image Model"
                )
                if success:
                    completed_steps[1] = True
            
            elif choice == "2":
                if not check_prerequisites(2):
                    continue
                success = run_script(
                    "src/train_clinical_improved.py",
                    "Training Improved Clinical Model"
                )
                if success:
                    completed_steps[2] = True
            
            elif choice == "3":
                if not check_prerequisites(3):
                    continue
                success = run_script(
                    "src/fusion_tuning_improved.py",
                    "Optimizing Fusion Configuration"
                )
                if success:
                    completed_steps[3] = True
            
            elif choice == "4":
                print("\n⚠️  This will run all steps sequentially.")
                confirm = input("Continue? (y/n): ").strip().lower()
                if confirm != 'y':
                    continue
                
                # Step 1
                if not completed_steps[1]:
                    if check_prerequisites(1):
                        success = run_script(
                            "src/train_image_improved.py",
                            "Training Improved Image Model"
                        )
                        if success:
                            completed_steps[1] = True
                        else:
                            print("\n⚠️  Image model training failed. Continuing...")
                
                # Step 2
                if not completed_steps[2]:
                    if check_prerequisites(2):
                        success = run_script(
                            "src/train_clinical_improved.py",
                            "Training Improved Clinical Model"
                        )
                        if success:
                            completed_steps[2] = True
                        else:
                            print("\n⚠️  Clinical model training failed. Stopping.")
                            break
                
                # Step 3
                if not completed_steps[3]:
                    if check_prerequisites(3):
                        success = run_script(
                            "src/fusion_tuning_improved.py",
                            "Optimizing Fusion Configuration"
                        )
                        if success:
                            completed_steps[3] = True
                
                # Summary
                print("\n" + "=" * 60)
                print("TRAINING PIPELINE COMPLETE")
                print("=" * 60)
                if all(completed_steps.values()):
                    print("\n✅ All steps completed successfully!")
                else:
                    print("\n⚠️  Some steps were skipped or failed.")
                break
            
            else:
                print("\n❌ Invalid choice. Please enter 0-4.")
                continue
            
            # Ask if user wants to continue
            if choice in ["1", "2", "3"]:
                continue_choice = input("\nRun another step? (y/n): ").strip().lower()
                if continue_choice != 'y':
                    print("\nExiting. You can resume training later.")
                    break
        
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            continue
    
    # Final summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("\nCompleted steps:")
    for step_num, completed in completed_steps.items():
        status = "✅" if completed else "⏸️"
        step_names = {1: "Image Model", 2: "Clinical Model", 3: "Fusion Optimization"}
        print(f"  {status} Step {step_num}: {step_names[step_num]}")
    
    if all(completed_steps.values()):
        print("\n🎉 All training steps completed!")
        print("\nNext steps:")
        print("  1. Evaluate the system: python src/system_eval.py")
        print("  2. Test the Flask app: python src/app.py")
    else:
        print("\n💡 To continue training, run: python src/train_step_by_step.py")
    print()

if __name__ == "__main__":
    main()










