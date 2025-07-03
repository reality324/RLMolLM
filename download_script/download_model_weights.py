#!/usr/bin/env python3
"""
RLMolLM Model Weights Download Script

This script automatically downloads the pre-trained model weights from Hugging Face
and sets up the model_weights/ directory structure.

Usage:
    python download_script/download_model_weights.py

Requirements:
    pip install huggingface_hub
"""

import os
import sys
from pathlib import Path

def install_requirements():
    """Install required packages if not available."""
    try:
        from huggingface_hub import hf_hub_download
        print("✅ huggingface_hub is already installed")
        return True
    except ImportError:
        print("📦 Installing huggingface_hub...")
        try:
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub"])
            print("✅ huggingface_hub installed successfully")
            return True
        except Exception as e:
            print(f"❌ Failed to install huggingface_hub: {e}")
            print("Please install manually: pip install huggingface_hub")
            return False

def download_model_weights():
    """Download model weights from Hugging Face."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("❌ huggingface_hub not available. Please install it first.")
        return False
    
    # Configuration
    REPO_ID = "scofieldlinlin/rlmollm-models"
    TOKEN = "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"
    
    # Get the project root directory (parent of download_script)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    model_weights_dir = project_root / "model_weights"
    
    # Create model_weights directory if it doesn't exist
    model_weights_dir.mkdir(exist_ok=True)
    print(f"📁 Model weights directory: {model_weights_dir}")
    
    # Files to download
    files_to_download = [
        "pytorch_model.bin",  # 208MB
        "config.json"         # 565B
    ]
    
    print(f"🔗 Downloading from repository: {REPO_ID}")
    print("📥 Starting download...")
    
    for filename in files_to_download:
        file_path = model_weights_dir / filename
        
        # Skip if file already exists
        if file_path.exists():
            print(f"⏭️  {filename} already exists, skipping...")
            continue
        
        try:
            print(f"📥 Downloading {filename}...")
            hf_hub_download(
                repo_id=REPO_ID,
                filename=filename,
                token=TOKEN,
                local_dir=str(model_weights_dir),
                local_dir_use_symlinks=False  # Create actual files, not symlinks
            )
            print(f"✅ {filename} downloaded successfully")
            
        except Exception as e:
            print(f"❌ Failed to download {filename}: {e}")
            return False
    
    return True

def verify_downloads():
    """Verify that all required files are present and have reasonable sizes."""
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    model_weights_dir = project_root / "model_weights"
    
    required_files = {
        "pytorch_model.bin": 200_000_000,  # ~200MB minimum
        "config.json": 100                 # ~100B minimum
    }
    
    print("\n🔍 Verifying downloads...")
    
    all_good = True
    for filename, min_size in required_files.items():
        file_path = model_weights_dir / filename
        
        if not file_path.exists():
            print(f"❌ {filename} not found")
            all_good = False
            continue
        
        file_size = file_path.stat().st_size
        if file_size < min_size:
            print(f"❌ {filename} seems too small ({file_size:,} bytes)")
            all_good = False
            continue
        
        print(f"✅ {filename} ({file_size:,} bytes)")
    
    return all_good

def main():
    """Main function to download and set up model weights."""
    print("🚀 RLMolLM Model Weights Download Script")
    print("=" * 50)
    
    # Step 1: Install requirements
    if not install_requirements():
        return 1
    
    # Step 2: Download model weights
    if not download_model_weights():
        print("\n❌ Download failed. Please check your internet connection and try again.")
        return 1
    
    # Step 3: Verify downloads
    if not verify_downloads():
        print("\n❌ Verification failed. Some files may be corrupted.")
        return 1
    
    print("\n🎉 Model weights download completed successfully!")
    print("📁 Files saved to: model_weights/")
    print("🚀 You can now run the training and inference scripts.")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 