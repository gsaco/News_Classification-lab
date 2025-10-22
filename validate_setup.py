"""
Basic validation script to check if the repository structure and dependencies are correct.

Run this before starting the training notebook to verify your environment.

Usage:
    python validate_setup.py
"""

import sys
import os

def check_python_version():
    """Check Python version is 3.10+"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 10):
        print(f"❌ Python 3.10+ required. Found: {version.major}.{version.minor}")
        return False
    print(f"✓ Python version: {version.major}.{version.minor}.{version.micro}")
    return True

def check_directory_structure():
    """Check if all required directories exist"""
    required_dirs = ['data', 'notebooks', 'outputs', 'src']
    all_exist = True
    
    for dir_name in required_dirs:
        if os.path.exists(dir_name):
            print(f"✓ Directory '{dir_name}' exists")
        else:
            print(f"❌ Directory '{dir_name}' missing")
            all_exist = False
    
    return all_exist

def check_required_files():
    """Check if required files exist"""
    required_files = [
        'requirements.txt',
        'README.md',
        'data/rpp_news_50.csv',
        'notebooks/agnews_train_eval.ipynb'
    ]
    all_exist = True
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✓ File '{file_path}' exists")
        else:
            print(f"❌ File '{file_path}' missing")
            all_exist = False
    
    return all_exist

def check_dependencies():
    """Check if key dependencies are installed"""
    dependencies = [
        'torch',
        'transformers',
        'datasets',
        'sklearn',
        'pandas',
        'numpy',
        'matplotlib',
        'seaborn'
    ]
    
    missing = []
    installed = []
    
    for dep in dependencies:
        try:
            __import__(dep)
            installed.append(dep)
        except ImportError:
            missing.append(dep)
    
    for dep in installed:
        print(f"✓ Package '{dep}' installed")
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print("Install with: pip install -r requirements.txt")
        return False
    
    return True

def check_data_file():
    """Validate the RPP news CSV file"""
    import pandas as pd
    
    try:
        df = pd.read_csv('data/rpp_news_50.csv')
        if len(df) >= 50:
            print(f"✓ RPP news file contains {len(df)} articles")
            required_cols = ['title', 'description', 'link', 'published']
            if all(col in df.columns for col in required_cols):
                print(f"✓ RPP news file has all required columns")
                return True
            else:
                print(f"❌ RPP news file missing required columns")
                return False
        else:
            print(f"❌ RPP news file has only {len(df)} articles (expected 50)")
            return False
    except Exception as e:
        print(f"❌ Error reading RPP news file: {e}")
        return False

def main():
    """Run all validation checks"""
    print("=" * 70)
    print("News Classification Lab - Setup Validation")
    print("=" * 70)
    
    checks = [
        ("Python Version", check_python_version),
        ("Directory Structure", check_directory_structure),
        ("Required Files", check_required_files),
        ("Dependencies", check_dependencies),
        ("Data File", check_data_file)
    ]
    
    results = []
    
    for check_name, check_func in checks:
        print(f"\n{check_name}:")
        print("-" * 70)
        try:
            result = check_func()
            results.append(result)
        except Exception as e:
            print(f"❌ Error during {check_name}: {e}")
            results.append(False)
    
    print("\n" + "=" * 70)
    if all(results):
        print("✅ All checks passed! You're ready to run the notebook.")
        print("\nNext steps:")
        print("1. Open notebooks/agnews_train_eval.ipynb")
        print("2. Run all cells to train and evaluate models")
        print("3. Check outputs/ directory for results")
    else:
        print("❌ Some checks failed. Please fix the issues above.")
        print("\nIf dependencies are missing, run:")
        print("  pip install -r requirements.txt")
    print("=" * 70)
    
    return all(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
