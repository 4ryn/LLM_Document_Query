#!/usr/bin/env python3
"""
Debug script to identify OpenAI compatibility issues
Run this to diagnose the ChatCompletion error
"""

import sys
import os
import subprocess
import importlib

def check_openai_installation():
    """Check OpenAI installation and version"""
    print("=== OpenAI Installation Check ===")
    
    try:
        import openai
        print(f"✓ OpenAI imported successfully")
        print(f"✓ Version: {openai.__version__}")
        print(f"✓ Location: {openai.__file__}")
        
        # Check if AsyncOpenAI is available
        from openai import AsyncOpenAI
        print(f"✓ AsyncOpenAI available")
        
        # Check if old ChatCompletion still exists (shouldn't in v1.0+)
        if hasattr(openai, 'ChatCompletion'):
            print(f"⚠️  WARNING: openai.ChatCompletion still exists (unexpected in v1.0+)")
        else:
            print(f"✓ openai.ChatCompletion properly removed")
            
    except ImportError as e:
        print(f"✗ OpenAI import failed: {e}")
        return False
    
    return True

def check_conflicting_packages():
    """Check for packages that might cause conflicts"""
    print("\n=== Checking for Conflicting Packages ===")
    
    conflicting_packages = [
        'openai-api',
        'openai-python', 
        'openai-whisper',
        'openai-gym'
    ]
    
    try:
        result = subprocess.run([sys.executable, '-m', 'pip', 'list'], 
                              capture_output=True, text=True)
        installed_packages = result.stdout.lower()
        
        for package in conflicting_packages:
            if package in installed_packages:
                print(f"⚠️  Potential conflict: {package} is installed")
            else:
                print(f"✓ No conflict with {package}")
                
    except Exception as e:
        print(f"Could not check packages: {e}")

def check_python_path():
    """Check Python path for multiple OpenAI installations"""
    print("\n=== Python Path Check ===")
    
    for path in sys.path:
        openai_path = os.path.join(path, 'openai')
        if os.path.exists(openai_path):
            print(f"OpenAI found in: {openai_path}")

def scan_code_for_old_syntax():
    """Scan current directory for old OpenAI syntax"""
    print("\n=== Scanning Code for Old Syntax ===")
    
    old_patterns = [
        'openai.ChatCompletion',
        'openai.Completion',
        'openai.api_key',
        'ChatCompletion.create'
    ]
    
    for root, dirs, files in os.walk('.'):
        # Skip common directories
        dirs[:] = [d for d in dirs if d not in ['.git', '__pycache__', '.env', 'venv', 'env']]
        
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                        for pattern in old_patterns:
                            if pattern in content:
                                print(f"⚠️  Found '{pattern}' in {filepath}")
                except Exception as e:
                    continue

def test_async_openai():
    """Test AsyncOpenAI initialization"""
    print("\n=== Testing AsyncOpenAI ===")
    
    try:
        from openai import AsyncOpenAI
        
        # Try to create client (should work even with dummy key)
        client = AsyncOpenAI(
            api_key="dummy-key-for-testing",
            base_url="https://openrouter.ai/api/v1"
        )
        print("✓ AsyncOpenAI client created successfully")
        
        # Check if the problematic method exists
        if hasattr(client, 'chat'):
            print("✓ client.chat exists")
            if hasattr(client.chat, 'completions'):
                print("✓ client.chat.completions exists")
                if hasattr(client.chat.completions, 'create'):
                    print("✓ client.chat.completions.create exists")
                else:
                    print("✗ client.chat.completions.create missing")
            else:
                print("✗ client.chat.completions missing")
        else:
            print("✗ client.chat missing")
            
    except Exception as e:
        print(f"✗ AsyncOpenAI test failed: {e}")

def clean_installation_suggestion():
    """Suggest clean installation steps"""
    print("\n=== Clean Installation Suggestion ===")
    print("If issues persist, try:")
    print("1. pip uninstall openai")
    print("2. pip cache purge")
    print("3. pip install openai==1.3.7")
    print("4. Restart your Python interpreter/server")

def main():
    """Run all diagnostic checks"""
    print("OpenAI Compatibility Diagnostic Tool")
    print("=" * 50)
    
    if not check_openai_installation():
        return
    
    check_conflicting_packages()
    check_python_path()
    scan_code_for_old_syntax()
    test_async_openai()
    clean_installation_suggestion()
    
    print("\n" + "=" * 50)
    print("Diagnostic complete!")

if __name__ == "__main__":
    main()