import os
import re

def check_imports():
    for root, dirs, files in os.walk('.'):
        for name in files:
            if name.endswith('.dart'):
                filepath = os.path.join(root, name)
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    imports = re.findall(r"import\s+['\"]([^'\"]+)['\"]", content)
                    for imp in imports:
                        if imp.startswith('package:') or imp.startswith('dart:'):
                            continue
                        
                        target_dir = os.path.dirname(os.path.normpath(os.path.join(root, imp)))
                        target_file = os.path.basename(imp)
                        
                        if not os.path.exists(target_dir):
                            continue
                            
                        actual_files = os.listdir(target_dir)
                        if target_file not in actual_files:
                            # It exists case-insensitively but not exactly
                            lower_files = {f.lower(): f for f in actual_files}
                            if target_file.lower() in lower_files:
                                actual_casing = lower_files[target_file.lower()]
                                print(f"CASE MISMATCH in {filepath}: imported '{imp}', actual file is '{actual_casing}'")

if __name__ == '__main__':
    check_imports()
    print("Done checking imports.")
