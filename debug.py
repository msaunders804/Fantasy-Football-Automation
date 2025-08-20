import sys
import os

print("Current directory:", os.getcwd())
print("Python files in directory:")
for file in os.listdir('.'):
    if file.endswith('.py'):
        print(f"  - {file}")

try:
    import ml_draft_analyzer
    print(f"\nSuccessfully imported ml_draft_analyzer")
    print("Available classes/functions:")
    for item in dir(ml_draft_analyzer):
        if not item.startswith('_'):
            print(f"  - {item}")
except Exception as e:
    print(f"Failed to import ml_draft_analyzer: {e}")