import os
import json

print("=" * 70)
print("TASK 06 - SUBMISSION VERIFICATION")
print("=" * 70)

required_files = {
    'Documentation': [
        'README.md',
        'discussion.txt'
    ],
    'Container 1': [
        'container1_activationbase/Dockerfile',
        'container1_activationbase/currentActivation.csv'
    ],
    'Container 2': [
        'container2_knowledgebase/Dockerfile',
        'container2_knowledgebase/currentSolution.pkl'
    ],
    'Container 3': [
        'container3_codebase/Dockerfile',
        'container3_codebase/UE_07_App5.py',
        'container3_codebase/pybrain'
    ],
    'Orchestration': [
        'docker-compose.yml',
        'App07.py'
    ],
    'Output': [
        'output/stage2_container_tests/test_results.json',
        'output/stage3_compose_execution/execution_summary.txt'
    ]
}

print("\nChecking required files...\n")

all_present = True
for category, files in required_files.items():
    print(f"{category}:")
    for file_path in files:
        exists = os.path.exists(file_path)
        symbol = "✓" if exists else "❌"
        print(f"  {symbol} {file_path}")
        if not exists:
            all_present = False
    print()

print("=" * 70)
if all_present:
    print("✅ ALL REQUIRED FILES PRESENT - READY FOR SUBMISSION!")
else:
    print("⚠️  SOME FILES MISSING - PLEASE COMPLETE BEFORE SUBMISSION")
print("=" * 70)