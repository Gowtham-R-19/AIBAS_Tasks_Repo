import subprocess
import os
import json
from datetime import datetime

print("=" * 70)
print("TASK 06 - TESTING ALL CONTAINERS")
print("=" * 70)

output_dir = os.path.join('output', 'stage2_container_tests')
os.makedirs(output_dir, exist_ok=True)

results = {
    'timestamp': datetime.now().isoformat(),
    'tests': []
}

# ============================================================
# TEST 1: Container 1 - ActivationBase
# ============================================================
print("\n" + "=" * 70)
print("TEST 1: CONTAINER 1 - ACTIVATIONBASE")
print("=" * 70)

try:
    result = subprocess.run(
        ['docker', 'run', '--rm', 'activationbase_app07'],
        capture_output=True,
        text=True,
        check=True
    )
    
    output_file = os.path.join(output_dir, 'container1_output.txt')
    with open(output_file, 'w') as f:
        f.write("CONTAINER 1 OUTPUT:\n")
        f.write("=" * 70 + "\n")
        f.write(result.stdout)
    
    print("✓ Container 1 ran successfully!")
    print(f"  Output saved to: {output_file}")
    print(f"\n  Content Preview:")
    print(f"  {result.stdout}")
    
    results['tests'].append({
        'container': 'activationbase_app07',
        'status': 'SUCCESS',
        'output': result.stdout
    })

except subprocess.CalledProcessError as e:
    print(f"❌ Container 1 failed: {e}")
    results['tests'].append({
        'container': 'activationbase_app07',
        'status': 'FAILED',
        'error': str(e)
    })

# ============================================================
# TEST 2: Container 2 - KnowledgeBase
# ============================================================
print("\n" + "=" * 70)
print("TEST 2: CONTAINER 2 - KNOWLEDGEBASE")
print("=" * 70)

try:
    result = subprocess.run(
        ['docker', 'run', '--rm', 'knowledgebase_app07'],
        capture_output=True,
        text=True,
        check=True
    )
    
    output_file = os.path.join(output_dir, 'container2_output.txt')
    with open(output_file, 'w') as f:
        f.write("CONTAINER 2 OUTPUT:\n")
        f.write("=" * 70 + "\n")
        f.write(result.stdout)
    
    print("✓ Container 2 ran successfully!")
    print(f"  Output saved to: {output_file}")
    print(f"\n  Content Preview:")
    print(f"  {result.stdout}")
    
    results['tests'].append({
        'container': 'knowledgebase_app07',
        'status': 'SUCCESS',
        'output': result.stdout
    })

except subprocess.CalledProcessError as e:
    print(f"❌ Container 2 failed: {e}")
    results['tests'].append({
        'container': 'knowledgebase_app07',
        'status': 'FAILED',
        'error': str(e)
    })

# ============================================================
# TEST 3: Container 3 - CodeBase (Test PyBrain Import)
# ============================================================
print("\n" + "=" * 70)
print("TEST 3: CONTAINER 3 - CODEBASE (PyBrain Test)")
print("=" * 70)

try:
    result = subprocess.run(
        ['docker', 'run', '--rm', 'codebase_app07', 
         'python3', '-c', 
         'import sys; sys.path.append("/opt/pybrain"); from pybrain.structure import FeedForwardNetwork; print("✓ PyBrain loaded successfully!")'],
        capture_output=True,
        text=True,
        check=True
    )
    
    output_file = os.path.join(output_dir, 'container3_pybrain_test.txt')
    with open(output_file, 'w') as f:
        f.write("CONTAINER 3 - PYBRAIN TEST:\n")
        f.write("=" * 70 + "\n")
        f.write(result.stdout)
    
    print("✓ Container 3 PyBrain test passed!")
    print(f"  Output saved to: {output_file}")
    print(f"\n  Content Preview:")
    print(f"  {result.stdout}")
    
    results['tests'].append({
        'container': 'codebase_app07',
        'test': 'pybrain_import',
        'status': 'SUCCESS',
        'output': result.stdout
    })

except subprocess.CalledProcessError as e:
    print(f"❌ Container 3 PyBrain test failed: {e}")
    print(f"   Error: {e.stderr}")
    results['tests'].append({
        'container': 'codebase_app07',
        'test': 'pybrain_import',
        'status': 'FAILED',
        'error': str(e),
        'stderr': e.stderr
    })

# ============================================================
# SAVE RESULTS
# ============================================================
results_file = os.path.join(output_dir, 'test_results.json')
with open(results_file, 'w') as f:
    json.dump(results, f, indent=2)

print("\n" + "=" * 70)
print("ALL CONTAINER TESTS COMPLETE")
print("=" * 70)
print(f"Results saved to: {results_file}")
print(f"\nSummary:")
for test in results['tests']:
    status_symbol = "✓" if test['status'] == 'SUCCESS' else "❌"
    print(f"  {status_symbol} {test['container']}")