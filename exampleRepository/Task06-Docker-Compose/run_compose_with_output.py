import os
import subprocess
import shutil
from datetime import datetime

print("=" * 70)
print("TASK 06 - DOCKER COMPOSE EXECUTION WITH OUTPUT CAPTURE")
print("=" * 70)

output_dir = os.path.join('output', 'stage3_compose_execution')
os.makedirs(output_dir, exist_ok=True)

# ============================================================
# STEP 1: Clean Up
# ============================================================
print("\n1. Cleaning up previous run...")

if os.path.exists('shared_data'):
    shutil.rmtree('shared_data')
    print("   ✓ Removed old shared_data/")

os.makedirs('shared_data', exist_ok=True)
print("   ✓ Created fresh shared_data/")

# ============================================================
# STEP 2: Run Docker Compose with Output Capture
# ============================================================
print("\n2. Running Docker Compose...")

compose_log = os.path.join(output_dir, 'docker_compose_output.log')

with open(compose_log, 'w', encoding="utf-8") as log_file:
    log_file.write(f"Docker Compose Execution Log\n")
    log_file.write(f"Timestamp: {datetime.now().isoformat()}\n")
    log_file.write("=" * 70 + "\n\n")
    
    result = subprocess.run(
        ['docker-compose', 'up', '--abort-on-container-exit'],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    
    log_file.write(result.stdout)

print(f"   ✓ Docker Compose executed")
print(f"   ✓ Log saved to: {compose_log}")

# ============================================================
# STEP 3: Copy Shared Data to Output
# ============================================================
print("\n3. Copying shared data to output folder...")

if os.path.exists('shared_data'):
    for file in os.listdir('shared_data'):
        src = os.path.join('shared_data', file)
        dst = os.path.join(output_dir, file)
        shutil.copy(src, dst)
        
        size = os.path.getsize(dst)
        print(f"   ✓ Copied: {file} ({size} bytes)")

# ============================================================
# STEP 4: Create Summary Report
# ============================================================
print("\n4. Creating summary report...")

summary_file = os.path.join(output_dir, 'execution_summary.txt')
with open(summary_file, 'w', encoding="utf-8") as f:
    f.write("DOCKER COMPOSE EXECUTION SUMMARY\n")
    f.write("=" * 70 + "\n")
    f.write(f"Timestamp: {datetime.now().isoformat()}\n\n")
    
    f.write("FILES GENERATED:\n")
    f.write("-" * 70 + "\n")
    
    if os.path.exists('shared_data'):
        for file in os.listdir('shared_data'):
            path = os.path.join('shared_data', file)
            size = os.path.getsize(path)
            f.write(f"  - {file} ({size} bytes)\n")
    
    f.write("\n")
    f.write("CONTAINER EXECUTION ORDER:\n")
    f.write("-" * 70 + "\n")
    f.write("  1. activationbase_app07  → Copied currentActivation.csv\n")
    f.write("  2. knowledgebase_app07   → Copied currentSolution.pkl\n")
    f.write("  3. codebase_app07        → Loaded model & activated\n")
    
    f.write("\n")
    f.write("OUTPUT LOCATION:\n")
    f.write("-" * 70 + "\n")
    f.write(f"  {os.path.abspath(output_dir)}\n")

print(f"   ✓ Summary saved to: {summary_file}")

# ============================================================
# STEP 5: Clean Up
# ============================================================
print("\n5. Cleaning up containers...")
subprocess.run(['docker-compose', 'down'], capture_output=True)
print("   ✓ Containers stopped")

print("\n" + "=" * 70)
print("EXECUTION COMPLETE!")
print("=" * 70)
print(f"\nAll outputs saved to:")
print(f"  {os.path.abspath(output_dir)}")