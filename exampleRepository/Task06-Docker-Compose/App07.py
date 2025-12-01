import os
import subprocess
import sys

print("=" * 60)
print("APP07: DOCKER-COMPOSE ORCHESTRATION")
print("=" * 60)

# ============================================================
# STEP 1: Clean Up Previous Run
# ============================================================
print("\n1. Cleaning up previous run...")

if os.path.exists('shared_data'):
    import shutil
    shutil.rmtree('shared_data')
    print("   ✓ Removed old shared_data/")

os.makedirs('shared_data', exist_ok=True)
print("   ✓ Created fresh shared_data/")

# ============================================================
# STEP 2: Run Docker Compose
# ============================================================
print("\n2. Starting Docker Compose orchestration...")
print("   This will:")
print("   - Start Container 1 → Copy activation data")
print("   - Start Container 2 → Copy AI model")
print("   - Start Container 3 → Load model & activate")
print()

try:
    # Run docker-compose
    result = subprocess.run(
        ['docker-compose', 'up', '--abort-on-container-exit'],
        check=True,
        capture_output=False
    )
    
    print("\n" + "=" * 60)
    print("DOCKER-COMPOSE EXECUTION COMPLETE")
    print("=" * 60)

except subprocess.CalledProcessError as e:
    print(f"\n❌ Error running docker-compose: {e}")
    sys.exit(1)

# ============================================================
# STEP 3: Verify Files Were Copied
# ============================================================
print("\n3. Verifying shared data files...")

files_to_check = [
    'shared_data/activation_data.csv',
    'shared_data/currentSolution.pkl'
]

all_present = True
for file_path in files_to_check:
    if os.path.exists(file_path):
        size = os.path.getsize(file_path)
        print(f"   ✓ {file_path} ({size} bytes)")
    else:
        print(f"   ❌ Missing: {file_path}")
        all_present = False

if all_present:
    print("\n✅ All files successfully copied!")
else:
    print("\n⚠️ Some files are missing!")

# ============================================================
# STEP 4: Clean Up Containers
# ============================================================
print("\n4. Cleaning up containers...")
subprocess.run(['docker-compose', 'down'], check=False)
print("   ✓ Containers stopped and removed")

print("\n" + "=" * 60)
print("APP07 COMPLETE")
print("=" * 60)