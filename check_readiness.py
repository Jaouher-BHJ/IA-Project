"""
Quick check to verify everything is ready for model comparison
"""
import os
import sys

print("=" * 70)
print("READINESS CHECK FOR MODEL COMPARISON")
print("=" * 70)

checks_passed = 0
checks_total = 0

# Check 1: XGBoost installed
checks_total += 1
try:
    import xgboost
    print(f"\n✓ [1/5] XGBoost installed (version {xgboost.__version__})")
    checks_passed += 1
except ImportError:
    print("\n✗ [1/5] XGBoost NOT installed")
    print("   Run: pip install xgboost")

# Check 2: Comparison script exists
checks_total += 1
if os.path.exists('models/compare_multitarget_models.py'):
    print("✓ [2/5] Comparison script exists (models/compare_multitarget_models.py)")
    checks_passed += 1
else:
    print("✗ [2/5] Comparison script NOT found")

# Check 3: Notebook exists
checks_total += 1
if os.path.exists('preprocess.ipynb'):
    print("✓ [3/5] Preprocessing notebook exists (preprocess.ipynb)")
    checks_passed += 1
else:
    print("✗ [3/5] Preprocessing notebook NOT found")

# Check 4: Data directory exists
checks_total += 1
if os.path.exists('data') and os.path.isdir('data'):
    csv_files = [f for f in os.listdir('data') if f.endswith('.csv')]
    print(f"✓ [4/5] Data directory exists ({len(csv_files)} CSV files found)")
    checks_passed += 1
else:
    print("✗ [4/5] Data directory NOT found")

# Check 5: model_ready.csv exists
checks_total += 1
if os.path.exists('data/model_ready.csv'):
    import pandas as pd
    df = pd.read_csv('data/model_ready.csv')
    print(f"✓ [5/5] model_ready.csv exists ({len(df):,} rows)")
    checks_passed += 1
    
    # Check for required columns
    required_cols = ['lum', 'atm', 'agg', 'int', 'hour', 'day_of_week', 'month', 'col', 'max_severity']
    missing_cols = [c for c in required_cols if c not in df.columns]
    
    if missing_cols:
        print(f"   ⚠ Warning: Missing columns: {missing_cols}")
    else:
        print(f"   ✓ All required columns present")
else:
    print("✗ [5/5] model_ready.csv NOT found")
    print("   → You need to run preprocess.ipynb first!")

print("\n" + "=" * 70)
print(f"RESULT: {checks_passed}/{checks_total} checks passed")
print("=" * 70)

if checks_passed == checks_total:
    print("\n🎉 ALL CHECKS PASSED! You're ready to run the comparison.")
    print("\nNext step:")
    print("   python models/compare_multitarget_models.py")
elif checks_passed >= 4:
    print("\n⚠ ALMOST READY! Just need to run the notebook.")
    print("\nNext steps:")
    print("   1. Open and run preprocess.ipynb")
    print("   2. Then run: python models/compare_multitarget_models.py")
else:
    print("\n❌ SETUP INCOMPLETE. Please check the failed items above.")
    print("\nRefer to README.md for detailed instructions.")

print("\n" + "=" * 70)
