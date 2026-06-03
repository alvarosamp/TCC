from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data'

for pasta in[
    'profiles',
    'src/tcc_pipeline',
    'scripts/adapters',
    'docs',
    'artefacts/reports',
    'artefacts/runs',
]:
    (PROJECT_ROOT / pasta).mkdir(parents=True, exist_ok=True)
    
    
print(f"Project root: {PROJECT_ROOT}"
      f"\nData directory: {DATA_DIR}")