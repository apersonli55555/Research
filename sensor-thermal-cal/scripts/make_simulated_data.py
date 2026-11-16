"""Create FAKE recordings in data/simulated/ so the analysis can be tested before real data exist."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from thermcal import simulate  # noqa: E402
simulate.make(ROOT / "data" / "simulated")
print("wrote FAKE data to", ROOT / "data" / "simulated")
