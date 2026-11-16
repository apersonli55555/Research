"""Record the sensor board's serial output to a CSV file on your laptop.

Example:
  python tools/log_serial.py --port /dev/ttyUSB0 --run 2026-11-20_warmup1 \
      --unit 0x68=A --unit 0x69=B --orientation z_up --note "freezer -> room, in foam box"

Find the port: Windows "COM3"-style (Device Manager), macOS /dev/cu.usbserial-*,
Linux /dev/ttyUSB0 or /dev/ttyACM0. Stop with Ctrl+C. Needs: pip install pyserial
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

HEADER = "ms,addr,n,ax,ay,az,gx,gy,gz,temp,sax,say,saz,sgx,sgy,sgz"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", required=True)
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--run", required=True, help="unique name for this recording, e.g. 2026-11-20_warmup1")
    ap.add_argument("--unit", action="append", default=[], help="ADDR=NAME, e.g. 0x68=A (repeatable)")
    ap.add_argument("--orientation", required=True, help="how the board sits, e.g. z_up, x_down")
    ap.add_argument("--note", default="")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "data" / "raw"))
    a = ap.parse_args()

    import serial  # imported here so --help works without pyserial

    units = {k.lower(): v for k, v in (u.split("=") for u in a.unit)}
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    csv_path = out / f"{a.run}.csv"
    if csv_path.exists():
        sys.exit(f"{csv_path} already exists. Choose a new --run name; never overwrite data.")
    meta = dict(run=a.run, orientation=a.orientation, note=a.note, units=units, port=a.port,
                started_utc=dt.datetime.now(dt.timezone.utc).isoformat(), simulated=False)
    (out / f"{a.run}.json").write_text(json.dumps(meta, indent=2))

    n, last = 0, 0.0
    with serial.Serial(a.port, a.baud, timeout=2) as ser, open(csv_path, "w") as f:
        f.write("utc,run,unit,orientation," + HEADER + "\n")
        print(f"Recording to {csv_path}. Ctrl+C to stop.")
        try:
            while True:
                line = ser.readline().decode("ascii", errors="replace").strip()
                if not line or line.startswith("#") or line.startswith("ms,"):
                    if line.startswith("#"):
                        print(line)
                    continue
                parts = line.split(",")
                if len(parts) != 16:
                    continue  # partial line
                unit = units.get(parts[1].lower(), parts[1])
                f.write(f"{dt.datetime.now(dt.timezone.utc).isoformat()},{a.run},{unit},{a.orientation},{line}\n")
                f.flush(); n += 1
                if time.time() - last > 10:
                    last = time.time()
                    print(f"  {n} lines | unit {unit}: {float(parts[9]) / 340 + 36.53:6.2f} degC, "
                          f"az = {float(parts[5]) / 16384:+.4f} g")
        except KeyboardInterrupt:
            print(f"\nStopped. {n} lines saved.")


if __name__ == "__main__":
    main()
