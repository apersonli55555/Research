"""Tests the firmware and the logger without hardware.

1. The REAL firmware source is compiled for the PC against a fake MPU6050
   (tests/host_mock) and its output is checked against the known fake values.
2. That output is sent through a pseudo serial port into tools/log_serial.py.
3. The file the logger writes is read by the analysis loader.
What this cannot test: real wiring, real I2C timing, the real chip. Do the
sanity checks in docs/EXPERIMENT_PLAN.md on the hardware.
Needs g++ and a Unix-like system; skipped otherwise.
"""
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from thermcal import io  # noqa: E402

pytestmark = pytest.mark.skipif(shutil.which("g++") is None or os.name != "posix", reason="needs g++ on a Unix-like system")


@pytest.fixture(scope="module")
def firmware_lines(tmp_path_factory):
    exe = tmp_path_factory.mktemp("fw") / "fw_host"
    subprocess.run(["g++", "-std=c++17", "-I", str(ROOT / "tests/host_mock"), "-include", "cstdlib", "-o", str(exe),
                    str(ROOT / "tests/host_mock/main.cpp")], check=True)
    return subprocess.run([str(exe), "400"], check=True, capture_output=True, text=True).stdout.splitlines()


def test_firmware_output_matches_the_fake_sensor(firmware_lines):
    assert "# sensor 0x68 found" in firmware_lines and "# sensor 0x69 found" in firmware_lines
    header = "ms,addr,n,ax,ay,az,gx,gy,gz,temp,sax,say,saz,sgx,sgy,sgz"
    assert header in firmware_lines
    data = [l.split(",") for l in firmware_lines if l and l[0].isdigit()]
    a = np.array([[float(x) for x in r[2:]] for r in data if r[1] == "0x68"])
    b = np.array([[float(x) for x in r[2:]] for r in data if r[1] == "0x69"])
    assert len(a) >= 398 and len(b) == len(a)
    assert np.all(a[:, 0] == 100)                                             # 100 readings per line
    expect = np.array([100, -200, 16384, 10, -20, 30, -521])                   # ax ay az gx gy gz temp
    assert np.allclose(a[:, 1:8], expect, atol=0.02) and np.allclose(b[:, 1:8], expect + 1000, atol=0.02)
    assert np.allclose(a[:, 8:], np.sqrt(2 / 3), atol=0.02)                    # std of the -1,0,+1 wobble
    ms = np.array([int(r[0]) for r in data if r[1] == "0x68"])
    assert np.all(np.diff(ms) == 1000)                                        # exactly one line per second


def test_logger_and_loader_read_what_the_firmware_writes(firmware_lines, tmp_path):
    import pty
    master, slave = pty.openpty()
    proc = subprocess.Popen([sys.executable, "-u", str(ROOT / "tools/log_serial.py"), "--port", os.ttyname(slave),
                             "--run", "chain_test", "--unit", "0x68=A", "--unit", "0x69=B", "--orientation", "z_up",
                             "--out", str(tmp_path)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    time.sleep(1.5)
    for line in firmware_lines:
        os.write(master, (line + "\r\n").encode())
    os.write(master, b"123,0x68,100,1,2\r\n")                                 # a cut-off line must be ignored
    time.sleep(1.5)
    proc.send_signal(signal.SIGINT)
    out = proc.communicate(timeout=20)[0]
    os.close(master); os.close(slave)
    assert "lines saved" in out, out
    csv = tmp_path / "chain_test.csv"
    rows = csv.read_text().splitlines()
    n_data = sum(1 for l in firmware_lines if l and l[0].isdigit())
    assert len(rows) == n_data + 1 and rows[0].startswith("utc,run,unit,orientation,ms,addr")
    # the logger must refuse to overwrite an existing recording
    again = subprocess.run([sys.executable, str(ROOT / "tools/log_serial.py"), "--port", "/dev/null", "--run", "chain_test",
                            "--orientation", "z_up", "--out", str(tmp_path)], capture_output=True, text=True)
    assert again.returncode != 0 and "already exists" in (again.stderr + again.stdout)
    # the analysis loader converts to physical units
    df = io.load_runs(tmp_path, warmup_s=0)
    assert set(df.unit) == {"A", "B"} and not df["simulated"].any()
    A = df[df.unit == "A"]
    assert np.allclose(A["T"], -521 / 340 + 36.53, atol=0.01)
    assert np.allclose(A["gz"], 30 / 131, atol=1e-3)
    assert np.allclose(A["anorm"], (np.sqrt(100 ** 2 + 200 ** 2 + 16384 ** 2) / 16384 - 1) * 1000, atol=0.01)
