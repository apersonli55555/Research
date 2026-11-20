import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle

fig, ax = plt.subplots(figsize=(12, 10))
ax.set_xlim(0, 12); ax.set_ylim(-1.2, 10.2); ax.axis("off")
fig.patch.set_facecolor("white")

# ---------------- ESP32 board (chip up, USB-C at bottom) ----------------
bx, by, bw, bh = 1.0, 0.9, 2.8, 7.6
ax.add_patch(FancyBboxPatch((bx, by), bw, bh, boxstyle="round,pad=0.02,rounding_size=0.15", fc="#1c1c1c", ec="black"))
ax.add_patch(Rectangle((bx + 0.85, by + 5.1), 1.1, 1.8, fc="#c9c9c9", ec="#888"))       # metal can
ax.text(bx + 1.4, by + 6.0, "ESP-32", ha="center", va="center", fontsize=10, fontweight="bold", color="#333")
ax.add_patch(Rectangle((bx + 1.05, by - 0.25), 0.7, 0.45, fc="#bbbbbb", ec="#666"))     # USB-C
ax.text(bx + 1.4, by - 0.55, "USB-C port (bottom)", ha="center", va="top", fontsize=10)
ax.text(bx + 1.4, by + bh + 0.25, "ESP32  (chip facing up)", ha="center", fontsize=13, fontweight="bold")

left = ["EN", "VP", "VN", "D34", "D35", "D32", "D33", "D25", "D26", "D27", "D14", "D12", "D13", "GND", "VIN"]
right = ["D23", "D22", "TX0", "RX0", "D21", "D19", "D18", "D5", "TX2", "RX2", "D4", "D2", "D15", "GND", "3V3"]
ys = [by + bh - 0.45 - i * 0.47 for i in range(15)]
pin_y = {}
for name, y in zip(left, ys):
    ax.add_patch(Circle((bx + 0.18, y), 0.08, fc="#d4af37", ec="none"))
    ax.text(bx + 0.33, y, name, va="center", fontsize=7.5, color="#bbbbbb")
use = {"D22": "#8B5A2B", "D21": "#ffffff", "GND": "#000000", "3V3": "#e02020"}
for i, (name, y) in enumerate(zip(right, ys)):
    hot = name in use and not (name == "GND" and i != 13)
    ax.add_patch(Circle((bx + bw - 0.18, y), 0.1 if hot else 0.08, fc="#ffd400" if hot else "#d4af37", ec="black" if hot else "none"))
    ax.text(bx + bw - 0.35, y, name, va="center", ha="right", fontsize=9 if hot else 7.5,
            color="#ffd400" if hot else "#bbbbbb", fontweight="bold" if hot else "normal")
    if hot:
        pin_y[name] = y
    # count from top / bottom helper labels
ax.text(bx + bw + 0.05, ys[1], "2nd from top", fontsize=7.5, color="#555", va="bottom")
ax.text(bx + bw + 0.05, ys[4], "5th from top", fontsize=7.5, color="#555", va="bottom")
ax.text(bx + bw + 0.05, ys[13], "2nd from bottom", fontsize=7.5, color="#555", va="bottom")
ax.text(bx + bw + 0.05, ys[14], "last pin, by USB", fontsize=7.5, color="#555", va="top")

# ---------------- MPU-6050 sensor ----------------
sx, sy, sw, sh = 8.0, 2.8, 3.2, 4.4
ax.add_patch(FancyBboxPatch((sx, sy), sw, sh, boxstyle="round,pad=0.02,rounding_size=0.12", fc="#2b3a8c", ec="black"))
ax.add_patch(Rectangle((sx + 2.55, sy + 0.25), 0.5, 0.5, fc="#111", ec="#444"))
ax.text(sx + sw / 2, sy + sh + 0.25, "MPU-6050 sensor (GY-521)", ha="center", fontsize=13, fontweight="bold")
spins = ["VCC", "GND", "SCL", "SDA", "XDA", "XCL", "AD0", "INT"]
sys_ = [sy + sh - 0.4 - i * 0.5 for i in range(8)]
s_y = {}
for name, y in zip(spins, sys_):
    used = name in ("VCC", "GND", "SCL", "SDA")
    ax.add_patch(Circle((sx + 0.18, y), 0.1 if used else 0.08, fc="#ffd400" if used else "#d4af37", ec="black" if used else "none"))
    ax.text(sx + 0.38, y, name + ("" if used else "   (leave empty)"), va="center", fontsize=9.5 if used else 8,
            color="white" if used else "#aab", fontweight="bold" if used else "normal")
    s_y[name] = y

# ---------------- wires ----------------
wires = [("VCC", "3V3", "#e02020", "red"), ("GND", "GND", "#111111", "black"),
         ("SCL", "D22", "#8B5A2B", "brown"), ("SDA", "D21", "#f2f2f2", "white")]
x0 = bx + bw - 0.18
x1 = sx + 0.18
for k, (sp, ep, col, cname) in enumerate(wires):
    ya, yb = pin_y[ep], s_y[sp]
    xm = [5.0, 5.6, 6.8, 6.2][k]
    path_x = [x0, xm, xm, x1]; path_y = [ya, ya, yb, yb]
    if cname == "white":
        ax.plot(path_x, path_y, color="#555", lw=6.5, solid_capstyle="round", zorder=3)
    ax.plot(path_x, path_y, color=col, lw=5, solid_capstyle="round", zorder=4)
    ax.text(xm + 0.08, (ya + yb) / 2, cname, fontsize=9, rotation=90, va="center", zorder=5,
            bbox=dict(fc="white", ec="none", pad=1))

# ---------------- table ----------------
rows = [("red", "VCC", "3V3"), ("black", "GND", "GND"), ("brown", "SCL", "D22"), ("white", "SDA", "D21")]
ax.add_patch(FancyBboxPatch((7.7, -0.95), 3.9, 2.55, boxstyle="round,pad=0.05", fc="#f4f4f4", ec="#999"))
ax.text(9.65, 1.25, "Wire     Sensor  →  ESP32", fontsize=11, fontweight="bold", family="monospace", ha="center")
for i, (c, s, e) in enumerate(rows):
    ax.text(9.65, 0.75 - i * 0.45, f"{c:<7}  {s:<6}  →  {e}", fontsize=11, family="monospace", ha="center")
ax.text(6, 10.0, "Wiring: one MPU-6050 sensor to the ESP32", ha="center", fontsize=16, fontweight="bold")
ax.text(6, 9.55, "Unplug USB while wiring. Check the printed labels before pushing each wire on.",
        ha="center", fontsize=10, color="#b00020")
import os
fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "wiring_one_sensor.png"), dpi=150, bbox_inches="tight")
