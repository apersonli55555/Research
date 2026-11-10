# Experiment plan

Read this whole page before buying or building anything. Keep a dated lab notebook from day one: what you did, what went wrong, photos of the setup.

## 1. Parts (roughly $25–40; purchase links are in `PARTS.md`)
| Item | Notes |
|---|---|
| 2–3 × GY-521 breakout boards (MPU6050 sensor) | Buy 3 so you have a spare. Boards from different sellers are better: more variety between units. |
| 1 × ESP32 dev board **or** Arduino Uno/Nano, with USB cable | The firmware compiles for both. |
| Jumper wires, small breadboard | Soldered header pins make far better contact than loose ones. |
| Long USB cable (2–3 m), ideally thin and flat | It has to pass a closed fridge door. |
| Small foam or insulated box, zip bags, silica gel packets | Slows temperature change and keeps moisture off the electronics. |
| A separate thermometer (kitchen probe or indoor/outdoor type) | An independent check on the chip's own thermometer. |

Check the current datasheet for the MPU6050's rated operating temperature range before choosing your extremes, and stay well inside it.

## 2. Wiring
| Sensor pin | ESP32 | Arduino Uno/Nano |
|---|---|---|
| VCC | 3V3 | 5V (GY-521 boards have a regulator) |
| GND | GND | GND |
| SDA | GPIO 21 | A4 |
| SCL | GPIO 22 | A5 |
| AD0 | leave open on sensor 1; connect to 3V3 on sensor 2 | same |

AD0 sets the sensor's address (0x68 or 0x69), which is how two sensors share the same wires. Keep wires short (under 20 cm).

## 3. Software
1. Install the Arduino IDE, open `firmware/thermcal_logger/thermcal_logger.ino`, choose your board and port, upload.
2. Open the Serial Monitor at 115200 baud. You should see `# sensor 0x68 found` and one line per sensor per second.
3. Close the Serial Monitor, then record with:
   `python tools/log_serial.py --port <PORT> --run 2026-11-20_test1 --unit 0x68=A --unit 0x69=B --orientation z_up`

## 4. Sanity checks (do these before any real run)
The firmware has never been run on real hardware, so these checks matter.
1. **Gravity:** board flat on a table, chip facing up. The logger's `az` should be near +1.00 g. Flip it over: near −1.00 g. On its edge: near 0.
2. **Stillness:** `gx, gy, gz` raw means should be steady, typically within a few hundred counts of zero (a few deg/s at most). Rotate the board by hand and watch them move.
3. **Temperature:** hold a finger on the chip; the temperature should rise several degrees in a minute. The chip reads a few degrees above room temperature because it heats itself.
4. **Rate:** `n` should be 100 on every line. If it isn't, tell the assistant.
5. **A 30-minute room-temperature recording:** run the analysis loader on it and confirm nothing crashes.

If any check fails, stop and fix it. Do not collect data on a setup that fails a check.

## 5. The measurement
**Principle.** A board that is not moving has zero true rotation and feels exactly gravity. So any change in its readings as temperature changes is sensor error.

**Setup.** Tape the board firmly to the inside of the foam box so it cannot shift. Put the box in a zip bag with silica gel. Route the USB cable out. The board stays powered and logging the whole time.

**One recording ("run") = one slow temperature sweep, about 3 hours:**
| Run type | How | Covers roughly |
|---|---|---|
| Cold start | Leave the box in the freezer 2 h (powered, logging), then move it to a room and let it warm for 3 h | freezer temperature → room |
| Fridge start | Same, from the fridge | ~5 °C → room |
| Warm start | Warm the box gently (see safety), then let it cool at room temperature for 3 h | ~45 °C → room |

**How many.** At least 6 runs per sensor, ideally 9–12: three or four of each type, on different days. Two sensors record at once. More runs matter more than longer runs, because each power-on behaves a little differently and the analysis holds out whole runs.

**Rules.**
- Power on at least 10 minutes before the part of the run you care about; the analysis discards the first 10 minutes.
- Do not touch, move or bump the box during a run. Note the time if you do. The analysis drops seconds where the gyroscope noise jumps.
- A fridge or freezer compressor vibrates. That is why the *warming on a table* part is the main data.
- Give each run a unique name with the date. Never overwrite or edit a data file.
- Write down for every run: date, start and end time, where the box was, the separate thermometer's readings at start and end, anything odd.
- Change one thing at a time. Keep the orientation `z_up` for all main runs.

## 6. Safety and care of the equipment
- **Warmth:** aim for 45–50 °C at most. Use a warm room, a sunny windowsill, a heating pad on low, or warm (not hot) air from a hair dryer at a distance. **No ovens, no heat guns, no open flame.** Never heat a battery; this setup is powered over USB only.
- **Condensation:** cold electronics collect water when they meet warm air. Keep the board sealed in the zip bag with silica gel until it is back at room temperature. Water on a powered board can destroy it.
- **Food:** keep the sealed box away from unwrapped food, and ask whoever runs the kitchen first.
- **Cables in doors:** a thin flat cable lets the fridge door seal. Check that the door still closes.

## 7. After the first 6 runs
Run `python scripts/run_analysis.py` and bring the output to the assistant for review. Do not change the analysis after seeing the results without writing down why (see `PREREGISTRATION.md`).

## 8. Known weaknesses to think about (reviewers will)
- The chip's thermometer measures the chip, not the air. During fast changes, different parts of the chip are at different temperatures. Slow sweeps reduce this; the heating-rate model term is meant to capture what remains.
- The accelerometer check uses the *magnitude* of gravity, so it mostly tests the axis pointing up.
- A home freezer and a foam box are not a temperature chamber. Say so plainly, and report how repeatable your runs were.
- Two or three sensors of one model cannot speak for all sensors.
