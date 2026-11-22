# Getting going again, and the first fridge run

Everything here assumes the setup that already works: one sensor, the breadboard, the ESP32 on COM5, and Python installed.

---

## Part A: Starting up again after unplugging (5 minutes)

**The program stays on the ESP32.** It's saved inside the board, so you do **not** need to upload it again. Just plug in and record.

1. **Check the wiring is still in place.** On the breadboard's a–e side, columns 1, 3, 5 and 7 should each hold two wires:
   - **1:** sensor red + ESP32 red
   - **3:** sensor black + ESP32 olive-brown
   - **5:** sensor brown + ESP32 orange
   - **7:** sensor white + ESP32 yellow

   On the ESP32: red on **3V3**, olive-brown on **GND**, orange on **D22**, yellow on **D21**. Push any loose wire back in.
2. **Plug the ESP32 into the laptop.** The red **PWR** light on the ESP32 and the small red light on the sensor should both come on.
3. **Check the COM port.** It was COM5, but Windows sometimes changes it.
   - Open the Arduino IDE and look at the bottom right: it says "ESP32 Dev Module on COMx".
   - Or open Windows **Device Manager → Ports (COM & LPT)** and look for "USB-SERIAL CH340 (COMx)".
4. **Optional quick test:** in the Arduino IDE, open the Serial Monitor (top-right icon), set 115200 baud, and press **EN** on the ESP32. You should see `# sensor 0x68 found` and a line every second. **Then close the Serial Monitor**, or the recording step can't use the port.
5. **Open the black command window in the right folder.** In File Explorer, go to **Documents → Arduino → Research-main → sensor-thermal-cal**, click the address bar, type `cmd` and press Enter.
6. **Record** (give every recording a **new name**, with the date):
   ```
   py tools/log_serial.py --port COM5 --run 2026-11-25_room1 --unit 0x68=A --orientation natural
   ```
   Press **Ctrl + C** to stop.
7. **Check every recording right after you make it:**
   ```
   py tools/check_recording.py data/raw/2026-11-25_room1.csv
   ```
   All lines should say **PASS**. If anything says **CHECK**, don't record more yet; ask for help.

**If you haven't done the 30-minute room-temperature test yet, do it first** (step 6, left untouched for 30 minutes), and make sure step 7 passes.

---

## Part B: Before the first fridge run

You need:
- **The probe thermometer** (only the probe goes inside; the display stays outside).
- **About 5 silica gel packets** (from shoe boxes or packaging is fine).
- **A large zip bag** and a **small foam box or insulated lunch bag**.
- **Tape.**
- **The long flat USB cable.**
- **A notebook** for writing down times and readings.

Also read and sign `PREREGISTRATION.md` **before** the first real (fridge or freezer) run, and commit it.

---

## Part C: One fridge run, step by step (about 5½ hours, mostly waiting)

**Packing (15 minutes)**
1. **Put everything in the box:** the sensor, the breadboard and the ESP32. The wires are short, so they all go together.
2. **Tape the sensor down** in its usual natural position so it can't move. Tape the breadboard and the ESP32 down too, and tape the wires near the sensor, so a tug on the cable can't shift anything.
3. **Put the thermometer's probe next to the sensor** (touching the box floor beside it, not on the chip) and tape its wire down. The display stays outside.
4. **Add the silica gel packets** to the box.
5. **Plug the long USB cable into the ESP32** and route it out.
6. **Put the whole box into the zip bag** and seal it, leaving only a small gap for the USB cable and the thermometer wire. Close the gap as much as you can with tape.

**Start recording at room temperature**
7. **Place the bag on the table** next to the fridge, plug the USB cable into the laptop, and check that both red lights are on.
8. **Start recording** (new name each time):
   ```
   py tools/log_serial.py --port COM5 --run 2026-11-25_fridge1 --unit 0x68=A --orientation natural --note "fridge run 1"
   ```
9. **Wait 15 minutes without touching anything.** This lets the board warm up after power-on. Write down the time and the thermometer reading.

**Into the fridge (2 hours)**
10. **Gently lift the bag into the fridge** and set it on a shelf. Run the flat cable out through the door, then close the door and check it seals fully. The laptop stays outside on the counter, still recording.
11. **Write down the time and the thermometer reading.**
12. **Leave it for 2 hours.** Don't open the fridge if you can avoid it, and keep the laptop awake: plug it in, and set it not to sleep.

**Back to the table (3 hours)**
13. **Gently take the bag out** and set it in the same spot on the table. **Don't open the bag.**
14. **Write down the time and the thermometer reading.**
15. **Leave it for 3 hours** while it warms back to room temperature, still recording.

**Finish**
16. **Press Ctrl + C** in the command window to stop. Write down the final time and thermometer reading.
17. **Check the recording:**
    ```
    py tools/check_recording.py data/raw/2026-11-25_fridge1.csv
    ```
18. **Only now open the bag,** once everything is back at room temperature.
19. **In your notebook, write down:** the date; the start, fridge-in, fridge-out and stop times; the thermometer readings at each; and anything unusual (a bump, the door opened, a cable pulled).

**Afterwards:** send the checker's output for review. The freezer run works exactly the same way, with "freezer" in place of "fridge". Mix the order of run types across different days (see `EXPERIMENT_PLAN.md`).

---

## If something goes wrong
- **"could not open port":** the Arduino Serial Monitor is open. Close it.
- **"already exists":** use a new `--run` name. Never delete or edit a recording.
- **The red lights don't come on:** check the red and olive-brown wires on the ESP32 (3V3 and GND) and in breadboard columns 1 and 3.
- **The laptop went to sleep mid-run:** the recording stops. Note the time and start a new run another day.
- **Anything gets hot or smells:** unplug immediately.
