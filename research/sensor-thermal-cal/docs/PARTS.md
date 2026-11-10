# Parts list with links

Links were found by web search on 10 November 2026. Prices, stock and seller quality were **not** checked, and listings change. Read each listing before buying, especially the points under "Check".

| # | Item | Link | Check on the listing |
|---|---|---|---|
| 1 | Motion sensors: AITIAO 4 × pre-soldered GY-521 (MPU-6050) | https://www.amazon.com/AITIAO-Pre-Soldered-Accelerometer-Gyroscope-Raspberry/dp/B0CJDS7WVY | That the header pins really are pre-soldered. Otherwise you need a soldering iron. |
| 1b | Alternative: AOICRIE 3 × pre-soldered GY-521 | https://www.amazon.com/AOICRIE-Accelerometer-Gyroscope-Pre-Soldered-Raspberry/dp/B0D2TJVMNY | Same. |
| 2 | Microcontroller: AITRIP 2 × ESP32 (USB-C) with expansion boards | https://www.amazon.com/clp/B0B82BBKCY | That it includes the expansion (breakout) boards, and has a USB-C socket. |
| 3 | Jumper wires: ELEGOO 120 pcs (male–male, male–female, female–female) | https://www.amazon.com/Elegoo-EL-CP-004-Multicolored-Breadboard-arduino/dp/B01EV70C78 | Length: 10 cm or 20 cm are both fine. |
| 4 | Long USB cable: ANLINK 10 ft flat USB-A to USB-C | https://www.amazon.com/ANLINK-Charging-Replacement-Security-Cord-White/dp/B0GNLCJDHL | That it carries **data**, not charging only. That your laptop has a USB-A port (otherwise get USB-C to USB-C). |
| 5 | Silica gel: Dry & Dry or similar, 50 × 5 g packets | https://www.amazon.com/Desiccant-Packets-Moisture-Absorber-Dehumidifiers/dp/B01NBQEYTJ | Nothing special. |
| 6 | Separate thermometer with wired probe (fridge/freezer type) | https://www.amazon.com/Urageuxy-Refrigerator-Thermometer-Waterproof-Function/dp/B0D3LNLKFG | That its range goes below freezer temperature and above 50 °C. |
| 7 | Small foam box or insulated lunch bag, zip bags, tape | from home | The box should close with the cable passing out. |

## Notes
- **Why pre-soldered sensors:** GY-521 boards usually ship with loose header pins. Loose pins give unreliable contact, which ruins long recordings.
- **Why the ESP32 with expansion board:** a 38-pin ESP32 is too wide for a small breadboard. With the expansion board, wires go straight from the sensor's pins to the board's pins and no breadboard is needed.
- **Drivers:** boards with a CH340 USB chip may need a driver on Windows or macOS before the port shows up.
- **One recommended buy is enough to start:** items 1, 2, 3 and 4. Items 5 to 7 are needed before the cold runs.
