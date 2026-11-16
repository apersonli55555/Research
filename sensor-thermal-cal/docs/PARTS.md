# Parts list with links

Links were found by web search on 10 November 2026. Prices, stock and seller quality were **not** checked, and listings change. Read each listing before buying, especially the points under "Check".

| # | Item | Link | Check on the listing |
|---|---|---|---|
| 1 | Motion sensors: AITIAO 4 × pre-soldered GY-521 (MPU-6050) | https://www.amazon.com/AITIAO-Pre-Soldered-Accelerometer-Gyroscope-Raspberry/dp/B0CJDS7WVY | That the header pins really are pre-soldered. Otherwise you need a soldering iron. |
| 1b | Alternative: AOICRIE 3 × pre-soldered GY-521 | https://www.amazon.com/AOICRIE-Accelerometer-Gyroscope-Pre-Soldered-Raspberry/dp/B0D2TJVMNY | Same. |
| 2 | Microcontroller: AITRIP 2 × ESP32 (USB-C) with expansion boards | https://www.amazon.com/clp/B0B82BBKCY | That it includes the expansion (breakout) boards, and has a USB-C socket. |
| 3 | Breadboards + jumper wires: ELEGOO 3 × 400-point breadboards and 120 wires (M–M, M–F, F–F) | https://www.amazon.com/ELEGOO-400-Point-Breadboard-Prototyping-Electronics/dp/B0H7RX1LC9 | Price and stock not checked. |
| 3b | Alternative: ELEGOO 120 wires only, **plus** any small (400-point or mini) breadboard | https://www.amazon.com/Elegoo-EL-CP-004-Multicolored-Breadboard-arduino/dp/B01EV70C78 | The breadboard is needed: see notes. |
| 4 | Long USB cable: ANLINK 10 ft flat USB-A to USB-C | https://www.amazon.com/ANLINK-Charging-Replacement-Security-Cord-White/dp/B0GNLCJDHL | That it carries **data**, not charging only. That your laptop has a USB-A port (otherwise get USB-C to USB-C). |
| 5 | Silica gel: Dry & Dry or similar, 50 × 5 g packets | https://www.amazon.com/Desiccant-Packets-Moisture-Absorber-Dehumidifiers/dp/B01NBQEYTJ | Nothing special. |
| 6 | Separate thermometer with wired probe (fridge/freezer type) | https://www.amazon.com/Urageuxy-Refrigerator-Thermometer-Waterproof-Function/dp/B0D3LNLKFG | That its range goes below freezer temperature and above 50 °C. |
| 7 | Small foam box or insulated lunch bag, zip bags, tape | from home | The box should close with the cable passing out. |

## Notes
- **Why pre-soldered sensors:** GY-521 boards usually ship with loose header pins. Loose pins give unreliable contact, which ruins long recordings.
- **Why the ESP32 with expansion board:** a 38-pin ESP32 is too wide for a small breadboard; the expansion board gives every pin its own header plus power and ground pins.
- **Why a breadboard as well (corrected 16 Nov 2026):** two sensors share the same two data pins (GPIO 21 and 22), and two wire ends cannot plug onto one header pin. A breadboard joins them: one wire from the ESP32 into a breadboard row, then one wire from that row to each sensor. With a single sensor no breadboard is needed.
- **Expansion board voltage jumper:** set it to 3.3 V.
- **Drivers:** boards with a CH340 USB chip may need a driver on Windows or macOS before the port shows up.
- **To start:** items 1, 2, 3 and 4 (about $47). **Before the cold runs:** items 5 and 6 (the separate thermometer is expected by reviewers) and the household items in 7.
