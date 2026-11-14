# Honest error bars for temperature calibration of a low-cost motion sensor

**Status: no real measurements yet.** Everything in `results/simulated/` and `figures/simulated/` comes from invented data and exists only to test the code. It is stamped as such and must never be reported as a result.

## The question
Low-cost motion sensors (accelerometers and gyroscopes) drift as their temperature changes. The usual fix is to calibrate the drift over some temperature range and subtract it. Any uncertainty quoted for the corrected reading is then trusted at whatever temperature the sensor later meets.

**When a sensor is used outside the temperature range it was calibrated over, do its stated error bars still hold? And can a method tell you when they don't?**

## The idea in one paragraph
A sensor sitting still has a known true reading: zero rotation, and gravity's constant magnitude. So its error can be measured directly while its temperature is swept from a freezer to a warm box. Calibration models are fitted on room-temperature data only, wrapped in conformal prediction to give 90 % error bars, and then scored on colder and warmer data from separate recordings. A second method, weighted conformal prediction, is tested for its ability to *refuse*: to return "no valid interval" at temperatures with no calibration support.

## Layout
```
firmware/thermcal_logger/   Arduino sketch for an MPU6050 on an ESP32 or Arduino Uno
tools/log_serial.py         records the board's output to data/raw/ on your laptop
tools/check_recording.py    checks a recording (timing, gravity, stillness, stuck channels)
src/thermcal/               loader, models, conformal methods, evaluation, simulator
scripts/run_analysis.py     full analysis -> results/ and figures/
scripts/make_simulated_data.py   FAKE data for testing the pipeline
docs/EXPERIMENT_PLAN.md     parts, wiring, protocol, safety, sanity checks  <- start here
docs/PREREGISTRATION.md     hypotheses and analysis choices, fixed before data
docs/JOURNAL_NOTES.md       fit to Review of Scientific Instruments, and what is still missing
```

## Quick start
```bash
pip install -r requirements.txt
python -m pytest -q                              # 8 tests (2 need g++ and Linux/macOS)
python scripts/make_simulated_data.py            # FAKE data
python scripts/run_analysis.py --simulated       # test the pipeline (~4 min)
# after you have recordings in data/raw/:
python scripts/run_analysis.py
```

## What has and hasn't been checked
- **Firmware:** compiles for Arduino Uno and for ESP32 (checked with arduino-cli). Its source also runs on a PC against a fake sensor chip (`tests/host_mock`), and a test checks that its output matches the fake values exactly: averaging, standard deviations, one line per second, two sensors. It has **not been run on real hardware**, so real wiring, I2C timing and the chip itself are untested. Do the sanity checks in the experiment plan.
- **Logger:** tested through a pseudo serial port with the firmware's output: it records every line, ignores a cut-off line, and refuses to overwrite a recording. Not tested with a real USB port.
- **Analysis:** 8 automated tests pass in total; the full pipeline runs on simulated recordings in the logger's file format. On simulated data, coverage inside the calibration range is 0.88–0.92 for a 0.90 target, which shows the code is wired correctly. It says nothing about real sensors.

## Methods in brief
- **Targets:** gyroscope x, y, z (true value 0 when still) and the accelerometer magnitude minus 1 g (true value constant). Single accelerometer axes are not used, because a 1° change in the tilt of the box shifts them by 17 mg.
- **Models:** linear, quadratic and cubic polynomials in temperature (the standard practice), a quadratic plus a heating-rate term (for thermal lag), and boosted trees.
- **Error bars:** split conformal prediction, with calibration errors pooled by cross-validation over whole recordings, because every power-on has its own slow drift.
- **Evaluation:** leave one recording out, with 95 % ranges from resampling whole recordings; coverage reported separately inside, colder than and warmer than the calibration range; a restricted (room-temperature) calibration is compared with a full-range one.
- **Weighted conformal prediction** (Tibshirani et al. 2019) is run as a second method. It corrects for a different mix of temperatures within the calibrated range and returns an infinite interval where there is no calibration data.

## AI-assistance disclosure
The firmware, logger, analysis code and these documents were written by an AI coding assistant at the student's request. The measurements, their interpretation and the final paper are the student's responsibility. Any submission must describe this accurately and follow the journal's policy on AI tools.
