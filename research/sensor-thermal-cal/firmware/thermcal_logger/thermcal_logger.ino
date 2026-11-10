// thermcal_logger: log averaged raw MPU6050 readings (accelerometer, gyroscope,
// on-chip temperature) once per second over USB serial, for thermal calibration.
//
// Works with one or two MPU6050 boards on the same I2C bus:
//   board 1: AD0 pin left unconnected or tied to GND  -> address 0x68
//   board 2: AD0 pin tied to 3.3 V                    -> address 0x69
//
// Wiring (ESP32 dev board):  VCC -> 3V3, GND -> GND, SDA -> GPIO 21, SCL -> GPIO 22
// Wiring (Arduino Uno/Nano): VCC -> 5V (GY-521 boards have a regulator), GND -> GND,
//                            SDA -> A4, SCL -> A5
//
// Output: one CSV line per sensor per second (115200 baud):
//   ms,addr,n,ax,ay,az,gx,gy,gz,temp,sax,say,saz,sgx,sgy,sgz
// ax..gz, temp = MEAN of n raw readings (LSB); s.. = their standard deviations.
// Conversions are done later in Python, so this file stays simple:
//   accel [g] = raw / 16384   (range +-2 g)
//   gyro [deg/s] = raw / 131  (range +-250 deg/s)
//   temperature [degC] = raw / 340 + 36.53   (MPU6050 datasheet formula)
//
// NOTE: written without access to hardware. Check the first lines of output
// against docs/EXPERIMENT_PLAN.md ("Sanity checks") before trusting any data.

#include <Wire.h>

const uint8_t ADDRS[2] = {0x68, 0x69};
bool present[2] = {false, false};

const uint16_t SAMPLE_HZ = 100;      // readings per second, per sensor
const uint16_t N_AVG = 100;          // readings averaged per output line (1 s)

int64_t sum[2][7];                   // ax ay az temp gx gy gz
int64_t sumsq[2][7];
uint16_t count = 0;
uint32_t nextSampleUs = 0;

void writeReg(uint8_t addr, uint8_t reg, uint8_t val) {
  Wire.beginTransmission(addr);
  Wire.write(reg);
  Wire.write(val);
  Wire.endTransmission();
}

bool readRaw(uint8_t addr, int16_t out[7]) {
  Wire.beginTransmission(addr);
  Wire.write(0x3B);                                  // ACCEL_XOUT_H
  if (Wire.endTransmission(false) != 0) return false;
  if (Wire.requestFrom((int)addr, 14) != 14) return false;
  for (uint8_t i = 0; i < 7; i++) {
    uint8_t hi = Wire.read();
    uint8_t lo = Wire.read();
    out[i] = (int16_t)((hi << 8) | lo);              // big-endian
  }
  return true;
}

bool initSensor(uint8_t addr) {
  Wire.beginTransmission(addr);
  if (Wire.endTransmission() != 0) return false;     // nothing answers at this address
  writeReg(addr, 0x6B, 0x01);                        // PWR_MGMT_1: wake up, clock from gyro X
  delay(100);
  writeReg(addr, 0x1A, 0x03);                        // CONFIG: low-pass filter ~44 Hz
  writeReg(addr, 0x1B, 0x00);                        // GYRO_CONFIG: +-250 deg/s
  writeReg(addr, 0x1C, 0x00);                        // ACCEL_CONFIG: +-2 g
  return true;
}

void resetSums() {
  for (uint8_t s = 0; s < 2; s++)
    for (uint8_t i = 0; i < 7; i++) { sum[s][i] = 0; sumsq[s][i] = 0; }
  count = 0;
}

void setup() {
  Serial.begin(115200);
  delay(500);
#if defined(ESP32)
  Wire.begin(21, 22);
#else
  Wire.begin();
#endif
  Wire.setClock(400000);
  Serial.println(F("# thermcal_logger v1"));
  for (uint8_t s = 0; s < 2; s++) {
    present[s] = initSensor(ADDRS[s]);
    Serial.print(F("# sensor 0x")); Serial.print(ADDRS[s], HEX);
    Serial.println(present[s] ? F(" found") : F(" not found"));
  }
  Serial.println(F("ms,addr,n,ax,ay,az,gx,gy,gz,temp,sax,say,saz,sgx,sgy,sgz"));
  resetSums();
  nextSampleUs = micros();
}

void printStat(uint8_t s, uint8_t i, bool wantStd) {
  double mean = (double)sum[s][i] / count;
  if (!wantStd) { Serial.print(mean, 2); return; }
  double var = (double)sumsq[s][i] / count - mean * mean;
  Serial.print(var > 0 ? sqrt(var) : 0.0, 2);
}

void loop() {
  if ((int32_t)(micros() - nextSampleUs) < 0) return;
  nextSampleUs += 1000000UL / SAMPLE_HZ;

  int16_t r[7];
  bool ok = true;
  int16_t buf[2][7];
  for (uint8_t s = 0; s < 2; s++) {
    if (!present[s]) continue;
    if (!readRaw(ADDRS[s], r)) { ok = false; break; }
    for (uint8_t i = 0; i < 7; i++) buf[s][i] = r[i];
  }
  if (!ok) return;                                   // skip this instant for all sensors
  for (uint8_t s = 0; s < 2; s++) {
    if (!present[s]) continue;
    for (uint8_t i = 0; i < 7; i++) {
      sum[s][i] += buf[s][i];
      sumsq[s][i] += (int32_t)buf[s][i] * (int32_t)buf[s][i];
    }
  }
  count++;

  if (count >= N_AVG) {
    uint32_t ms = millis();
    // raw order in the chip: ax ay az temp gx gy gz  -> print as ax ay az gx gy gz temp
    const uint8_t ORDER[7] = {0, 1, 2, 4, 5, 6, 3};
    for (uint8_t s = 0; s < 2; s++) {
      if (!present[s]) continue;
      Serial.print(ms); Serial.print(F(",0x")); Serial.print(ADDRS[s], HEX);
      Serial.print(','); Serial.print(count);
      for (uint8_t k = 0; k < 7; k++) { Serial.print(','); printStat(s, ORDER[k], false); }
      for (uint8_t k = 0; k < 6; k++) { Serial.print(','); printStat(s, ORDER[k], true); }
      Serial.println();
    }
    resetSums();
  }
}
