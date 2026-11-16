// Fake I2C bus with one or two fake MPU6050 chips. Each chip returns known raw
// values (big-endian, in the chip's register order) plus a deterministic +-1 wobble.
#pragma once
#include "Arduino.h"
struct WireMock {
  uint8_t addr = 0, reg = 0, pos = 0; bool has_reg = false; uint8_t buf[14]; uint32_t reads = 0;
  bool present(uint8_t a) { return a == 0x68 || a == 0x69; }
  void begin() {}
  void begin(int, int) {}
  void setClock(long) {}
  void beginTransmission(uint8_t a) { addr = a; has_reg = false; }
  void write(uint8_t v) { if (!has_reg) { reg = v; has_reg = true; } }
  uint8_t endTransmission(bool = true) { return present(addr) ? 0 : 2; }
  uint8_t requestFrom(int a, int n) {
    if (!present((uint8_t)a) || reg != 0x3B || n != 14) return 0;
    // ax ay az temp gx gy gz ; chip 0x69 is offset by +1000 on every channel
    int16_t base[7] = {100, -200, 16384, -521, 10, -20, 30};
    int16_t wob = (int16_t)((reads / 2) % 3) - 1;            // -1, 0, +1 repeating; mean 0
    int16_t off = (a == 0x69) ? 1000 : 0;
    for (int i = 0; i < 7; i++) { int16_t v = base[i] + off + wob; buf[2 * i] = (uint8_t)(v >> 8); buf[2 * i + 1] = (uint8_t)(v & 0xFF); }
    reads++; pos = 0; return 14;
  }
  uint8_t read() { return buf[pos++]; }
};
extern WireMock Wire;
