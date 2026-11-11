// Minimal stand-ins for the Arduino core, so the real firmware source can be
// compiled and run on a PC against a fake MPU6050. Used only by tests.
#pragma once
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#define F(x) (x)
#define HEX 16
extern uint64_t fake_us;
inline uint32_t micros() { fake_us += 2500; return (uint32_t)fake_us; }   // time advances on every call
inline uint32_t millis() { return (uint32_t)(fake_us / 1000); }
inline void delay(unsigned long ms) { fake_us += (uint64_t)ms * 1000; }
struct SerialMock {
  void begin(long) {}
  void print(const char* s) { fputs(s, stdout); }
  void print(char c) { fputc(c, stdout); }
  void print(uint32_t v) { printf("%lu", (unsigned long)v); }
  void print(uint16_t v) { printf("%u", (unsigned)v); }
  void print(int v, int base) { if (base == 16) printf("%X", v); else printf("%d", v); }
  void print(double v, int digits) { printf("%.*f", digits, v); }
  void println(const char* s) { puts(s); }
  void println() { fputc('\n', stdout); }
};
extern SerialMock Serial;
