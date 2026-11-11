#include "Arduino.h"
#include "Wire.h"
uint64_t fake_us = 0; SerialMock Serial; WireMock Wire;
#include "../../firmware/thermcal_logger/thermcal_logger.ino"
int main(int argc, char** argv) {
  int seconds = argc > 1 ? atoi(argv[1]) : 5;
  setup();
  uint64_t end = fake_us + (uint64_t)seconds * 1000000ULL + 500000ULL;
  while (fake_us < end) loop();
  return 0;
}
