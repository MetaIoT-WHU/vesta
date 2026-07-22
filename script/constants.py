"""Shared GNSS constants for the interference-cancellation."""

import math


class GnssConstants:
    # Carrier wavelengths (m)
    WAVE_LENGTH_B1 = 0.19203948611111112  # BeiDou B1
    WAVE_LENGTH_L1 = 0.1902936727953304   # GPS L1
    WAVE_LENGTH_E1 = 0.1902936727953304   # Galileo E1

    # Radians per meter: 2*pi / wavelength
    M2R_B1 = 2 * math.pi / WAVE_LENGTH_B1
    M2R_L1 = 2 * math.pi / WAVE_LENGTH_L1
    M2R_E1 = 2 * math.pi / WAVE_LENGTH_E1

    # Plot color for demo SVGs
    COLOR_BLUE = [0 / 255, 122 / 255, 161 / 255]

    # Band ids used when parsing satinfo
    FREQID_B1_ID = 0
    FREQID_L1_ID = 0
    FREQID_E1_ID = 2
