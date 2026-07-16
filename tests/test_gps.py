
"""GPS NMEA parsing tests — pure functions, no hardware."""

import pytest

from cruze.telemetry.gps import _dm_to_decimal, _parse_gga_or_rmc, _parse_endpoint, _parse_nmea_burst

# Standard NMEA reference sentence (Wikipedia/u-blox docs): 22.4 knots SOG.
RMC = "$GPRMC,123519,A,4807.038,N,01131.000,E,022.4,084.4,230394,003.1,W*6A"
GGA = "$GPGGA,123519,4807.038,N,01131.000,E,1,08,0.9,545.4,M,46.9,M,,*47"


def test_rmc_speed_over_ground_knots_to_mps():
    fix = _parse_gga_or_rmc(RMC)
    assert fix is not None
    lat, lon, alt_m, heading_deg, speed_mps = fix
    # 22.4 knots × 0.514444 = 11.5235 m/s
    assert speed_mps == pytest.approx(11.5235, abs=0.001)
    assert heading_deg == pytest.approx(84.4)
    assert lat == pytest.approx(48.1173, abs=0.0001)


def test_rmc_empty_speed_field_gives_none():
    sentence = "$GPRMC,123519,A,4807.038,N,01131.000,E,,084.4,230394,003.1,W*6A"
    fix = _parse_gga_or_rmc(sentence)
    assert fix is not None
    assert fix[4] is None


def test_gga_has_no_speed():
    fix = _parse_gga_or_rmc(GGA)
    assert fix is not None
    assert fix[4] is None
    assert fix[2] == pytest.approx(545.4)  # altitude


def test_garbage_returns_none():
    assert _parse_gga_or_rmc("not nmea") is None
    assert _parse_gga_or_rmc("$GPXTE,A,A,0.67,L,N*6F") is None


def test_dm_to_decimal_south_west_negative():
    assert _dm_to_decimal("4807.038", "S") == pytest.approx(-48.1173, abs=0.0001)
    assert _dm_to_decimal("01131.000", "W") == pytest.approx(-11.5167, abs=0.0001)


def test_parse_endpoint_sim():
    assert _parse_endpoint("sim") == ("sim", "", 0)


def test_parse_endpoint_serial():
    assert _parse_endpoint("COM5") == ("serial", "COM5", 0)
    assert _parse_endpoint("/dev/ttyUSB0") == ("serial", "/dev/ttyUSB0", 0)


def test_parse_endpoint_udp():
    assert _parse_endpoint("udp://0.0.0.0:10110") == ("udp", "0.0.0.0", 10110)


def test_parse_endpoint_tcp():
    assert _parse_endpoint("tcp://192.168.1.50:2947") == ("tcp", "192.168.1.50", 2947)


def test_parse_nmea_burst_prefers_rmc_speed():
    burst = (GGA + "\r\n" + RMC + "\r\n").encode("ascii")
    fix = _parse_nmea_burst(burst)
    assert fix is not None
    assert fix[4] is not None  # RMC speed wins over GGA's None


def test_parse_nmea_burst_garbage_returns_none():
    assert _parse_nmea_burst(b"\xff\xfe garbage\r\n") is None


def test_parse_nmea_burst_gga_only_has_no_speed():
    burst = (GGA + "\r\n").encode("ascii")
    fix = _parse_nmea_burst(burst)
    assert fix is not None
    assert fix[4] is None


def test_parse_nmea_burst_last_rmc_wins():
    # Second RMC reports 10.0 knots; it should override the first (22.4 kn).
    rmc2 = "$GPRMC,123520,A,4807.038,N,01131.000,E,010.0,084.4,230394,003.1,W*6A"
    burst = (RMC + "\r\n" + rmc2 + "\r\n").encode("ascii")
    fix = _parse_nmea_burst(burst)
    assert fix is not None
    assert fix[4] == pytest.approx(10.0 * 0.514444, abs=0.001)
