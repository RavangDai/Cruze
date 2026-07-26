import pathlib
import yaml

from cruze.core.config import PerceptionConfig


def test_vision_net_defaults_off():
    c = PerceptionConfig()
    assert c.autospeed_enabled is False
    assert c.autosteer_enabled is False
    assert c.autodrive_enabled is False
    assert c.onnx_provider == "cpu"
    assert c.autospeed_model_path.endswith("autospeed_fp32.onnx")
    # 0.7, not vision_pilot's 0.6: measured knee on real footage — same recall,
    # ~30% fewer boxes.
    assert c.autospeed_conf_threshold == 0.7
    assert c.autospeed_iou_threshold == 0.45


def test_vision_pilot_profile_enables_all_three():
    data = yaml.safe_load(pathlib.Path("config/hardware/vision_pilot.yaml").read_text())
    p = data["perception"]
    assert p["autospeed_enabled"] is True
    assert p["autosteer_enabled"] is True
    assert p["autodrive_enabled"] is True
