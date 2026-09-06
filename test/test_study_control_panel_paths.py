import json

from kinova_gen3_lite_rl_bridge.study_control_panel import (
    _is_capture_device,
    _parse_pipewire_sources,
    _parse_udevadm_properties,
    _session_paths,
    _video_device_label,
)

# Real `udevadm info --query=property` output for a Logitech BRIO webcam --
# it exposes multiple /dev/videoN nodes per physical camera, only some of
# which report the 'capture' capability (the rest are metadata/other nodes).
UDEVADM_CAPTURE_NODE = """DEVPATH=/devices/pci0000:00/.../video4linux/video4
SUBSYSTEM=video4linux
ID_V4L_PRODUCT=Logitech BRIO
ID_V4L_CAPABILITIES=:capture:
"""

UDEVADM_NON_CAPTURE_NODE = """DEVPATH=/devices/pci0000:00/.../video4linux/video5
SUBSYSTEM=video4linux
ID_V4L_PRODUCT=Logitech BRIO
ID_V4L_CAPABILITIES=:
"""

# Trimmed real `pw-dump` output (a mix of Audio/Sink and Audio/Source
# nodes) -- sinks (speakers/HDMI outputs) must be excluded from the
# recording-device picker; only sources (microphones) are usable there.
PW_DUMP_OUTPUT = json.dumps([
    {
        'id': 58, 'type': 'PipeWire:Interface:Node',
        'info': {'props': {
            'media.class': 'Audio/Sink',
            'node.name': 'alsa_output.pci-...HiFi__hw_sofhdadsp__sink',
            'node.description': 'Tiger Lake-LP ... Speaker + Headphones',
        }},
    },
    {
        'id': 59, 'type': 'PipeWire:Interface:Node',
        'info': {'props': {
            'media.class': 'Audio/Source',
            'node.name': 'alsa_input.pci-...HiFi__hw_sofhdadsp__source',
            'node.description': (
                'Tiger Lake-LP Smart Sound Technology Audio Controller '
                'Headset Mono Microphone + Headphones Stereo Microphone'
            ),
        }},
    },
    {
        'id': 100, 'type': 'PipeWire:Interface:Node',
        'info': {'props': {
            'media.class': 'Audio/Source',
            'node.name': 'alsa_input.usb-Generic_Blue_Microphones_2137BAH07JU8-00.analog-stereo.4',
            'node.description': 'Blue Microphones Analog Stereo',
        }},
    },
])


def test_session_paths_are_siblings_under_one_session_dir():
    session_dir, bag_path, audio_path, video_path = _session_paths(
        '/sessions', '42', '20260101_120000')
    assert session_dir == '/sessions/42_20260101_120000'
    assert bag_path == '/sessions/42_20260101_120000/rosbag'
    assert audio_path == '/sessions/42_20260101_120000/audio.wav'
    assert video_path == '/sessions/42_20260101_120000/video.mp4'


def test_bag_path_is_not_the_session_dir_itself():
    # ros2 bag record creates its -o target itself, so it must be a path
    # that doesn't already exist -- it can't be session_dir, which this
    # handler creates up front so audio.wav/video.mp4 have somewhere to live.
    session_dir, bag_path, _, _ = _session_paths('/sessions', '7', '20260101_120000')
    assert bag_path != session_dir


def test_parse_pipewire_sources_always_includes_default_first():
    devices = _parse_pipewire_sources('[]')
    assert devices == [{'id': 'default', 'label': 'default'}]


def test_parse_pipewire_sources_falls_back_to_default_on_bad_json():
    devices = _parse_pipewire_sources('not json')
    assert devices == [{'id': 'default', 'label': 'default'}]


def test_parse_pipewire_sources_excludes_sinks():
    devices = _parse_pipewire_sources(PW_DUMP_OUTPUT)
    ids = [d['id'] for d in devices]
    assert 'alsa_output.pci-...HiFi__hw_sofhdadsp__sink' not in ids


def test_parse_pipewire_sources_extracts_each_source_by_stable_node_name():
    devices = _parse_pipewire_sources(PW_DUMP_OUTPUT)
    ids = [d['id'] for d in devices]
    assert ids == [
        'default',
        'alsa_input.pci-...HiFi__hw_sofhdadsp__source',
        'alsa_input.usb-Generic_Blue_Microphones_2137BAH07JU8-00.analog-stereo.4',
    ]


def test_parse_pipewire_sources_labels_are_human_readable():
    devices = _parse_pipewire_sources(PW_DUMP_OUTPUT)
    usb_mic = next(
        d for d in devices
        if d['id'] == 'alsa_input.usb-Generic_Blue_Microphones_2137BAH07JU8-00.analog-stereo.4')
    assert usb_mic['label'] == 'Blue Microphones Analog Stereo'


def test_parse_udevadm_properties_extracts_key_value_pairs():
    props = _parse_udevadm_properties(UDEVADM_CAPTURE_NODE)
    assert props['ID_V4L_PRODUCT'] == 'Logitech BRIO'
    assert props['ID_V4L_CAPABILITIES'] == ':capture:'


def test_is_capture_device_true_for_capture_node():
    props = _parse_udevadm_properties(UDEVADM_CAPTURE_NODE)
    assert _is_capture_device(props) is True


def test_is_capture_device_false_for_metadata_node():
    # A UVC webcam exposes more than one /dev/videoN node per physical
    # camera -- only the one reporting 'capture' is actually usable.
    props = _parse_udevadm_properties(UDEVADM_NON_CAPTURE_NODE)
    assert _is_capture_device(props) is False


def test_is_capture_device_false_when_property_missing():
    assert _is_capture_device({}) is False


def test_video_device_label_combines_product_and_node():
    props = _parse_udevadm_properties(UDEVADM_CAPTURE_NODE)
    assert _video_device_label('/dev/video4', props) == 'Logitech BRIO (/dev/video4)'
