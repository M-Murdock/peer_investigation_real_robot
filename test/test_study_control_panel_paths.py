import json

from kinova_gen3_lite_rl_bridge.study_control_panel import _parse_pipewire_sources, _session_paths

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
    session_dir, bag_path, audio_path = _session_paths('/sessions', '42', '20260101_120000')
    assert session_dir == '/sessions/42_20260101_120000'
    assert bag_path == '/sessions/42_20260101_120000/rosbag'
    assert audio_path == '/sessions/42_20260101_120000/audio.wav'


def test_bag_path_is_not_the_session_dir_itself():
    # ros2 bag record creates its -o target itself, so it must be a path
    # that doesn't already exist -- it can't be session_dir, which this
    # handler creates up front so audio.wav has somewhere to live.
    session_dir, bag_path, _ = _session_paths('/sessions', '7', '20260101_120000')
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
