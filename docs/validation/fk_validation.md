# FK Cross-Validation — Kinova Gen3 Lite

**Sources compared:**
- Source A: MuJoCo `ee_site` world position (`gen3_lite.xml`)
- Source B: kinpy `tool_frame` transform (`gen3_lite_resolved.urdf`)

**Acceptance criterion:** position error < 1 mm

| Config | MuJoCo ee [x,y,z] m | kinpy ee [x,y,z] m | Error (mm) |
|--------|---------------------|--------------------|:----------:|
| zero | [0.0570, -0.0100, 1.0032] | [0.0570, -0.0100, 1.0032] | 0.001 |
| rand1 | [-0.3799, 0.3216, 0.6896] | [-0.3799, 0.3216, 0.6896] | 0.000 |
| rand2 | [0.0091, 0.5296, 0.6157] | [0.0091, 0.5296, 0.6157] | 0.000 |
| rand3 | [0.1543, 0.2293, 0.9079] | [0.1543, 0.2293, 0.9079] | 0.001 |
| rand4 | [0.5574, -0.0887, 0.6873] | [0.5574, -0.0887, 0.6873] | 0.000 |
| rand5 | [-0.6631, 0.0157, 0.5283] | [-0.6631, 0.0157, 0.5283] | 0.000 |

**Max error: 0.001 mm** — PASS ✓

## Status
- [x] FK validated — MuJoCo ee_site matches URDF kinpy FK
