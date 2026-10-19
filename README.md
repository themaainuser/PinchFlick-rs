# PinchFlick (Rust edition — `pinchflick-rs`)

A Rust rewrite of **[PinchFlick v1](../PinchFlick/README.md)** (Python): one
webcam app where hand gestures scroll **anywhere** and seek **YouTube** — same
decision semantics, verified by the same 25 logic checks (identical scenarios
and numbers as v1's `selftest.py`, plus one runner test = 135 green in
`cargo test`).

| Gesture | Action | Where it works |
|---|---|---|
| ☝️ One finger up | scroll **up** | **Everywhere** — blogs, docs, games, any window |
| ✌️ Two fingers up | scroll **down** | **Everywhere** |
| 🤏 Double-pinch **left** half of the view | seek **−10s** | **Only while a YouTube tab is focused** |
| 🤏 Double-pinch **right** half of the view | seek **+10s** | **Only while a YouTube tab is focused** |

Double-pinch = thumb + index finger together, apart, together again quickly
(on the same side of the screen).

> **Status (honest, per the design doc's §9):** v1 remains the daily driver.
> Same-day measurements on the same machine (below) show the Rust build is
> **faster and far smaller**, but v1 still uses **less CPU and RAM** — so this
> repo is an honest write-up with a fully working tool, not a takeover.

## Quick start

**Needs:** Windows 10/11 x64 and a webcam. *(To build: [rustup](https://rustup.rs)
with the MSVC toolchain + Visual Studio Build Tools — "Desktop development
with C++".)*

```powershell
cd pinchflick-rs

# one-time: fetch the three ONNX models (SHA256-verified, see "Models" below)
powershell -ExecutionPolicy Bypass -File scripts\download-models.ps1

cargo build --release

# run from this folder -- the app loads models\ from the current directory
target\release\pinchflick-rs.exe --live
```

- **No injection yet / watching what would happen?** add `--dry-run` — every
  action is printed instead of sent (`--live --dry-run`).
- **Offline integrity check of models already downloaded:**
  `scripts\download-models.ps1 -VerifyOnly` (exit 1 = anything missing/corrupt).

**Before you start:** `target\release\pinchflick-rs.exe --selftest` runs the
25 logic tests with **no camera needed** and must say `ALL GREEN` (exit code
0) — the same checks `cargo test` runs.

### Camera window keys (same as v1)

- `q` / `ESC` — quit (the console's `q` also works; the window's ✕ too)
- `d` — toggle the debug overlay (skeleton, live pinch ratio, FPS, window title)

### The HUD

A small always-on-top window in the top-right corner shows the live camera
feed, a vertical **center line**, your pinch status
(`YouTube: SEEK ON` / `off`, `pinch:` ratio, `fingers up:`), and flashes
**◀ / ▶** on the side every registered pinch was seen — over fullscreen video
too (`always_on_top`), so you're never pinching blind.

## Left vs right — why a double-pinch sometimes "doesn't register"

- The view is split by a vertical **center line** (always drawn on the
  preview). Each pinch belongs to the half where the thumb+index midpoint
  lands — and **both pinches must land on the same half**, or the pair
  doesn't count.
- Pinches within 4% of the line (`tap_side_dead_zone`) are **refused with a
  message** ("Too close to the middle…") instead of being assigned a random
  side that would silently break the pairing.
- Hand tracking gets weaker at the very edges of the frame — pinch where the
  preview clearly shows your whole hand.

## Why seeks only fire on YouTube (and why they're reliable)

Every ~200 ms the app reads the **foreground window's title**. Browsers label
the active tab `… - YouTube`; if the focused window's title carries that
marker (at the end, or before the browser's own name), seek keystrokes are
allowed — otherwise a double-pinch is silently ignored (the HUD shows
`Seek ignored -- not on YouTube`).

The seek itself is YouTube's own keyboard shortcuts (`j`/`l` for 10s, arrow
keys for 5s) sent to that same focused window, so a seek physically cannot
land on the wrong player or fire off YouTube.

Notes:

- Keep the YouTube window focused while seeking. If you click the HUD window,
  seek turns off (its title isn't YouTube) — click back to the video.
- Fullscreen (`F`) keeps the same window title, so seeking keeps working.
- Browsers that append their own name (Brave, Chrome, Firefox) are handled
  automatically: the marker must be at the end or followed only by
  ` - <browser name>`.
- Using something exotic? Press `d` — the debug overlay shows the exact title
  being read — and add its pattern to `youtube_title_markers` in `config.json`.

## No phantom scrolls while pinching

Curling your index finger to pinch could otherwise look like "1 finger up".
Two guards prevent it (identical thresholds to v1):

1. Scroll only counts while the hand is clearly **open**
   (`scroll_open_hand_ratio`).
2. Scrolling is suspended while pinched and for
   `scroll_suppress_after_seek_ms` after every seek.

## Configuration (`config.json`)

The schema is **byte-compatible with v1** — the same 32 keys, defaults, and
validation messages; both apps can share one file.

- Loaded from **next to the exe** if present, else built-in defaults.
- `--config PATH` loads an alternate file instead.
- Invalid values refuse to start with a specific message (e.g.
  `seek_seconds: 12 must be a multiple of 5`).

The full key tables (camera/tracking, scroll, seek, YouTube detection, misc —
defaults and meanings) are in **[v1's README](../PinchFlick/README.md#configuration-configjson)**;
every key, default, and meaning maps over — with these differences:

| Key | In this build |
|---|---|
| `model_path` | Accepted and validated (schema parity), but models always load from `models\` in the current folder |
| `num_hands` | Validated like v1 (1–2), but the runtime tracks **1** hand — v1's shipped default, and v1's controller only consumes the first tracked hand anyway (`controller.py:192`) |
| `show_debug_window` | Works — starts with the debug overlay on (there is no `--debug` flag yet) |
| `always_on_top`, `pyautogui_failsafe` | Work — top-right always-on-top HUD; slam the mouse into a screen corner to stop |

## Calibrating to your hand

1. Run with `d` pressed (or `show_debug_window: true`).
2. Open your hand flat — `pinch:` should sit well above `0.55`. Make a
   deliberate pinch — it should drop below `0.35`.
3. If pinches fire too easily, lower `pinch_on_ratio` (e.g. `0.30`); if they
   never register, raise it. Keep `pinch_off_ratio` ~`0.15`–`0.20` above it.
4. The HUD's tap status line tells you exactly where you are in the
   double-pinch rhythm.

## Commands

```powershell
target\release\pinchflick-rs.exe --selftest        # 25 logic tests, no camera (ALL GREEN / exit 1)
target\release\pinchflick-rs.exe --live            # real mode: gestures inject input
target\release\pinchflick-rs.exe --live --dry-run  # print every action instead of injecting
target\release\pinchflick-rs.exe --camera 1        # override camera index (launches live, like v1)
target\release\pinchflick-rs.exe --config my.json  # alternate config file (also launches live)
```

There is no `--smoke N`/`--debug` flag yet (v1 has them); for a short
unattended run use `--dry-run` and quit with `q`.

## Models — provenance and verification

Three ONNX files in `models\` (14.9 MB total), fetched and hash-checked by
`scripts\download-models.ps1`:

| File | SHA256 |
|---|---|
| `palm_detector.onnx` | `939e53b8c438bdbdcaaf2e5589c8204cb6c92d383a6fe87c939ca3bb52f605c7` |
| `hand_landmark.onnx` | `716f2d75eb41cbb8f222cda9099eb617ec0a3248ecb96e77c0dc916759b5bb1c` |
| `hand_recrop.onnx` | `d71150bd45f3a0060805bda7147302d3d9215a26e7bc270ef840f9718f55d797` |

- **Source:** ONNX conversions of the official Google MediaPipe hand models —
  `takoyakisoft/vtuberkit-vision-landmarkers-onnx` (**Apache-2.0**), on
  [Hugging Face](https://huggingface.co/takoyakisoft/vtuberkit-vision-landmarkers-onnx)
  (the original GitHub repo is gone). The hashes above are the repository's
  LFS object hashes; any download that doesn't match is deleted, never used.
- The app itself does not auto-download; everything works offline afterwards.
- v1's `hand_landmarker.task` is the same upstream MediaPipe models (also
  Apache-2.0), just a different bundle format.
- Third-party code notices: `third_party\nokhwa-bindings-windows\VENDOR-NOTICE.md`.

## Size & speed vs v1 (measured 2026-10-19)

Same machine (Windows, 12 logical cores, 1920×1080 @1.25x), same webcam,
sequential runs, **no hand in frame** (both builds' detector runs at full
cadence), release build / v1's shipped venv:

| | v1 (Python + MediaPipe) | this build (Rust) |
|---|---|---|
| Install size | **349.5 MB** (venv 342.0 + app/model 7.5) | **75.4 MB** (exe 42.8 + DirectML.dll 17.7 + models 14.9) |
| Frame rate | 26.1 fps (summary over 13.0 s) | **30.1 fps steady** (29.5–30.4 across 6 windows; 27.0 fps summary incl. first-paint warmup) |
| Peak RAM | **234 MB** | 267 MB |
| Steady CPU | **0.66 cores** of 12 | 0.84 cores of 12 |
| Decode (frame → RGB) | — | 0.77 ms avg |

fps comes from the `m5_*_smoke.txt` run summaries; CPU/RAM from the paired
`m5_*_smoke_res.txt` sampler runs (method, PID, and interval recorded in
each file). For the representative **interactive** condition (hand actually
in frame, real injection), one live run measured **278 MB peak / 0.95
cores** with a hand visible 32% of frames — `real_scroll_gate_run.txt` +
`real_scroll_gate_res_run.txt` — i.e. a visible hand costs roughly +11 MB
and +0.11 cores over the no-hand condition above.

Reading: the Rust build is **~4.6× smaller** and holds a **full-30-fps**
loop (v1's real-world ~25 fps, measured here at 26.1), while v1 uses **~21%
less CPU and ~12% less RAM**. Under the design doc's §9 rule ("wins on
CPU/RAM *and* parity → v1's README gets a pointer; otherwise an honest
write-up"), v1 stays the daily driver for now. Caveats: the summary column
includes each build's first-paint warmup; CPU/RAM depend on hand visibility
(tracking policy throttles the detector while a hand is locked on), so treat
these as measured indications, not a controlled benchmark.

Two design-doc §2 size targets are missed and recorded as missed (ledger):
exe size **42.8 MB** vs the ≤ 25 MB target (the M0 ort route carries a
17.7 MB DirectML.dll) and steady RAM **267 MB** vs the < 200 MB target.

The perf gate history is in the ledger: M1 bench-measured the stages
(detector 5.67 ms, landmark 3.82 ms per inference); in the live loop the
detector runs at 7–10 ms and landmarks cost 5.3 ms with a hand visible,
holding 29.6–30.5 fps (M3.6 evidence run), and M4 verified 30 fps *with*
the HUD + live camera texture on.

## Troubleshooting

- **Camera won't open** — another app (Teams/Zoom/browser) is using it; or
  raise `camera_index` / `--camera 1`.
- **Build fails with `link.exe` not found** — install VS Build Tools (C++
  workload), or run cargo inside a *Developer Command Prompt* / after
  `vcvars64.bat`.
- **`models\...` not found** — run from the folder that contains `models\`,
  or fetch them: `scripts\download-models.ps1`.
- **Scrolling goes the wrong way** — set `scroll_invert: true`.
- **Scroll too fast/slow** — lower/raise `scroll_speed`, or set
  `scroll_interval_ms` to `50` for a steadier crawl.
- **Seek never fires on YouTube** — check the HUD's `YouTube: SEEK ON` line
  (it requires the *video window* to be focused, not the HUD). Still off?
  Press `d` — the overlay prints the exact window title the app is reading —
  and add a matching marker to `youtube_title_markers`.
- **Pinch ratio never drops low enough** — bring your hand closer, improve
  lighting, and read the live ratio in debug to pick thresholds that fit you.

## Development

- **Tests:** `cargo test` — 144 passing + 1 ignored. The ignored one is the
  M0 `.task`-extraction fixture (`cargo test -- --ignored` runs it where
  `../PinchFlick/hand_landmarker.task` exists — it's absent on a fresh clone,
  hence ignored). The 25 v1 parity checks live in `src/selftest.rs` (their
  single source of truth, also what `--selftest` runs);
  `tests/parity.rs` wraps them as the M2 gate.
- **Dev probes:** `--detect` / `--landmarks` (single-frame model dumps) and
  `--video [N]` (default 3 — the unattended frame loop the gates used) run
  against a still image; they take no config (`--config`/`--camera` imply
  `--live` instead, like v1). The live app is `--live`.
- **CI:** `.github/workflows/ci.yml` runs `cargo test` on `windows-latest`
  for every push/PR (the repo has no remote yet — it activates on first push).
- **Process:** test-driven for pure logic; OS/camera/injection wiring is
  compiler- + human-gated and declared as spikes. Design doc:
  `docs\superpowers\specs\2026-10-15-pinchflick-rs-design.md`;
  append-only work ledger: `.superpowers\sdd\2026-10-15-pinchflick-rs-design\progress.md`.
