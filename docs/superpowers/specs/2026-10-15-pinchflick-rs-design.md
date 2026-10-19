# pinchflick-rs — Rust rewrite of PinchFlick (side project)

Date: 2026-10-15 · Status: draft design, awaiting builder-check
Companion project: `../PinchFlick` (the working Python v1 — remains the daily driver)

## 1. Intent and why

PinchFlick v1 (Python) is feature-complete and verified, but its environment
is 350 MB, and the weight is immovable: mediapipe's own pip metadata hard-
requires `numpy`, `opencv-contrib-python`, `matplotlib`, `sounddevice` — all
verified experimentally (uninstall → import failures → restore, 2026-10-15).
Candidate lighter Python/Rust libraries were probed and rejected with evidence:
`rustynum` (won't even build on Py3.14), `purecv`/`rustcv` (don't exist on
PyPI), `kornia-rs` 0.2.0 (real and zero-dep, but has no camera capture, no
window/GUI, no drawing primitives — only image math).

This side project answers: **how small and self-contained can the same
tool get as one native Rust binary?** Secondary goals: learning value, and a
possible donation of improvements (config schema, gesture math) back to v1.

## 2. Goals

- Single Windows `.exe`, target **≤ 25 MB**, no Python, no runtime installs
  (no DLL zoo if avoidable — pure-Rust inference preferred).
- **Feature parity with v1** (all of it):
  - 1 finger up = scroll up, 2 fingers (index+middle) = scroll down, works in
    any window (wheel events to whatever the mouse hovers).
  - Double-pinch (thumb+index) left half = −10 s, right half = +10 s —
    **only while a YouTube tab is focused**; center dead zone refuses
    ambiguous middle pinches with an on-screen message.
  - Same suppression rules: no scroll while pinched, no scroll
    `scroll_suppress_after_seek_ms` after a seek, scroll requires open hand
    ratio, pinch hysteresis (on/off thresholds), double-tap pairing window,
    **no seek cooldown** (v1 default `seek_cooldown_ms: 0`).
  - Foreground-window-title YouTube detection: marker at title end OR
    followed only by ` - <browser>` (the Brave fix: `… - YouTube - Brave`).
    Seek via keyboard shortcuts (`j`/`l` for 10 s, arrows for 5 s) sent to
    the focused window.
  - Floating always-on-top preview (~480×360, top-right), always-drawn
    center divider, ◀/▶ side flash on tap and seek, status HUD line,
    `d` debug overlay, `q`/`ESC` quit.
  - Same `config.json` schema (all 32 keys) with the same validation
    messages; `model_path` keeps pointing at the `.task` zip both apps
    understand; `pyautogui_failsafe` keeps its meaning as a panic-quit
    (mouse parked in the screen corner = clean exit, same as v1).
- Performance: **30 fps** capture→decision loop at 640×480; inference
  **< 15 ms/frame** on CPU; RAM **< 200 MB**.
- Test parity: all **25 v1 selftests** ported to `cargo test` with the same
  synthetic-landmark scenarios; nothing merges without them green.

## 3. Non-goals and anti-goals

- **Not replacing v1.** PinchFlick.exe stays the daily tool until the Rust
  build proves parity AND wins on resource use. This repo may die quietly;
  that is an acceptable outcome.
- No mouse/cursor control (the old `gesture_mouse_drag.py` idea stays out).
- No multi-hand, no new gestures, no gesture editor UI.
- Windows 10/11 first. Portability is welcome where free (nokhwa is
  cross-platform) but never at the cost of the schedule.
- Anti-goals: no Python anywhere in the runtime path; no build-time C++
  toolchain, vcpkg, or system OpenCV (that dependency class is the enemy);
  no config-format divergence from v1.

## 4. Constraints

- Hardware/OS: the author's Windows 10/11 machine, ordinary USB webcam.
- Model: MediaPipe `hand_landmarker.task` (Apache-2.0 per its model card —
  re-confirm the license file at conversion time). The `.task` file is a
  zip containing TWO models — `hand_detector.tflite` (~2.3 MB, finds the
  hand in the frame) and `hand_landmarks_detector.tflite` (~5.5 MB, 21
  points on a cropped hand). Both get extracted in-memory from the zip at
  startup (a few lines with the `zip` crate); nothing is written to disk.
- Layout: live under `air-mouse-controler\pinchflick-rs\`, sibling of
  `PinchFlick\`, its own git repo when work starts.
- The v1 reference implementation (`gesture_core.py`, dependency-free pure
  math) is the spec for gesture semantics — where the port and the doc
  disagree, v1 behavior wins.

## 5. Architecture (initial calls — all overridable by the builder)

| Concern | Choice | Why / fallback |
|---|---|---|
| Camera capture | **nokhwa** (MSMF on Windows), 640×480 | Actively maintained; fallback: `kamadak`-era `v4l`/win32 GDI capture only if nokhwa misbehaves with the specific webcam (test in M1) |
| NN inference | **tract** (pure-Rust TFLite), both models from the `.task` zip. Tracking policy mirrors MediaPipe's VIDEO mode: detector on the first frame and again whenever landmarks are lost or low-confidence; landmark model on the tracked ROI every frame | Zero DLLs, static binary. **M0 gate:** if tract fails on either model's ops, fall back to **ort** (ONNX Runtime bindings) + community ONNX conversions — accepted cost: ~15–20 MB of runtime DLLs |
| Landmark math / gestures | Straight port of `gesture_core.py` (PinchState hysteresis, DoubleTapSeeker, ScrollDriver, dead-zone, YouTubeDetector suffix rule) | It is dependency-free Python; semantics are already test-proven; port test-by-test |
| Preview window + HUD | **eframe/egui**, viewport flag `always_on_top`, ~480×360 window placed top-right, resized/scaled render | Immediate-mode HUD (flash, status, divider) is trivial; fallback: `minifb` (worse text) or no preview at all (CLI-only mode stays useful, see M3) |
| Input injection | **windows-rs direct `keybd_event`/`mouse_event`** — the exact legacy Win32 calls pyautogui itself makes: wheel scroll + `j`/`l`/arrow keypresses to the focused window | Byte-parity by construction. **M3.6 amendment:** enigo was dropped after source review — its wheel path sends −length×120 (sign-flipped and 120× vs pyautogui's raw `dwData=clicks`), `Key::Unicode('L')` maps to an invalid VK (silent no-op breaking shift-char parity), and its EVENT_MARKER/EXTENDEDKEY usage diverges. Same calls ⇒ same focus semantics |
| Window title read | **windows** crate: `GetForegroundWindow` + `GetWindowTextW`, polled every 200 ms (config `youtube_check_interval_ms`) | Same rule as v1, incl. rstrip + casefold + browser-suffix tolerance |
| Config | `serde_json`, identical 32-key schema + identical validation messages; default file = `config.json` next to the exe (`--config <path>` overrides) | Schema is shared with v1; each app defaults to the file beside itself — point `--config` at one file to truly share |
| Logging | `println!` status lines + `env_logger` behind `--debug` | v1 parity |

Module layout (mirrors v1 so ports are mechanical):

```
src/
  main.rs            camera loop, timing, CLI flags (--debug, --config, --camera)
  config.rs          load/merge/validate (32 keys)
  gestures/
    pinch.rs         PinchState hysteresis
    scroll.rs        ScrollDriver
    seek.rs          DoubleTapSeeker + dead zone + key composition
  track/
    capture.rs       nokhwa wrapper
    inference.rs     tract (or ort) session, both models, frame -> 21 landmarks
  sys/
    title.rs         foreground window title + YouTubeDetector
    input.rs         keybd_event/mouse_event (wheel + keys, pyautogui-parity)
    ui.rs            egui overlay window, flash, HUD
tests/
    parity.rs        the 25 v1 selftests, same scenarios
```

## 6. Milestones (each has a hard gate)

- **M0 — inference spike (the make-or-break).** Extract BOTH `.tflite`s
  from the `.task` zip, run detector then landmark model on one frame
  through tract, print 21 landmarks, time each stage separately.
  PASS = tract route; FAIL on either model = switch to ort+ONNX and
  re-estimate the size budget.
- **M1 — capture + model loop.** nokhwa frames → tracking-policy
  inference → live FPS + landmark console dump. Gate: ≥ 30 fps sustained,
  landmark inference < 15 ms/frame, amortized detector overhead < 5 ms
  (≈ < 20 ms total per frame). **Escalation ladder if the gate is
  missed:** (1) cheap wins — detector on every other frame, smaller
  detector input, tighter ROI carry-forward; (2) accept ≥ 25 fps (v1's
  real-world smoothness) and write the real number down; (3) switch to
  ort+ONNX; (4) never silently drop a goal — final numbers land in the
  README either way.
- **M2 — gesture engine port.** All 25 parity tests green in `cargo test`.
- **M3 — the tool is real.** input injection + window-title gating: scroll
  and seek work CLI-only (no preview yet). Gate: a human double-pinches and
  YouTube actually seeks; scroll works in a browser + a game.
- **M4 — preview/HUD.** egui window, divider, flashes, debug overlay,
  config validation messages.
- **M5 — polish.** README (v1-style), `--selftest`, GitHub Actions
  windows-latest running `cargo test`, size/speed report vs v1.

## 7. Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| tract can't run the tflite graph | medium | M0 exists to find out on day one; ort+ONNX fallback pre-designed (costs DLLs, still ≪ 350 MB) |
| nokhwa flaky with the author's specific webcam | low-medium | Proven in M1; fallback capture backends behind one trait |
| egui always-on-top/DPI quirks over fullscreen video | medium | v1 solved the same problem with raw Win32 habits; test over YouTube fullscreen early in M4 |
| injection semantics differ from pyautogui (per-keypress ordering, failsafe corners) | low | We now call the identical `keybd_event`/`mouse_event` APIs pyautogui calls, with the modifier dance + failsafe predicate ported verbatim; M3 human test is the gate |
| One model conversion licensing/quality issue (ONNX path) | low | Model card says Apache-2.0; verify bundled LICENSE |
| Effort overrun / motivation loss | high (it's a side project!) | M3 (usable CLI-only tool) is deliberately early; abandoning after any milestone leaves a working v1 untouched |

## 8. Testing strategy

- Unit/parity: `cargo test` hosts the 25 ported selftests (synthetic
  landmarks, exact timestamps — same numbers as `selftest.py`).
- Integration: keep v1's `integration_test.py` as a cross-check harness for
  the OS seam (real window title → gate → keystroke). The seam itself
  (real foreground window → title → injected keystroke) stays a **manual
  M3 checklist** — an automated Rust harness that manipulates real
  windows is deliberately OUT of the milestones (side-project budget); it
  can return later as an optional M6. Title-**rule** parsing is fully
  covered by the parity tests with the same synthetic title table.
- Human smoke checklist (M3/M4): fullscreen YouTube seek, brave-suffix
  title, blog page blocked, game-window scroll, dead-zone message, flash
  sides, `q`/`d` keys.

## 9. Relationship to PinchFlick v1

- v1 stays the daily driver and the semantic reference.
- Shared artifact: `config.json` schema (both apps read the same file).
- If the Rust build wins on CPU/RAM and reaches parity, v1's README gets a
  pointer; if not, v1 ships on and this repo becomes an honest write-up.

**Amendment (2026-10-19, owner's decision at closeout):** the measured
numbers gave CPU/RAM to v1 and size/fps to the Rust build, so the rule above
would have kept v1 primary. The owner overrode it — **the Rust build is the
daily driver**; v1 remains the semantic reference; v1's README got the
pointer either way. LICENSE: MIT.

## 10. Open for the builder

Crate version pinning, exact M0 spike procedure (tract version/features),
CI wiring details, and the M0-outcome-dependent size re-estimate are
explicitly left to the builder.

## 11. Builder-check resolutions (2026-10-15)

A fresh-builder review raised five questions; all are resolved above
(north star: v1 behavior wins where the doc was silent):

1. **The `.task` holds two models, not one** → both extracted in-memory;
   tracking policy = MediaPipe-VIDEO-style (detect on start/loss,
   landmarks every frame); perf gate restated per-stage (§4, §5, M0, M1).
2. **Model at runtime** → loaded in-memory from the `model_path` `.task`
   zip (config-swap with v1 preserved); optional M5 embedding via
   `include_bytes!` if a true single-file exe is wanted (+~8 MB, still
   under budget).
3. **Key inventory / failsafe / config path** → 32 keys is the true
   count; `pyautogui_failsafe` = corner-quit parity; default config
   beside the exe with `--config` override.
4. **Tract runs but misses the perf gate** → defined escalation ladder
   at M1 (optimize → accept v1-parity 25 fps → ort → document honestly).
5. **Automated OS-seam harness** → out of the milestones (manual M3
   checklist + synthetic-title parity tests); optional M6 later.

## 12. M0 results (2026-10-15)

**Verdict: tract FAILS — the pre-designed ort + ONNX fallback (§5, §11.4)
takes over.**

Both models in `hand_landmarker.task` store weights as float16 constants
consumed by `DEQUANTIZE` ops. tract 0.23.8 rejects that op at proto
translation — "Unsupported: builtin_code: DEQUANTIZE … inputs: F16" — so
neither model even finishes loading, before any inference runs. Confirmed
independently for `hand_detector.tflite` (opcode DEQUANTIZE, F16 blob) and
`hand_landmarks_detector.tflite` (same op, same shape of failure).

What the spike proved and stays reusable:

- in-memory extraction of both `.tflite`s from the `.task` zip (3/3 unit
  tests green: archive extraction, nearest resize, [-1,1] normalization)
- the load → shape-probe → warmup → timed-runs harness in `src/main.rs`
- a fully static tract build weighs **13.8 MB** — proof the size thesis
  holds even though this engine can't run these two models

Consequences:

- **M1 re-scoped:** obtain ONNX conversions of both models (community
  conversions of the palm-detection and hand-landmark graphs exist; their
  validation is the next spike's job), run them via `ort` — accepting the
  ~15–20 MB onnxruntime DLLs, still >15× under the Python stack. Timing
  budget re-measured there.
- Build note for this machine: cargo needs the `vcvars64.bat` wrapper
  (VS instance registration never became visible to vswhere; vcvars sets
  INCLUDE/LIB/PATH directly). A zombie Dec-2025 BuildTools install — one
  whose Windows SDK had been silently skipped — blocked every repair until
  it was uninstalled and reinstalled clean with VCTools + Win11SDK 26100.

## 13. M1 results (2026-10-15) � ort route PASSES

Models: `takoyakisoft/vtuberkit-vision-landmarkers-onnx` (converted from the
official MediaPipe `.task` bundles; SHA256 verified against the model card).
Runtime: `ort` 2.0.0-rc.14 + onnxruntime, CPU, this machine.

| stage | input | outputs | avg | min |
|---|---|---|---|---|
| palm_detector | [1,192,192,3] | [1,2016,18] anchors, [1,2016,1] scores | **5.67 ms** | 4.92 ms |
| hand_landmark | [1,224,224,3] | [1,63] landmarks, [1,1] flag, [1,1] score, [1,63] world | **3.82 ms** | 3.17 ms |
| hand_recrop | [1,256,256,3] | [1,1,1,4] bbox | (metadata probe only) | � |

Combined ~**9.5 ms/frame** on CPU � inside the 15 ms budget with headroom
for the detect-every-N tracking policy. **M1 gate: PASS.** Sanity: a
no-hand center crop scored flag=0.0017 (correct rejection); session loads
take ~35 ms each, once at startup. onnxruntime DLL size measured at M5.

Deferred to M2 as scoped: palm-anchor decode ([1,2016,18] -> boxes),
crop-affine for the landmark model, presence gating, tracking policy, and
the gesture-engine port with its 25 parity tests.
