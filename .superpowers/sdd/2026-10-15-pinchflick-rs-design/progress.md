# SDD ledger — plan: docs/superpowers/specs/2026-10-15-pinchflick-rs-design.md

Pre-flight: single active task (M0) cut from design doc section 6; no cross-task interfaces to scan.
Ruling: implementation on new repo's master directly — solo side project, no shared branch; consent implicit in 'go implement it'.
Ruling: TDD applies in full to deterministic pure code (zip extraction, resize, normalization); the tract inference experiment itself is the design doc's declared spike (M0 gate) whose outcome IS the test — documented, not skipped.

Task M0: complete — TDD cycle RED (3x not-implemented) -> GREEN (3/3) on pure parts (in-memory .task zip extraction, nearest resize, [-1,1] normalize); experiment main per the declared spike exception.
M0 RESULT: tract 0.23.8 FAILS both hand_detector.tflite and hand_landmarks_detector.tflite at proto translation — unsupported DEQUANTIZE on F16 constants; no inference ran. Exe (static tract, release): see commit message.
Ruling: per design doc section 5 / 11.4 the verdict routes M1 to ort + ONNX conversions (accepted onnxruntime DLL cost); tract abandoned for these models. Reusable skeleton kept: zip extraction, frame prep, timing harness.
Env ruling: all cargo commands run through the vcvars64.bat wrapper — VS instance registration never became visible to vswhere, vcvars sets INCLUDE/LIB/PATH directly. Toolchain repaired today after unblocking a zombie Dec-2025 BuildTools instance (missing SDK) that made the installer no-op.
