//! Input injection — Windows keyboard + mouse wheel, pyautogui-parity.
//!
//! v1 parity target: pyautogui on Windows as called from `PinchFlick/controller.py`
//! (`pyautogui.scroll(...)` and `pyautogui.press(key)` per seek key).
//!
//! Why not enigo (the original design-doc plan): enigo's wheel sends
//! `length * -120` wheel deltas — sign-flipped and 120x stronger than v1's raw
//! one-click `dwData` — and `Key::Unicode('L')` maps to an invalid virtual key
//! that silently does nothing, breaking uppercase shift-character parity. We
//! therefore call the same legacy Win32 APIs pyautogui itself uses
//! (`keybd_event`, `mouse_event`, `VkKeyScanA`) and reproduce its event
//! choreography exactly, which keeps focus semantics identical to v1.
//!
//! pyautogui references (`PinchFlick/.venv/Lib/site-packages/pyautogui/`):
//! - `_pyautogui_win.py:113-235` — named key -> VK table (121 entries)
//! - `_pyautogui_win.py:244-247` — printable ASCII 32..=127 filled LAST via
//!   `VkKeyScanA` (so it overrides single-char named entries such as `' '`)
//! - `_pyautogui_win.py:250-292` — `_keyDown`: mod DOWN (alt, ctrl, shift),
//!   key DOWN, mod UP (shift, ctrl, alt)
//! - `_pyautogui_win.py:295-332` — `_keyUp`: mod DOWN (alt, ctrl, shift — yes,
//!   keydown again), key UP, mod UP (shift, ctrl, alt)
//! - `_pyautogui_win.py:507-539` — `_scroll`: `mouse_event(MOUSEEVENTF_WHEEL,
//!   x, y, clicks, 0)` with raw click count in dwData (no *120)
//! - `__init__.py:526` — `isShiftCharacter`: `character.isupper() or character
//!   in set('~!@#$%^&*()_+{}|:"<>?')`
//! - `__init__.py:1582-1616` — `press`: lowercases the key only when
//!   `len(keys) > 1` BEFORE `_keyDown`; the 50ms inter-key sleep lives in the
//!   caller (`controller.py:211-224`), not here
//! - `__init__.py:569-573, 1732, 2169-2171` — FAILSAFE: cursor in any of the 4
//!   exact screen corners raises before an action is sent

use windows::Win32::Foundation::POINT;
use windows::Win32::UI::HiDpi::GetDpiForSystem;
use windows::Win32::UI::Input::KeyboardAndMouse::{
    keybd_event, mouse_event, VkKeyScanA, KEYBD_EVENT_FLAGS, KEYEVENTF_KEYUP, MOUSEEVENTF_WHEEL,
};
use windows::Win32::UI::WindowsAndMessaging::{
    GetCursorPos, GetSystemMetrics, SM_CXSCREEN, SM_CYSCREEN,
};

/// One `keybd_event`: virtual-key code + direction. `down: true` maps to
/// `KEYBD_EVENT_FLAGS(0)` (pyautogui has no KEYDOWN constant), `false` to
/// `KEYEVENTF_KEYUP`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KeyEvent {
    pub vk: u16,
    pub down: bool,
}

/// Named key -> virtual key, from pyautogui's `keyboardMapping` update dict
/// (`_pyautogui_win.py:113-235`, 121 entries, verbatim). Case-sensitive and
/// lowercase-only, exactly like the Python dict: `named_vk("LEFT")` is `None`
/// (`press()` lowercases multi-char names before lookup — that lives in
/// [`key_plan`]). Single printable chars are NOT resolved here; pyautogui
/// overwrites them with `VkKeyScanA` results last.
pub fn named_vk(name: &str) -> Option<u16> {
    let vk = match name {
        "backspace" | "\u{8}" => 0x08,
        "super" => 0x5B,
        "tab" | "\t" => 0x09,
        "clear" => 0x0C,
        "enter" | "\n" | "return" => 0x0D,
        "shift" => 0x10,
        "ctrl" => 0x11,
        "alt" => 0x12,
        "pause" => 0x13,
        "capslock" => 0x14,
        "kana" | "hanguel" | "hangul" => 0x15,
        "junja" => 0x17,
        "final" => 0x18,
        "hanja" | "kanji" => 0x19,
        "esc" | "escape" => 0x1B,
        "convert" => 0x1C,
        "nonconvert" => 0x1D,
        "accept" => 0x1E,
        "modechange" => 0x1F,
        " " | "space" => 0x20,
        "pgup" | "pageup" => 0x21,
        "pgdn" | "pagedown" => 0x22,
        "end" => 0x23,
        "home" => 0x24,
        "left" => 0x25,
        "up" => 0x26,
        "right" => 0x27,
        "down" => 0x28,
        "select" => 0x29,
        "print" => 0x2A,
        "execute" => 0x2B,
        "prtsc" | "prtscr" | "prntscrn" | "printscreen" => 0x2C,
        "insert" => 0x2D,
        "del" | "delete" => 0x2E,
        "help" => 0x2F,
        "win" | "winleft" => 0x5B,
        "winright" => 0x5C,
        "apps" => 0x5D,
        "sleep" => 0x5F,
        "num0" => 0x60,
        "num1" => 0x61,
        "num2" => 0x62,
        "num3" => 0x63,
        "num4" => 0x64,
        "num5" => 0x65,
        "num6" => 0x66,
        "num7" => 0x67,
        "num8" => 0x68,
        "num9" => 0x69,
        "multiply" => 0x6A,
        "add" => 0x6B,
        "separator" => 0x6C,
        "subtract" => 0x6D,
        "decimal" => 0x6E,
        "divide" => 0x6F,
        "f1" => 0x70,
        "f2" => 0x71,
        "f3" => 0x72,
        "f4" => 0x73,
        "f5" => 0x74,
        "f6" => 0x75,
        "f7" => 0x76,
        "f8" => 0x77,
        "f9" => 0x78,
        "f10" => 0x79,
        "f11" => 0x7A,
        "f12" => 0x7B,
        "f13" => 0x7C,
        "f14" => 0x7D,
        "f15" => 0x7E,
        "f16" => 0x7F,
        "f17" => 0x80,
        "f18" => 0x81,
        "f19" => 0x82,
        "f20" => 0x83,
        "f21" => 0x84,
        "f22" => 0x85,
        "f23" => 0x86,
        "f24" => 0x87,
        "numlock" => 0x90,
        "scrolllock" => 0x91,
        "shiftleft" => 0xA0,
        "shiftright" => 0xA1,
        "ctrlleft" => 0xA2,
        "ctrlright" => 0xA3,
        "altleft" => 0xA4,
        "altright" => 0xA5,
        "browserback" => 0xA6,
        "browserforward" => 0xA7,
        "browserrefresh" => 0xA8,
        "browserstop" => 0xA9,
        "browsersearch" => 0xAA,
        "browserfavorites" => 0xAB,
        "browserhome" => 0xAC,
        "volumemute" => 0xAD,
        "volumedown" => 0xAE,
        "volumeup" => 0xAF,
        "nexttrack" => 0xB0,
        "prevtrack" => 0xB1,
        "stop" => 0xB2,
        "playpause" => 0xB3,
        "launchmail" => 0xB4,
        "launchmediaselect" => 0xB5,
        "launchapp1" => 0xB6,
        "launchapp2" => 0xB7,
        _ => return None,
    };
    Some(vk)
}

/// Python-exact `str.isupper()` + shift-symbol membership
/// (`__init__.py:526`). `isupper` semantics: at least one cased character and
/// no lowercase ones; uncased characters (digits, titlecase, punctuation) are
/// skipped, matching CPython.
pub fn is_shift_character(s: &str) -> bool {
    // Python: character.isupper() or character in set('~!@#$%^&*()_+{}|:"<>?')
    // isupper = at least one cased char and no lowercase cased chars; uncased
    // chars (digits, punctuation, titlecase) are skipped, as in CPython.
    let mut has_upper = false;
    for c in s.chars() {
        if c.is_uppercase() {
            has_upper = true;
        } else if c.is_lowercase() {
            return false;
        }
    }
    if has_upper {
        return true;
    }
    // `x in set(...)` over single-char strings: multi-char names never match.
    matches!(
        s,
        "~" | "!" | "@" | "#" | "$" | "%" | "^" | "&" | "*" | "(" | ")" | "_" | "+" | "{" | "}"
            | "|" | ":" | "\"" | "<" | ">" | "?"
    )
}

/// Python `divmod(raw, 0x100)` on the `VkKeyScanA` i16 result, ported with
/// euclidean division so negatives behave identically: `-1` -> `(-1, 255)`,
/// `-256` -> `(-1, 0)`. `mods` stays signed (v1 happily sends the garbage
/// that `mods = -1` produces); `vk` is the low byte as `u16`.
pub fn vkscan_decode(raw: i16) -> (i32, u16) {
    let raw = i32::from(raw);
    (raw.div_euclid(0x100), raw.rem_euclid(0x100) as u16)
}

/// Full pyautogui `press(key)` event plan: name resolution (multi-char names
/// lowercased then named-table; single printable char 32..=127 via `vkscan`;
/// single control chars via the named table; unknown -> `None` = silent skip,
/// matching `_keyDown`'s early return), then the `_keyDown` + `_keyUp` mod
/// dances (alt, ctrl, shift DOWN -> key -> shift, ctrl, alt UP; `_keyUp`
/// repeats the dance with the key UP).
///
/// `vkscan` is called at most once and only for single printable chars — it is
/// the `VkKeyScanA` stand-in; tests pass panicking closures to prove the
/// lookup precedence.
pub fn key_plan(name: &str, vkscan: impl FnOnce(char) -> i16) -> Option<Vec<KeyEvent>> {
    // press() lowercases only when len(keys) > 1 (__init__.py:1588-1591), so
    // single chars keep their case ('L' stays shift-worthy).
    let normalized;
    let name: &str = if name.chars().count() > 1 {
        normalized = name.to_lowercase();
        &normalized
    } else {
        name
    };

    // Lookup order mirrors the Python dict build: printable ASCII 32..=127 was
    // filled LAST via VkKeyScanA, so single printable chars always go through
    // `vkscan` (overriding named entries like ' '); every other key resolves
    // through the named table; unknown keys return None (= silent skip).
    let mut chars = name.chars();
    let first = chars.next();
    let second = chars.next();
    let (raw, needs_shift) = match (first, second) {
        (Some(c), None) if ('\u{20}'..='\u{7f}').contains(&c) => {
            // needsShift runs on the KEY NAME (single char here) in pyautogui.
            (vkscan(c), is_shift_character(name))
        }
        // Named VKs max out at 0xB7, so this cast is lossless.
        _ => (named_vk(name)? as i16, is_shift_character(name)),
    };

    let (mods, vk) = vkscan_decode(raw);
    let shift = (mods & 1) != 0 || needs_shift;
    let ctrl = (mods & 2) != 0;
    let alt = (mods & 4) != 0;

    let mut plan = Vec::with_capacity(14);
    // _keyDown: mods DOWN (alt, ctrl, shift) -> key DOWN -> mods UP (shift, ctrl, alt).
    for (on, vk_mod) in [(alt, 0x12u16), (ctrl, 0x11), (shift, 0x10)] {
        if on {
            plan.push(KeyEvent { vk: vk_mod, down: true });
        }
    }
    plan.push(KeyEvent { vk, down: true });
    for (on, vk_mod) in [(shift, 0x10u16), (ctrl, 0x11), (alt, 0x12)] {
        if on {
            plan.push(KeyEvent { vk: vk_mod, down: false });
        }
    }
    // _keyUp: mods DOWN again (pyautogui sends them as keydowns too) -> key UP
    // -> mods UP (shift, ctrl, alt).
    for (on, vk_mod) in [(alt, 0x12u16), (ctrl, 0x11), (shift, 0x10)] {
        if on {
            plan.push(KeyEvent { vk: vk_mod, down: true });
        }
    }
    plan.push(KeyEvent { vk, down: false });
    for (on, vk_mod) in [(shift, 0x10u16), (ctrl, 0x11), (alt, 0x12)] {
        if on {
            plan.push(KeyEvent { vk: vk_mod, down: false });
        }
    }
    Some(plan)
}

/// `dwData` for `MOUSEEVENTF_WHEEL`: pyautogui passes the raw click count
/// (`_pyautogui_win.py:507-539`); no scaling.
pub fn wheel_mouse_data(clicks: i32) -> i32 {
    clicks
}

/// True when `(x, y)` is one of pyautogui's four FAILSAFE corner points for a
/// `w * h` screen (`__init__.py:2169-2171`): `(0,0)`, `(0,h-1)`, `(w-1,0)`,
/// `(w-1,h-1)`. Points one pixel outside the screen (e.g. `(0, h)`) are not
/// corners.
pub fn failsafe_point(x: i32, y: i32, w: i32, h: i32) -> bool {
    // FAILSAFE_POINTS = [(0, 0), (0, h-1), (w-1, 0), (w-1, h-1)]
    (x == 0 && y == 0) || (x == 0 && y == h - 1) || (x == w - 1 && y == 0) || (x == w - 1 && y == h - 1)
}

// ---------------------------------------------------------------------------
// Spike wiring (declared exception: thin Windows-API glue around the pure,
// TDD-covered functions above — same stance as capture/inference wiring).
// ---------------------------------------------------------------------------

/// pyautogui `press(key)` end-to-end: resolve the key plan (VkKeyScanA as the
/// printable-char resolver, exactly how pyautogui fills its table) and send
/// each event with `keybd_event`. Unknown keys are silently skipped, matching
/// `_keyDown`'s early return.
pub fn press(name: &str) {
    let Some(plan) = key_plan(name, |c| unsafe { VkKeyScanA(c as u8 as i8) }) else {
        return;
    };
    for ev in plan {
        // pyautogui has no KEYDOWN constant; keydown is flags = 0.
        let flags = if ev.down {
            KEYBD_EVENT_FLAGS(0)
        } else {
            KEYEVENTF_KEYUP
        };
        unsafe { keybd_event(ev.vk as u8, 0, flags, 0) };
    }
}

/// pyautogui `_scroll`: pass the current cursor position as dx/dy (Windows
/// ignores them without MOUSEEVENTF_MOVE) and the raw click count as dwData.
pub fn scroll(clicks: i32) {
    let mut pt = POINT::default();
    let _ = unsafe { GetCursorPos(&mut pt) };
    unsafe { mouse_event(MOUSEEVENTF_WHEEL, pt.x, pt.y, wheel_mouse_data(clicks), 0) };
}

/// Primary-monitor size in physical pixels, via the same
/// `GetSystemMetrics(SM_CXSCREEN/CYSCREEN)` pair pyautogui uses to build its
/// failsafe points — feeds `hud::window_position` for v1's top-right HUD
/// placement (controller.py:144-146).
pub fn screen_size() -> (i32, i32) {
    let w = unsafe { GetSystemMetrics(SM_CXSCREEN) };
    let h = unsafe { GetSystemMetrics(SM_CYSCREEN) };
    (w, h)
}

/// System DPI scale as a multiplier (96 dpi = 1.0), for converting screen
/// metrics into the logical points egui/winit windows are positioned in.
/// This process reads `screen_size()` while still DPI-unaware (before winit
/// declares awareness), so the metrics arrive already virtualized — divided
/// by the (virtualized) system DPI this yields the point space directly; on
/// a machine where a compat override pinned an aware context instead, the
/// physical metrics divide down to the same points. Verified end-to-end:
/// EnumWindows probe saw the HUD at rect (1036, 16) 494x398 visible +
/// topmost = v1's `sw - 500` placement on a 1920x1080 @1.25x screen.
pub fn system_dpi_scale() -> f32 {
    let dpi = unsafe { GetDpiForSystem() };
    if dpi == 0 {
        1.0
    } else {
        dpi as f32 / 96.0
    }
}

/// pyautogui `failSafeCheck`: raise (= stop) when the cursor sits in any of
/// the four screen corners and `pyautogui_failsafe` is enabled. When
/// GetCursorPos fails, pt stays (0, 0) — a corner — which matches v1, where a
/// failing ctypes call likewise leaves position() at (0, 0) and trips the
/// failsafe.
pub fn failsafe_triggered(enabled: bool) -> bool {
    if !enabled {
        return false;
    }
    let mut pt = POINT::default();
    let _ = unsafe { GetCursorPos(&mut pt) };
    let w = unsafe { GetSystemMetrics(SM_CXSCREEN) };
    let h = unsafe { GetSystemMetrics(SM_CYSCREEN) };
    failsafe_point(pt.x, pt.y, w, h)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ev(vk: u16, down: bool) -> KeyEvent {
        KeyEvent { vk, down }
    }

    // ---------------------------------------------------------------- named

    #[test]
    fn named_vk_full_pyautogui_table() {
        // Verbatim transcription of `_pyautogui_win.py:113-235` (121 entries).
        const TABLE: &[(&str, u16)] = &[
            ("backspace", 0x08),
            ("\u{8}", 0x08),
            ("super", 0x5B),
            ("tab", 0x09),
            ("\t", 0x09),
            ("clear", 0x0C),
            ("enter", 0x0D),
            ("\n", 0x0D),
            ("return", 0x0D),
            ("shift", 0x10),
            ("ctrl", 0x11),
            ("alt", 0x12),
            ("pause", 0x13),
            ("capslock", 0x14),
            ("kana", 0x15),
            ("hanguel", 0x15),
            ("hangul", 0x15),
            ("junja", 0x17),
            ("final", 0x18),
            ("hanja", 0x19),
            ("kanji", 0x19),
            ("esc", 0x1B),
            ("escape", 0x1B),
            ("convert", 0x1C),
            ("nonconvert", 0x1D),
            ("accept", 0x1E),
            ("modechange", 0x1F),
            (" ", 0x20),
            ("space", 0x20),
            ("pgup", 0x21),
            ("pgdn", 0x22),
            ("pageup", 0x21),
            ("pagedown", 0x22),
            ("end", 0x23),
            ("home", 0x24),
            ("left", 0x25),
            ("up", 0x26),
            ("right", 0x27),
            ("down", 0x28),
            ("select", 0x29),
            ("print", 0x2A),
            ("execute", 0x2B),
            ("prtsc", 0x2C),
            ("prtscr", 0x2C),
            ("prntscrn", 0x2C),
            ("printscreen", 0x2C),
            ("insert", 0x2D),
            ("del", 0x2E),
            ("delete", 0x2E),
            ("help", 0x2F),
            ("win", 0x5B),
            ("winleft", 0x5B),
            ("winright", 0x5C),
            ("apps", 0x5D),
            ("sleep", 0x5F),
            ("num0", 0x60),
            ("num1", 0x61),
            ("num2", 0x62),
            ("num3", 0x63),
            ("num4", 0x64),
            ("num5", 0x65),
            ("num6", 0x66),
            ("num7", 0x67),
            ("num8", 0x68),
            ("num9", 0x69),
            ("multiply", 0x6A),
            ("add", 0x6B),
            ("separator", 0x6C),
            ("subtract", 0x6D),
            ("decimal", 0x6E),
            ("divide", 0x6F),
            ("f1", 0x70),
            ("f2", 0x71),
            ("f3", 0x72),
            ("f4", 0x73),
            ("f5", 0x74),
            ("f6", 0x75),
            ("f7", 0x76),
            ("f8", 0x77),
            ("f9", 0x78),
            ("f10", 0x79),
            ("f11", 0x7A),
            ("f12", 0x7B),
            ("f13", 0x7C),
            ("f14", 0x7D),
            ("f15", 0x7E),
            ("f16", 0x7F),
            ("f17", 0x80),
            ("f18", 0x81),
            ("f19", 0x82),
            ("f20", 0x83),
            ("f21", 0x84),
            ("f22", 0x85),
            ("f23", 0x86),
            ("f24", 0x87),
            ("numlock", 0x90),
            ("scrolllock", 0x91),
            ("shiftleft", 0xA0),
            ("shiftright", 0xA1),
            ("ctrlleft", 0xA2),
            ("ctrlright", 0xA3),
            ("altleft", 0xA4),
            ("altright", 0xA5),
            ("browserback", 0xA6),
            ("browserforward", 0xA7),
            ("browserrefresh", 0xA8),
            ("browserstop", 0xA9),
            ("browsersearch", 0xAA),
            ("browserfavorites", 0xAB),
            ("browserhome", 0xAC),
            ("volumemute", 0xAD),
            ("volumedown", 0xAE),
            ("volumeup", 0xAF),
            ("nexttrack", 0xB0),
            ("prevtrack", 0xB1),
            ("stop", 0xB2),
            ("playpause", 0xB3),
            ("launchmail", 0xB4),
            ("launchmediaselect", 0xB5),
            ("launchapp1", 0xB6),
            ("launchapp2", 0xB7),
        ];
        assert_eq!(TABLE.len(), 121);
        for &(name, vk) in TABLE {
            assert_eq!(named_vk(name), Some(vk), "named_vk({name:?})");
        }
    }

    #[test]
    fn named_vk_unknown_names_are_none() {
        // Case-sensitive dict, lowercase keys only (Python: 'LEFT' not in mapping).
        assert_eq!(named_vk("LEFT"), None);
        assert_eq!(named_vk("Shift"), None);
        assert_eq!(named_vk("notakey"), None);
        // In KEY_NAMES but never given a value -> None in Python too (silent skip).
        assert_eq!(named_vk("\r"), None);
        assert_eq!(named_vk(""), None);
    }

    // ------------------------------------------------------------- shift char

    #[test]
    fn is_shift_character_matches_python() {
        // The 21 symbols from `set('~!@#$%^&*()_+{}|:"<>?')` (__init__.py:526).
        for s in [
            '~', '!', '@', '#', '$', '%', '^', '&', '*', '(', ')', '_', '+', '{', '}', '|', ':',
            '"', '<', '>', '?',
        ] {
            assert!(is_shift_character(&s.to_string()), "{s:?} must shift");
        }
        assert!(is_shift_character("L"));
        assert!(is_shift_character("LEFT"));
        assert!(is_shift_character("L1"));
        assert!(!is_shift_character("l"));
        assert!(!is_shift_character("left"));
        assert!(!is_shift_character("ll"));
        assert!(!is_shift_character("1"));
        assert!(!is_shift_character("\t"));
        assert!(!is_shift_character(","));
        assert!(!is_shift_character(""));
    }

    // ---------------------------------------------------------------- vkscan

    #[test]
    fn vkscan_decode_matches_python_divmod() {
        assert_eq!(vkscan_decode(0x014C), (1, 0x4C)); // shift + 'L'
        assert_eq!(vkscan_decode(0x0661), (6, 0x61)); // ctrl+alt + 'a'
        assert_eq!(vkscan_decode(0x0000), (0, 0));
        assert_eq!(vkscan_decode(-1), (-1, 255)); // VkKeyScanA failure
        assert_eq!(vkscan_decode(-256), (-1, 0));
        assert_eq!(vkscan_decode(0x7FFF), (127, 255));
    }

    // -------------------------------------------------------------- key plan

    #[test]
    fn key_plan_plain_lowercase_char_is_two_events() {
        let plan = key_plan("l", |c| {
            assert_eq!(c, 'l');
            0x004C
        })
        .unwrap();
        assert_eq!(plan, vec![ev(0x4C, true), ev(0x4C, false)]);
    }

    #[test]
    fn key_plan_named_keys_skip_vkscan() {
        // Arrow keys are the v1 seek keys; the VkKeyScan closure must not run.
        let plan = key_plan("left", |_| panic!("vkscan must not run for named keys")).unwrap();
        assert_eq!(plan, vec![ev(0x25, true), ev(0x25, false)]);
        let plan = key_plan("right", |_| panic!("vkscan must not run for named keys")).unwrap();
        assert_eq!(plan, vec![ev(0x27, true), ev(0x27, false)]);
    }

    #[test]
    fn key_plan_multichar_names_are_lowercased_first() {
        // press() lowercases when len(keys) > 1 (__init__.py:1588-1591).
        let plan = key_plan("LEFT", |_| panic!("vkscan must not run for named keys")).unwrap();
        assert_eq!(plan, vec![ev(0x25, true), ev(0x25, false)]);
        let plan =
            key_plan("BrowserBack", |_| panic!("vkscan must not run for named keys")).unwrap();
        assert_eq!(plan, vec![ev(0xA6, true), ev(0xA6, false)]);
        let plan = key_plan("Shift", |_| panic!("vkscan must not run for named keys")).unwrap();
        assert_eq!(plan, vec![ev(0x10, true), ev(0x10, false)]);
    }

    #[test]
    fn key_plan_uppercase_char_runs_shift_dance() {
        // 'L' stays uppercase (single char), VkKeyScanA -> shift bit + 0x4C.
        // is_shift_character('L') is also true; either way exactly one shift.
        let plan = key_plan("L", |_| 0x014C).unwrap();
        assert_eq!(
            plan,
            vec![
                ev(0x10, true),
                ev(0x4C, true),
                ev(0x10, false),
                ev(0x10, true),
                ev(0x4C, false),
                ev(0x10, false),
            ]
        );
    }

    #[test]
    fn key_plan_ctrl_alt_mods_send_ten_events() {
        // mods = 6 -> alt down, ctrl down, key down, ctrl up, alt up;
        // then the _keyUp dance repeats the mod downs (as keydowns!) with the
        // key UP in the middle.
        let plan = key_plan("a", |_| 0x0661).unwrap();
        assert_eq!(
            plan,
            vec![
                ev(0x12, true),
                ev(0x11, true),
                ev(0x61, true),
                ev(0x11, false),
                ev(0x12, false),
                ev(0x12, true),
                ev(0x11, true),
                ev(0x61, false),
                ev(0x11, false),
                ev(0x12, false),
            ]
        );
    }

    #[test]
    fn key_plan_negative_scan_code_is_mod_garbage() {
        // VkKeyScanA failure (-1) -> mods=-1, vk=255; Python's bitwise ops
        // make every mod bit truthy, so 7 events per half = 14 total. Ported
        // faithfully (v1 sends this garbage too).
        let plan = key_plan("a", |_| -1).unwrap();
        assert_eq!(
            plan,
            vec![
                ev(0x12, true),
                ev(0x11, true),
                ev(0x10, true),
                ev(255, true),
                ev(0x10, false),
                ev(0x11, false),
                ev(0x12, false),
                ev(0x12, true),
                ev(0x11, true),
                ev(0x10, true),
                ev(255, false),
                ev(0x10, false),
                ev(0x11, false),
                ev(0x12, false),
            ]
        );
        assert_eq!(plan.len(), 14);
    }

    #[test]
    fn key_plan_control_char_tab_via_named_table() {
        // '\t' is a single char outside 32..=127 -> named table, no vkscan.
        let plan = key_plan("\t", |_| panic!("vkscan must not run for '\t'")).unwrap();
        assert_eq!(plan, vec![ev(0x09, true), ev(0x09, false)]);
    }

    #[test]
    fn key_plan_single_printable_char_beats_named_table() {
        // pyautogui fills printable ASCII 32..=127 LAST via VkKeyScanA, so ' '
        // resolves through vkscan, not the named table's 0x20. A distinctive
        // return value (shift+space) proves which path won.
        let plan = key_plan(" ", |_| 0x0120).unwrap();
        assert_eq!(
            plan,
            vec![
                ev(0x10, true),
                ev(0x20, true),
                ev(0x10, false),
                ev(0x10, true),
                ev(0x20, false),
                ev(0x10, false),
            ]
        );
    }

    #[test]
    fn key_plan_unknown_and_empty_names_are_none() {
        assert_eq!(key_plan("notakey", |_| 0), None);
        assert_eq!(key_plan("", |_| 0), None);
    }

    // ----------------------------------------------------------------- wheel

    #[test]
    fn wheel_mouse_data_is_identity() {
        assert_eq!(wheel_mouse_data(5), 5);
        assert_eq!(wheel_mouse_data(-5), -5);
        assert_eq!(wheel_mouse_data(0), 0);
        assert_eq!(wheel_mouse_data(120), 120);
    }

    // -------------------------------------------------------------- failsafe

    #[test]
    fn failsafe_point_is_exactly_the_four_corners() {
        let (w, h) = (1920, 1080);
        assert!(failsafe_point(0, 0, w, h));
        assert!(failsafe_point(0, h - 1, w, h));
        assert!(failsafe_point(w - 1, 0, w, h));
        assert!(failsafe_point(w - 1, h - 1, w, h));
        // Non-corners.
        assert!(!failsafe_point(1, 0, w, h));
        assert!(!failsafe_point(0, 1, w, h));
        assert!(!failsafe_point(w - 2, h - 1, w, h));
        assert!(!failsafe_point(w - 1, h - 2, w, h));
        assert!(!failsafe_point(w / 2, h / 2, w, h));
        // One pixel outside the screen is not a corner.
        assert!(!failsafe_point(0, h, w, h));
        assert!(!failsafe_point(w, 0, w, h));
    }
}
