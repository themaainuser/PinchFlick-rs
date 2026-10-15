//! v1's YouTube detection via the foreground window's title.

/// Caches "is the focused window a YouTube tab?" every `interval` ms.
pub struct YouTubeDetector {
    pub markers: Vec<String>,
    pub interval: i64,
    last_check: Option<i64>,
    pub title: String,
    pub is_youtube: bool,
}

impl YouTubeDetector {
    pub fn new(markers: Vec<String>, check_interval_ms: i64) -> Self {
        Self {
            markers,
            interval: check_interval_ms,
            last_check: None,
            title: String::new(),
            is_youtube: false,
        }
    }

    /// Returns `(is_youtube, title)`; re-reads the title at most every
    /// `interval` ms unless `force`.
    pub fn check(&mut self, now_ms: i64, force: bool) -> (bool, String) {
        if force
            || self
                .last_check
                .is_none_or(|t| now_ms - t >= self.interval)
        {
            self.last_check = Some(now_ms);
            self.title = get_foreground_window_title();
            self.is_youtube = is_youtube_title(Some(&self.title), &self.markers);
        }
        (self.is_youtube, self.title.clone())
    }
}

/// True when the title carries the marker as a complete tail segment.
///
/// Real titles: `"X - YouTube"`, `"X - YouTube - Brave"` — so the marker must
/// sit at the very end OR be followed only by `" - <browser name>"`. Keeps
/// `"My blog post about YouTube"` / `"GitHub - YouTube.js"` from matching
/// while accepting every real browser's format (v1's regex translated).
pub fn is_youtube_title(title: Option<&str>, markers: &[String]) -> bool {
    let Some(title) = title else {
        return false;
    };
    if title.is_empty() {
        return false;
    }
    let folded = title.trim_end().to_lowercase();
    for marker in markers {
        let m = marker.trim_end().to_lowercase();
        if m.is_empty() {
            continue;
        }
        // v1: re.search(re.escape(m) + r"(?: [-–—] .*)?$", folded) — the
        // marker must end the string or be followed only by
        // " - <browser name>" segments
        let mut from = 0;
        while let Some(pos) = folded[from..].find(&m) {
            let i = from + pos;
            let rest = &folded[i + m.len()..];
            if rest.is_empty() || is_browser_suffix(rest) {
                return true;
            }
            from = i + 1;
        }
    }
    false
}

/// v1's regex tail ` [-–—] .*`: space, one dash-class char, space, rest.
fn is_browser_suffix(rest: &str) -> bool {
    let mut it = rest.chars();
    matches!(
        (it.next(), it.next(), it.next()),
        (Some(' '), Some('-' | '\u{2013}' | '\u{2014}'), Some(' '))
    )
}

/// Best-effort title of the focused window ("" if unknown).
pub fn get_foreground_window_title() -> String {
    use windows::Win32::UI::WindowsAndMessaging::{
        GetForegroundWindow, GetWindowTextLengthW, GetWindowTextW,
    };

    unsafe {
        let hwnd = GetForegroundWindow();
        if hwnd.0.is_null() {
            return String::new();
        }
        let len = GetWindowTextLengthW(hwnd);
        if len <= 0 {
            return String::new();
        }
        let mut buf = vec![0u16; len as usize + 1];
        let n = GetWindowTextW(hwnd, &mut buf);
        if n <= 0 {
            return String::new();
        }
        String::from_utf16_lossy(&buf[..n as usize])
    }
}
