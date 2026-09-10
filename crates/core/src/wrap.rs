//! Fixed wrapper strings that enter prompts. Versioned by [`WRAP_VERSION`]; changing any of
//! these strings is a breaking change and requires a version bump (plan §11 invariant 5).
//!
//! Codecs must reference these functions and never re-type the strings inline.

/// Version tag of all fixed wrapper strings that enter prompts.
pub const WRAP_VERSION: &str = "wrap_v1";

/// Wrap instruction text for inline-wrap fallback (plan §7.1):
///
/// ```text
/// <system_message>
/// {text}
/// </system_message>
/// ```
pub fn system_message(text: &str) -> String {
    format!("<system_message>\n{text}\n</system_message>")
}

/// Wrap a text document (plan §7.5). With a title:
///
/// ```text
/// <document title="{title}">
/// {text}
/// </document>
/// ```
///
/// Without a title the opening tag is just `<document>`. Any `"` in the title is escaped to
/// `&quot;` so the attribute cannot be broken.
pub fn document(title: Option<&str>, text: &str) -> String {
    match title {
        Some(t) => {
            let escaped = t.replace('"', "&quot;");
            format!("<document title=\"{escaped}\">\n{text}\n</document>")
        }
        None => format!("<document>\n{text}\n</document>"),
    }
}

/// Fold a foreign provider-tool result into a text block (plan §7.4):
///
/// ```text
/// [{tool} result]
/// {text}
/// ```
pub fn provider_tool_fold(tool: &str, text: &str) -> String {
    format!("[{tool} result]\n{text}")
}

/// The fixed prefix for a tool-result image/attachment carried as a following user message
/// (plan §7.4): `[tool result attachment for call {call_id}]`.
pub fn tool_result_attachment(call_id: &str) -> String {
    format!("[tool result attachment for call {call_id}]")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wrap_version_is_v1() {
        assert_eq!(WRAP_VERSION, "wrap_v1");
    }

    #[test]
    fn system_message_exact() {
        assert_eq!(system_message("hi"), "<system_message>\nhi\n</system_message>");
    }

    #[test]
    fn document_with_title_exact() {
        assert_eq!(
            document(Some("Report"), "body"),
            "<document title=\"Report\">\nbody\n</document>"
        );
    }

    #[test]
    fn document_without_title_exact() {
        assert_eq!(document(None, "body"), "<document>\nbody\n</document>");
    }

    #[test]
    fn document_title_escapes_quotes() {
        assert_eq!(
            document(Some("a\"b"), "x"),
            "<document title=\"a&quot;b\">\nx\n</document>"
        );
    }

    #[test]
    fn provider_tool_fold_exact() {
        assert_eq!(provider_tool_fold("web_search", "found"), "[web_search result]\nfound");
    }

    #[test]
    fn tool_result_attachment_exact() {
        assert_eq!(
            tool_result_attachment("call_9"),
            "[tool result attachment for call call_9]"
        );
    }
}
