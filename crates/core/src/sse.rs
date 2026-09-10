//! A byte-boundary-safe, incremental Server-Sent-Events parser and a matching writer.
//!
//! [`SseParser`] accepts arbitrary byte chunks (a UTF-8 char or a CRLF may be split across
//! `push` calls) and yields complete [`SseEvent`]s. It joins multi-line `data:` fields with
//! `"\n"`, treats `:`-prefixed lines as comments, and ignores unknown fields.

use bytes::Bytes;

/// A parsed SSE event. `event`/`id` are `None` unless the corresponding field was present.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SseEvent {
    /// The `event:` field value, if any.
    pub event: Option<String>,
    /// The joined `data:` payload (multiple `data:` lines joined by `"\n"`).
    pub data: String,
    /// The last-seen `id:` field value, if any (persists across events per the SSE spec).
    pub id: Option<String>,
}

/// An incremental SSE parser. Feed bytes with [`SseParser::push`]; call [`SseParser::finish`]
/// at end of stream to flush any trailing event that lacked a final blank line.
#[derive(Debug, Default)]
pub struct SseParser {
    buf: Vec<u8>,
    event: Option<String>,
    data: Vec<String>,
    id: Option<String>,
}

impl SseParser {
    /// A fresh parser.
    pub fn new() -> Self {
        Self::default()
    }

    /// Feed the next chunk of bytes; returns any events that completed.
    pub fn push(&mut self, bytes: &[u8]) -> Vec<SseEvent> {
        self.buf.extend_from_slice(bytes);
        self.drain(false)
    }

    /// Flush at end of stream. Processes a trailing line without a terminator and dispatches
    /// a pending event that was not followed by a blank line.
    pub fn finish(&mut self) -> Vec<SseEvent> {
        self.drain(true)
    }

    fn drain(&mut self, final_: bool) -> Vec<SseEvent> {
        let mut out = Vec::new();
        let mut start = 0usize;
        loop {
            let rel = self.buf[start..].iter().position(|&b| b == b'\n' || b == b'\r');
            let Some(rel) = rel else { break };
            let i = start + rel;
            let b = self.buf[i];
            let next = if b == b'\r' {
                if i + 1 < self.buf.len() {
                    if self.buf[i + 1] == b'\n' {
                        i + 2
                    } else {
                        i + 1
                    }
                } else if final_ {
                    i + 1
                } else {
                    // A lone trailing '\r' might be the first half of a CRLF split across
                    // pushes; wait for more bytes.
                    break;
                }
            } else {
                i + 1
            };
            let line = String::from_utf8_lossy(&self.buf[start..i]).into_owned();
            start = next;
            self.process_line(&line, &mut out);
        }
        if start > 0 {
            self.buf.drain(..start);
        }
        if final_ {
            if !self.buf.is_empty() {
                let line = String::from_utf8_lossy(&self.buf).into_owned();
                self.buf.clear();
                self.process_line(&line, &mut out);
            }
            if !self.data.is_empty() {
                self.dispatch(&mut out);
            }
        }
        out
    }

    fn process_line(&mut self, line: &str, out: &mut Vec<SseEvent>) {
        if line.is_empty() {
            self.dispatch(out);
            return;
        }
        if line.starts_with(':') {
            return; // comment
        }
        let (field, value) = match line.find(':') {
            Some(idx) => {
                let f = &line[..idx];
                let mut v = &line[idx + 1..];
                if let Some(stripped) = v.strip_prefix(' ') {
                    v = stripped;
                }
                (f, v)
            }
            None => (line, ""),
        };
        match field {
            "event" => self.event = Some(value.to_string()),
            "data" => self.data.push(value.to_string()),
            // Per the SSE spec, an `id` containing a NUL is ignored.
            "id" if !value.contains('\0') => self.id = Some(value.to_string()),
            _ => {} // retry / unknown / NUL-bearing id ignored
        }
    }

    fn dispatch(&mut self, out: &mut Vec<SseEvent>) {
        if self.data.is_empty() {
            // Nothing to dispatch; reset the event-type buffer, keep the last id.
            self.event = None;
            return;
        }
        let data = self.data.join("\n");
        out.push(SseEvent { event: self.event.take(), data, id: self.id.clone() });
        self.data.clear();
    }
}

/// SSE frame writer (stateless).
pub struct SseWriter;

impl SseWriter {
    /// Render an SSE frame. With `event`, emits `event: <event>\n` then one `data:` line per
    /// line of `data`, terminated by a blank line. Multi-line `data` is split correctly.
    pub fn frame(event: Option<&str>, data: &str) -> Bytes {
        let mut s = String::new();
        if let Some(e) = event {
            s.push_str("event: ");
            s.push_str(e);
            s.push('\n');
        }
        for line in data.split('\n') {
            s.push_str("data: ");
            s.push_str(line);
            s.push('\n');
        }
        s.push('\n');
        Bytes::from(s)
    }

    /// Render an SSE comment frame (`: <text>`), used for keepalives.
    pub fn comment(text: &str) -> Bytes {
        Bytes::from(format!(": {text}\n\n"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn parse_all(chunks: &[&[u8]]) -> Vec<SseEvent> {
        let mut p = SseParser::new();
        let mut out = Vec::new();
        for c in chunks {
            out.extend(p.push(c));
        }
        out.extend(p.finish());
        out
    }

    #[test]
    fn basic_event() {
        let evs = parse_all(&[b"data: hello\n\n"]);
        assert_eq!(evs, vec![SseEvent { event: None, data: "hello".into(), id: None }]);
    }

    #[test]
    fn event_and_id_and_multiline_data() {
        let evs = parse_all(&[b"event: message\nid: 7\ndata: a\ndata: b\n\n"]);
        assert_eq!(
            evs,
            vec![SseEvent { event: Some("message".into()), data: "a\nb".into(), id: Some("7".into()) }]
        );
    }

    #[test]
    fn crlf_and_comments_and_unknown_fields() {
        let evs = parse_all(&[b": keepalive\r\nfoo: bar\r\ndata: x\r\n\r\n"]);
        assert_eq!(evs, vec![SseEvent { event: None, data: "x".into(), id: None }]);
    }

    #[test]
    fn crlf_split_across_pushes() {
        let mut p = SseParser::new();
        let mut evs = p.push(b"data: hi\r");
        evs.extend(p.push(b"\n\r\n"));
        assert_eq!(evs, vec![SseEvent { event: None, data: "hi".into(), id: None }]);
    }

    #[test]
    fn utf8_split_across_pushes() {
        // "é" is 0xC3 0xA9; split it across pushes.
        let mut p = SseParser::new();
        let mut evs = p.push(b"data: caf\xc3");
        evs.extend(p.push(b"\xa9\n\n"));
        assert_eq!(evs, vec![SseEvent { event: None, data: "café".into(), id: None }]);
    }

    #[test]
    fn id_persists_across_events() {
        let evs = parse_all(&[b"id: 1\ndata: a\n\ndata: b\n\n"]);
        assert_eq!(evs[0].id, Some("1".into()));
        assert_eq!(evs[1].id, Some("1".into()));
    }

    #[test]
    fn finish_flushes_trailing_event_without_blank_line() {
        let evs = parse_all(&[b"data: tail\n"]);
        assert_eq!(evs, vec![SseEvent { event: None, data: "tail".into(), id: None }]);
    }

    #[test]
    fn writer_frame_and_comment() {
        assert_eq!(&SseWriter::frame(Some("ping"), "{}")[..], b"event: ping\ndata: {}\n\n");
        assert_eq!(&SseWriter::frame(None, "[DONE]")[..], b"data: [DONE]\n\n");
        assert_eq!(&SseWriter::frame(None, "a\nb")[..], b"data: a\ndata: b\n\n");
        assert_eq!(&SseWriter::comment("hb")[..], b": hb\n\n");
    }

    // A canned multi-event document exercising CRLF, comments, multi-line data, unicode.
    const DOC: &[u8] = b"event: start\ndata: {\"a\":1}\n\n: hb\r\nevent: delta\ndata: line1\ndata: line2\r\n\r\ndata: caf\xc3\xa9\n\nid: z\nevent: done\ndata: END\n\n";

    proptest! {
        #[test]
        fn any_chunking_yields_identical_events(sizes in prop::collection::vec(1usize..7, 0..80)) {
            let one_shot = parse_all(&[DOC]);
            // Split DOC into chunks of the given sizes (byte boundaries, arbitrary).
            let mut p = SseParser::new();
            let mut got = Vec::new();
            let mut pos = 0usize;
            for sz in &sizes {
                if pos >= DOC.len() { break; }
                let end = (pos + sz).min(DOC.len());
                got.extend(p.push(&DOC[pos..end]));
                pos = end;
            }
            if pos < DOC.len() {
                got.extend(p.push(&DOC[pos..]));
            }
            got.extend(p.finish());
            prop_assert_eq!(got, one_shot);
        }
    }
}
