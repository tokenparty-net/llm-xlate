//! Probe schema, loader, validation, dependency ordering, model-set expansion, and run-time
//! placeholder substitution.
//!
//! A *probe* is a single question about a live API: a verbatim wire body (with placeholders), a
//! protocol, a model set, a streaming mode, a stated hypothesis, and an expectation against
//! which the observed result is judged. Probes live as `[[probe]]` arrays in `probes/**/*.toml`.

use crate::models::{ModelSets, ModelsSpec};
use anyhow::{anyhow, bail, Context, Result};
use serde::Deserialize;
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

/// The three wire protocols a probe can target.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Protocol {
    /// Anthropic Messages (`ant.` prefix).
    Anthropic,
    /// OpenAI Chat Completions (`chat.` prefix).
    Chat,
    /// OpenAI Responses (`resp.` prefix).
    Responses,
}

impl Protocol {
    /// The dotted id prefix required of probes for this protocol.
    pub fn prefix(self) -> &'static str {
        match self {
            Protocol::Anthropic => "ant.",
            Protocol::Chat => "chat.",
            Protocol::Responses => "resp.",
        }
    }

    /// The `protocol` field spelling.
    pub fn as_str(self) -> &'static str {
        match self {
            Protocol::Anthropic => "anthropic",
            Protocol::Chat => "chat",
            Protocol::Responses => "responses",
        }
    }

    /// Parse the `protocol` field.
    pub fn parse(s: &str) -> Result<Self> {
        match s {
            "anthropic" => Ok(Protocol::Anthropic),
            "chat" => Ok(Protocol::Chat),
            "responses" => Ok(Protocol::Responses),
            other => bail!("unknown protocol `{other}` (want anthropic|chat|responses)"),
        }
    }
}

/// The streaming mode a probe requests.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamMode {
    /// Non-streaming only.
    Off,
    /// Streaming only.
    On,
    /// Both a non-streaming and a streaming run.
    Both,
}

impl StreamMode {
    /// The concrete stream flags this mode expands to (non-stream ordered first).
    pub fn flags(self) -> &'static [bool] {
        match self {
            StreamMode::Off => &[false],
            StreamMode::On => &[true],
            StreamMode::Both => &[false, true],
        }
    }
}

/// What a probe expects of the outcome.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Expect {
    /// Any outcome is acceptable (still observed and reported).
    Any,
    /// A specific HTTP status.
    Status(u16),
    /// A specific status and provider error `type`.
    StatusAndType {
        /// Expected HTTP status.
        status: u16,
        /// Expected provider error `type` string.
        error_type: String,
    },
}

impl Expect {
    /// The expected status, if any.
    pub fn status(&self) -> Option<u16> {
        match self {
            Expect::Any => None,
            Expect::Status(s) => Some(*s),
            Expect::StatusAndType { status, .. } => Some(*status),
        }
    }

    /// The expected error type, if any.
    pub fn error_type(&self) -> Option<&str> {
        match self {
            Expect::StatusAndType { error_type, .. } => Some(error_type),
            _ => None,
        }
    }
}

/// A fully-parsed probe.
#[derive(Debug, Clone)]
pub struct Probe {
    /// Dotted, unique id; its prefix matches [`Probe::protocol`].
    pub id: String,
    /// Target wire protocol.
    pub protocol: Protocol,
    /// Free-form tags used for filtering.
    pub tags: Vec<String>,
    /// Model set name or explicit list.
    pub models: ModelsSpec,
    /// Streaming mode.
    pub stream: StreamMode,
    /// Stated hypothesis (required in spirit for `explore` probes).
    pub hypothesis: Option<String>,
    /// Outcome expectation.
    pub expect: Expect,
    /// Preconditions that cannot be triggered reliably (`manual`, `files_api`, …).
    pub requires: Vec<String>,
    /// Explicit probe-id dependencies (implicit ones come from `${ref:…}`).
    pub after: Vec<String>,
    /// Free-text notes.
    pub notes: Option<String>,
    /// When true, `max_tokens`-family fields are not clamped.
    pub allow_long: bool,
    /// The verbatim wire body (placeholders unresolved).
    pub body: serde_json::Value,
    /// Optional extra request headers.
    pub headers: BTreeMap<String, String>,
}

impl Probe {
    /// The union of `after` and the probe ids referenced via `${ref:<id>:…}` in body/headers.
    pub fn dependencies(&self) -> BTreeSet<String> {
        let mut deps: BTreeSet<String> = self.after.iter().cloned().collect();
        collect_refs(&self.body, &mut deps);
        for v in self.headers.values() {
            collect_refs_str(v, &mut deps);
        }
        deps
    }
}

// ---------------------------------------------------------------------------------------------
// Raw TOML shapes.
// ---------------------------------------------------------------------------------------------

#[derive(Deserialize)]
struct RawFile {
    #[serde(default)]
    probe: Vec<RawProbe>,
}

#[derive(Deserialize)]
struct RawProbe {
    id: String,
    protocol: String,
    #[serde(default)]
    tags: Vec<String>,
    models: toml::Value,
    #[serde(default)]
    stream: Option<toml::Value>,
    #[serde(default)]
    hypothesis: Option<String>,
    #[serde(default)]
    expect: Option<toml::Value>,
    #[serde(default)]
    requires: Vec<String>,
    #[serde(default)]
    after: Vec<String>,
    #[serde(default)]
    notes: Option<String>,
    #[serde(default)]
    allow_long: bool,
    #[serde(default = "empty_toml_table")]
    body: toml::Value,
    #[serde(default)]
    headers: BTreeMap<String, String>,
}

/// The default probe body: an empty TOML table (`toml::Value` has no `Default`).
fn empty_toml_table() -> toml::Value {
    toml::Value::Table(toml::map::Map::new())
}

fn parse_models(v: &toml::Value) -> Result<ModelsSpec> {
    match v {
        toml::Value::String(s) => Ok(ModelsSpec::Set(s.clone())),
        toml::Value::Array(a) => {
            let mut list = Vec::new();
            for e in a {
                let s = e
                    .as_str()
                    .ok_or_else(|| anyhow!("models list entries must be strings"))?;
                list.push(s.to_string());
            }
            Ok(ModelsSpec::List(list))
        }
        _ => bail!("`models` must be a set name (string) or a list of ids"),
    }
}

fn parse_stream(v: &Option<toml::Value>) -> Result<StreamMode> {
    match v {
        None => Ok(StreamMode::Off),
        Some(toml::Value::Boolean(b)) => Ok(if *b { StreamMode::On } else { StreamMode::Off }),
        Some(toml::Value::String(s)) if s == "both" => Ok(StreamMode::Both),
        Some(other) => bail!("`stream` must be a bool or \"both\" (got {other:?})"),
    }
}

fn parse_expect(v: &Option<toml::Value>) -> Result<Expect> {
    match v {
        None => Ok(Expect::Any),
        Some(toml::Value::String(s)) if s == "any" => Ok(Expect::Any),
        Some(toml::Value::String(s)) => bail!("`expect` string must be \"any\" (got {s:?})"),
        Some(toml::Value::Table(t)) => {
            let status = t
                .get("status")
                .and_then(|x| x.as_integer())
                .ok_or_else(|| anyhow!("`expect.status` required and must be an integer"))?
                as u16;
            match t.get("error_type") {
                None => Ok(Expect::Status(status)),
                Some(toml::Value::String(et)) => Ok(Expect::StatusAndType {
                    status,
                    error_type: et.clone(),
                }),
                Some(_) => bail!("`expect.error_type` must be a string"),
            }
        }
        Some(other) => bail!("`expect` must be \"any\" or a table (got {other:?})"),
    }
}

/// Convert a TOML value into the equivalent JSON value (no datetimes in probe bodies).
fn toml_to_json(v: &toml::Value) -> Result<serde_json::Value> {
    serde_json::to_value(v).context("converting probe body from TOML to JSON")
}

impl Probe {
    fn from_raw(raw: RawProbe) -> Result<Self> {
        let protocol = Protocol::parse(&raw.protocol)
            .with_context(|| format!("probe `{}`", raw.id))?;
        let body = toml_to_json(&raw.body)
            .with_context(|| format!("probe `{}` body", raw.id))?;
        Ok(Probe {
            protocol,
            models: parse_models(&raw.models).with_context(|| format!("probe `{}`", raw.id))?,
            stream: parse_stream(&raw.stream).with_context(|| format!("probe `{}`", raw.id))?,
            expect: parse_expect(&raw.expect).with_context(|| format!("probe `{}`", raw.id))?,
            hypothesis: raw.hypothesis,
            tags: raw.tags,
            requires: raw.requires,
            after: raw.after,
            notes: raw.notes,
            allow_long: raw.allow_long,
            body,
            headers: raw.headers,
            id: raw.id,
        })
    }
}

// ---------------------------------------------------------------------------------------------
// Catalogue: load + validate + topological order.
// ---------------------------------------------------------------------------------------------

/// A validated, dependency-ordered set of probes.
#[derive(Debug, Clone)]
pub struct Catalogue {
    /// Probes in topological order (dependencies before dependents).
    pub probes: Vec<Probe>,
}

impl Catalogue {
    /// Parse probes from TOML text (one file's worth).
    pub fn parse_file(text: &str) -> Result<Vec<Probe>> {
        let raw: RawFile = toml::from_str(text).context("parsing probe TOML")?;
        raw.probe.into_iter().map(Probe::from_raw).collect()
    }

    /// Load every `*.toml` under `dir` (recursively), validate, and order.
    pub fn load(dir: &Path, models: &ModelSets) -> Result<Self> {
        let pattern = format!("{}/**/*.toml", dir.display());
        let mut files: Vec<_> = glob::glob(&pattern)
            .context("probe glob")?
            .filter_map(|r| r.ok())
            .collect();
        files.sort();
        let mut probes = Vec::new();
        for f in files {
            let txt = std::fs::read_to_string(&f)
                .with_context(|| format!("reading {}", f.display()))?;
            let ps = Catalogue::parse_file(&txt)
                .with_context(|| format!("in {}", f.display()))?;
            probes.extend(ps);
        }
        Catalogue::from_probes(probes, models)
    }

    /// Validate and topologically order an already-parsed probe list.
    pub fn from_probes(probes: Vec<Probe>, models: &ModelSets) -> Result<Self> {
        validate(&probes, models)?;
        let ordered = topo_sort(probes)?;
        Ok(Catalogue { probes: ordered })
    }

    /// Look up a probe by id.
    pub fn get(&self, id: &str) -> Option<&Probe> {
        self.probes.iter().find(|p| p.id == id)
    }
}

fn validate(probes: &[Probe], models: &ModelSets) -> Result<()> {
    // Unique ids.
    let mut seen: BTreeSet<&str> = BTreeSet::new();
    for p in probes {
        if !seen.insert(&p.id) {
            bail!("duplicate probe id `{}`", p.id);
        }
    }
    let ids: BTreeSet<&str> = probes.iter().map(|p| p.id.as_str()).collect();
    for p in probes {
        // Prefix matches protocol.
        if !p.id.starts_with(p.protocol.prefix()) {
            bail!(
                "probe `{}` has protocol {} but id must start with `{}`",
                p.id,
                p.protocol.as_str(),
                p.protocol.prefix()
            );
        }
        // Model set exists (explicit lists are always fine).
        if let ModelsSpec::Set(name) = &p.models {
            if !models.has_set(name) {
                bail!("probe `{}` references unknown model set `{}`", p.id, name);
            }
        }
        // Dependency targets exist.
        for dep in p.dependencies() {
            if !ids.contains(dep.as_str()) {
                bail!("probe `{}` depends on unknown probe `{}`", p.id, dep);
            }
        }
    }
    Ok(())
}

/// Deterministic topological sort (Kahn's algorithm with a sorted ready queue). Errors on a
/// dependency cycle.
fn topo_sort(probes: Vec<Probe>) -> Result<Vec<Probe>> {
    let mut by_id: BTreeMap<String, Probe> = probes.into_iter().map(|p| (p.id.clone(), p)).collect();
    // Build indegree over dependencies.
    let mut deps: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for (id, p) in &by_id {
        deps.insert(id.clone(), p.dependencies());
    }
    let mut ready: Vec<String> = deps
        .iter()
        .filter(|(_, d)| d.is_empty())
        .map(|(id, _)| id.clone())
        .collect();
    ready.sort();
    let mut out = Vec::new();
    let mut done: BTreeSet<String> = BTreeSet::new();
    while let Some(id) = ready.pop() {
        done.insert(id.clone());
        out.push(by_id.remove(&id).expect("id present"));
        // Recompute ready set deterministically.
        let mut newly: Vec<String> = deps
            .iter()
            .filter(|(cid, d)| {
                !done.contains(cid.as_str())
                    && !ready.contains(cid)
                    && d.iter().all(|dp| done.contains(dp))
            })
            .map(|(cid, _)| cid.clone())
            .collect();
        newly.sort();
        // Push in reverse so the smallest is popped first (stable, ascending output).
        for n in newly.into_iter().rev() {
            ready.push(n);
        }
    }
    if !by_id.is_empty() {
        let mut remaining: Vec<&String> = by_id.keys().collect();
        remaining.sort();
        bail!(
            "dependency cycle among probes: {}",
            remaining.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(", ")
        );
    }
    Ok(out)
}

// ---------------------------------------------------------------------------------------------
// Expansion.
// ---------------------------------------------------------------------------------------------

/// One concrete unit of work: a probe against a specific model in a specific stream mode.
#[derive(Debug, Clone)]
pub struct Expansion {
    /// The probe (cloned so downstream code needs no back-reference).
    pub probe: Probe,
    /// The resolved model id.
    pub model: String,
    /// Whether to stream.
    pub stream: bool,
}

impl Expansion {
    /// The on-disk directory name relative to the run root: `<probe id>/<model>[.stream]`.
    pub fn rel_dir(&self) -> String {
        let model = sanitize(&self.model);
        if self.stream {
            format!("{}/{}.stream", self.probe.id, model)
        } else {
            format!("{}/{}", self.probe.id, model)
        }
    }
}

/// Replace path-hostile characters in a model id for use as a directory name.
pub fn sanitize(s: &str) -> String {
    s.chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == '.' { c } else { '_' })
        .collect()
}

/// Options controlling expansion.
#[derive(Debug, Clone, Default)]
pub struct ExpandOpts {
    /// Only expand probes carrying at least one of these tags (empty = no tag filter).
    pub tags: Vec<String>,
    /// Only expand probes with these exact ids (empty = no id filter).
    pub probe_ids: Vec<String>,
    /// Only expand probes for this protocol.
    pub protocol: Option<Protocol>,
    /// Override every probe's model set with this set name.
    pub models_override: Option<String>,
    /// Use only the cheapest member of each set.
    pub cheap: bool,
    /// Force a stream mode for every probe (`None` = honour the probe).
    pub stream_override: Option<StreamMode>,
}

impl Catalogue {
    /// Expand the catalogue into concrete units of work, preserving topological order.
    pub fn expand(&self, models: &ModelSets, opts: &ExpandOpts) -> Result<Vec<Expansion>> {
        let mut out = Vec::new();
        for p in &self.probes {
            if let Some(proto) = opts.protocol {
                if p.protocol != proto {
                    continue;
                }
            }
            if !opts.tags.is_empty() && !opts.tags.iter().any(|t| p.tags.iter().any(|pt| pt == t)) {
                continue;
            }
            if !opts.probe_ids.is_empty() && !opts.probe_ids.iter().any(|i| i == &p.id) {
                continue;
            }
            let spec = match &opts.models_override {
                Some(name) => ModelsSpec::Set(name.clone()),
                None => p.models.clone(),
            };
            let model_ids = models
                .resolve(&spec, opts.cheap)
                .with_context(|| format!("probe `{}`", p.id))?;
            let mode = opts.stream_override.unwrap_or(p.stream);
            for model in &model_ids {
                for &stream in mode.flags() {
                    out.push(Expansion {
                        probe: p.clone(),
                        model: model.clone(),
                        stream,
                    });
                }
            }
        }
        Ok(out)
    }
}

// ---------------------------------------------------------------------------------------------
// Placeholder substitution.
// ---------------------------------------------------------------------------------------------

/// Supplies asset bytes and MIME types for `$ASSET` / `$ASSET_DATA_URL`.
pub trait AssetSource {
    /// Raw bytes of the named asset.
    fn bytes(&self, name: &str) -> Result<Vec<u8>>;
    /// The asset's MIME type (for data URLs).
    fn mime(&self, name: &str) -> Result<String>;
}

/// Resolves `${ref:<id>:<pointer>}` against earlier captures of the same run + model.
pub trait RefSource {
    /// The JSON value at `pointer` in the response captured for `probe_id`.
    fn resolve(&self, probe_id: &str, pointer: &str) -> Result<serde_json::Value>;
}

/// Everything substitution needs.
pub struct SubstCtx<'a> {
    /// The current model id.
    pub model: &'a str,
    /// A stable id for this run (for `$RUN_ID`).
    pub run_id: &'a str,
    /// Asset provider.
    pub assets: &'a dyn AssetSource,
    /// Reference provider.
    pub refs: &'a dyn RefSource,
}

/// Substitute placeholders throughout a body value.
pub fn substitute(body: &serde_json::Value, ctx: &SubstCtx) -> Result<serde_json::Value> {
    match body {
        serde_json::Value::String(s) => expand_string(s, ctx),
        serde_json::Value::Array(a) => {
            let mut out = Vec::with_capacity(a.len());
            for e in a {
                out.push(substitute(e, ctx)?);
            }
            Ok(serde_json::Value::Array(out))
        }
        serde_json::Value::Object(o) => {
            let mut map = serde_json::Map::new();
            for (k, v) in o {
                map.insert(k.clone(), substitute(v, ctx)?);
            }
            Ok(serde_json::Value::Object(map))
        }
        other => Ok(other.clone()),
    }
}

/// Substitute placeholders in a header value (always textual).
pub fn substitute_header(value: &str, ctx: &SubstCtx) -> Result<String> {
    replace_textual(value, ctx)
}

/// Expand a JSON string. If it is exactly one `${ref:…}`, the referenced JSON value replaces it
/// wholesale (so a ref may yield a non-string); otherwise all placeholders are textually
/// substituted and the result stays a string.
fn expand_string(s: &str, ctx: &SubstCtx) -> Result<serde_json::Value> {
    if let Some(inner) = whole_ref(s) {
        let (id, ptr) = split_ref(inner)?;
        return ctx.refs.resolve(id, ptr);
    }
    Ok(serde_json::Value::String(replace_textual(s, ctx)?))
}

/// If the whole string is exactly `${ref:...}`, return its inner text (`ref:...`).
fn whole_ref(s: &str) -> Option<&str> {
    let t = s.trim();
    if t.starts_with("${") && t.ends_with('}') && t[2..].starts_with("ref:") {
        Some(&t[2..t.len() - 1])
    } else {
        None
    }
}

/// Split `ref:<id>:<pointer>` into (`<id>`, `<pointer>`). The pointer may be empty or contain
/// further colons only inside the JSON-pointer syntax (which uses `/`, not `:`), so we split on
/// the first two colons.
fn split_ref(inner: &str) -> Result<(&str, &str)> {
    let rest = inner
        .strip_prefix("ref:")
        .ok_or_else(|| anyhow!("malformed ref `{inner}`"))?;
    match rest.split_once(':') {
        Some((id, ptr)) => Ok((id, ptr)),
        None => Ok((rest, "")),
    }
}

/// Textually replace every placeholder in `s`.
fn replace_textual(s: &str, ctx: &SubstCtx) -> Result<String> {
    let bytes = s.as_bytes();
    let mut out = String::with_capacity(s.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'$' {
            if i + 1 < bytes.len() && bytes[i + 1] == b'{' {
                // Braced: ${ref:...}
                let close = s[i..]
                    .find('}')
                    .ok_or_else(|| anyhow!("unterminated `${{` in {s:?}"))?;
                let inner = &s[i + 2..i + close];
                let (id, ptr) = split_ref(inner)?;
                let v = ctx.refs.resolve(id, ptr)?;
                out.push_str(&json_to_plain(&v));
                i += close + 1;
                continue;
            }
            // Bare token: read [A-Z_]+ then optional :name
            let tok_start = i + 1;
            let mut j = tok_start;
            while j < bytes.len() && (bytes[j].is_ascii_uppercase() || bytes[j] == b'_') {
                j += 1;
            }
            let token = &s[tok_start..j];
            match token {
                "MODEL" => {
                    out.push_str(ctx.model);
                    i = j;
                    continue;
                }
                "RUN_ID" => {
                    out.push_str(ctx.run_id);
                    i = j;
                    continue;
                }
                "ASSET_DATA_URL" | "ASSET" => {
                    // Expect ":name" next.
                    if j < bytes.len() && bytes[j] == b':' {
                        let name_start = j + 1;
                        let mut k = name_start;
                        while k < bytes.len()
                            && (bytes[k].is_ascii_alphanumeric()
                                || bytes[k] == b'_'
                                || bytes[k] == b'.'
                                || bytes[k] == b'-')
                        {
                            k += 1;
                        }
                        let name = &s[name_start..k];
                        let data = ctx.assets.bytes(name)?;
                        let b64 = b64encode(&data);
                        if token == "ASSET_DATA_URL" {
                            let mime = ctx.assets.mime(name)?;
                            out.push_str(&format!("data:{mime};base64,{b64}"));
                        } else {
                            out.push_str(&b64);
                        }
                        i = k;
                        continue;
                    }
                    // No name: emit `$` literally.
                    out.push('$');
                    i += 1;
                    continue;
                }
                _ => {
                    // Unknown token; emit `$` literally and continue past it.
                    out.push('$');
                    i += 1;
                    continue;
                }
            }
        }
        // Copy the current UTF-8 char intact.
        let ch_len = utf8_len(bytes[i]);
        out.push_str(&s[i..i + ch_len]);
        i += ch_len;
    }
    Ok(out)
}

fn utf8_len(b: u8) -> usize {
    if b < 0x80 {
        1
    } else if b >> 5 == 0b110 {
        2
    } else if b >> 4 == 0b1110 {
        3
    } else {
        4
    }
}

fn json_to_plain(v: &serde_json::Value) -> String {
    match v {
        serde_json::Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

/// Base64 (standard, padded) encode.
pub fn b64encode(data: &[u8]) -> String {
    use base64::Engine;
    base64::engine::general_purpose::STANDARD.encode(data)
}

fn collect_refs(v: &serde_json::Value, out: &mut BTreeSet<String>) {
    match v {
        serde_json::Value::String(s) => collect_refs_str(s, out),
        serde_json::Value::Array(a) => a.iter().for_each(|e| collect_refs(e, out)),
        serde_json::Value::Object(o) => o.values().for_each(|e| collect_refs(e, out)),
        _ => {}
    }
}

fn collect_refs_str(s: &str, out: &mut BTreeSet<String>) {
    let mut rest = s;
    while let Some(pos) = rest.find("${ref:") {
        let after = &rest[pos + 2..]; // skip "${"
        if let Some(close) = after.find('}') {
            let inner = &after[..close];
            if let Ok((id, _)) = split_ref(inner) {
                out.insert(id.to_string());
            }
            rest = &after[close + 1..];
        } else {
            break;
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Token clamping.
// ---------------------------------------------------------------------------------------------

/// The token-limit cap applied unless a probe sets `allow_long`.
pub const TOKEN_CAP: u64 = 256;

/// Clamp `max_tokens` / `max_output_tokens` / `max_completion_tokens` on the top-level object to
/// [`TOKEN_CAP`] unless `allow_long`, and — crucially — **inject** the protocol's token-limit field
/// at [`TOKEN_CAP`] when the body carries none, so a probe that simply omits the field can never run
/// to the model's default maximum and blow the spend ceiling. Returns the (possibly modified) body.
pub fn clamp_tokens(mut body: serde_json::Value, allow_long: bool, protocol: Protocol) -> serde_json::Value {
    if allow_long {
        return body;
    }
    let Some(obj) = body.as_object_mut() else {
        return body;
    };
    // Lower any present token-limit field that exceeds the cap.
    let mut present = false;
    for key in ["max_tokens", "max_output_tokens", "max_completion_tokens"] {
        if let Some(v) = obj.get_mut(key) {
            present = true;
            if let Some(n) = v.as_u64() {
                if n > TOKEN_CAP {
                    *v = serde_json::json!(TOKEN_CAP);
                }
            }
        }
    }
    // No token-limit field at all: inject the protocol-appropriate one at the cap so a generating
    // call cannot silently spend unbounded.
    if !present {
        let field = match protocol {
            Protocol::Anthropic => "max_tokens",
            Protocol::Responses => "max_output_tokens",
            Protocol::Chat => "max_completion_tokens",
        };
        obj.insert(field.to_string(), serde_json::json!(TOKEN_CAP));
    }
    body
}

#[cfg(test)]
mod tests {
    use super::*;

    struct MapAssets;
    impl AssetSource for MapAssets {
        fn bytes(&self, name: &str) -> Result<Vec<u8>> {
            match name {
                "png_1px" => Ok(vec![1, 2, 3, 4]),
                _ => bail!("no asset {name}"),
            }
        }
        fn mime(&self, name: &str) -> Result<String> {
            match name {
                "png_1px" => Ok("image/png".into()),
                _ => bail!("no mime {name}"),
            }
        }
    }

    struct MapRefs;
    impl RefSource for MapRefs {
        fn resolve(&self, probe_id: &str, pointer: &str) -> Result<serde_json::Value> {
            match (probe_id, pointer) {
                ("ant.tools.call", "/content/1/id") => Ok(serde_json::json!("toolu_real")),
                ("ant.tools.call", "/usage/output_tokens") => Ok(serde_json::json!(42)),
                _ => bail!("no ref {probe_id}{pointer}"),
            }
        }
    }

    fn ctx<'a>() -> SubstCtx<'a> {
        SubstCtx { model: "claude-opus-4-8", run_id: "run-xyz", assets: &MapAssets, refs: &MapRefs }
    }

    const SETS: &str = r#"
refreshed_at = "2026-09-10"
[sets.anthropic_all]
models = ["claude-opus-4-8", "claude-sonnet-4-8"]
[sets.chat_set]
models = ["gpt-4o-mini"]
"#;

    fn models() -> ModelSets {
        ModelSets::from_toml(SETS).unwrap()
    }

    #[test]
    fn parses_probe_with_body_and_headers() {
        let toml = r#"
[[probe]]
id = "ant.text.hello"
protocol = "anthropic"
tags = ["example"]
models = "anthropic_all"
stream = false
hypothesis = "returns 200"
expect = { status = 200 }
notes = "hi"

[probe.body]
model = "$MODEL"
max_tokens = 64
messages = [ { role = "user", content = "hi" } ]

[probe.headers]
"anthropic-beta" = "x"
"#;
        let ps = Catalogue::parse_file(toml).unwrap();
        assert_eq!(ps.len(), 1);
        let p = &ps[0];
        assert_eq!(p.protocol, Protocol::Anthropic);
        assert_eq!(p.expect, Expect::Status(200));
        assert_eq!(p.stream, StreamMode::Off);
        assert_eq!(p.headers.get("anthropic-beta").map(String::as_str), Some("x"));
        assert_eq!(p.body["max_tokens"], serde_json::json!(64));
    }

    #[test]
    fn stream_both_and_expect_variants() {
        let toml = r#"
[[probe]]
id = "chat.err.badkey"
protocol = "chat"
models = ["gpt-4o-mini"]
stream = "both"
expect = { status = 401, error_type = "invalid_request_error" }
[probe.body]
model = "$MODEL"
"#;
        let p = &Catalogue::parse_file(toml).unwrap()[0];
        assert_eq!(p.stream, StreamMode::Both);
        assert_eq!(
            p.expect,
            Expect::StatusAndType { status: 401, error_type: "invalid_request_error".into() }
        );
        assert_eq!(p.stream.flags(), &[false, true]);
    }

    #[test]
    fn expect_any_default_and_word() {
        let toml = r#"
[[probe]]
id = "resp.any.default"
protocol = "responses"
models = ["m"]
[probe.body]
model = "$MODEL"

[[probe]]
id = "resp.any.word"
protocol = "responses"
models = ["m"]
expect = "any"
[probe.body]
model = "$MODEL"
"#;
        let ps = Catalogue::parse_file(toml).unwrap();
        assert_eq!(ps[0].expect, Expect::Any);
        assert_eq!(ps[1].expect, Expect::Any);
    }

    #[test]
    fn validation_rejects_prefix_mismatch() {
        let toml = r#"
[[probe]]
id = "chat.bad"
protocol = "anthropic"
models = ["m"]
[probe.body]
x = 1
"#;
        let ps = Catalogue::parse_file(toml).unwrap();
        let err = Catalogue::from_probes(ps, &models()).unwrap_err();
        assert!(err.to_string().contains("must start with `ant.`"));
    }

    #[test]
    fn validation_rejects_duplicate_ids() {
        let toml = r#"
[[probe]]
id = "ant.a"
protocol = "anthropic"
models = ["m"]
[probe.body]
x = 1
[[probe]]
id = "ant.a"
protocol = "anthropic"
models = ["m"]
[probe.body]
x = 2
"#;
        let ps = Catalogue::parse_file(toml).unwrap();
        assert!(Catalogue::from_probes(ps, &models()).unwrap_err().to_string().contains("duplicate"));
    }

    #[test]
    fn validation_rejects_unknown_model_set() {
        let toml = r#"
[[probe]]
id = "ant.a"
protocol = "anthropic"
models = "ghost_set"
[probe.body]
x = 1
"#;
        let ps = Catalogue::parse_file(toml).unwrap();
        assert!(Catalogue::from_probes(ps, &models()).unwrap_err().to_string().contains("unknown model set"));
    }

    #[test]
    fn validation_rejects_unknown_dependency() {
        let toml = r#"
[[probe]]
id = "ant.a"
protocol = "anthropic"
models = ["m"]
after = ["ant.missing"]
[probe.body]
x = 1
"#;
        let ps = Catalogue::parse_file(toml).unwrap();
        assert!(Catalogue::from_probes(ps, &models()).unwrap_err().to_string().contains("unknown probe"));
    }

    #[test]
    fn dependencies_include_refs_and_after() {
        let toml = r#"
[[probe]]
id = "ant.base"
protocol = "anthropic"
models = ["m"]
[probe.body]
x = 1

[[probe]]
id = "ant.dependent"
protocol = "anthropic"
models = ["m"]
after = ["ant.base"]
[probe.body]
tool_id = "${ref:ant.base:/content/0/id}"
"#;
        let ps = Catalogue::parse_file(toml).unwrap();
        let dep = ps.iter().find(|p| p.id == "ant.dependent").unwrap();
        let deps = dep.dependencies();
        assert!(deps.contains("ant.base"));
    }

    #[test]
    fn topo_sort_orders_deps_before_dependents() {
        let toml = r#"
[[probe]]
id = "ant.dependent"
protocol = "anthropic"
models = ["m"]
[probe.body]
tool_id = "${ref:ant.base:/id}"

[[probe]]
id = "ant.base"
protocol = "anthropic"
models = ["m"]
[probe.body]
x = 1
"#;
        let ps = Catalogue::parse_file(toml).unwrap();
        let cat = Catalogue::from_probes(ps, &models()).unwrap();
        let pos_base = cat.probes.iter().position(|p| p.id == "ant.base").unwrap();
        let pos_dep = cat.probes.iter().position(|p| p.id == "ant.dependent").unwrap();
        assert!(pos_base < pos_dep);
    }

    #[test]
    fn topo_sort_detects_cycle() {
        let toml = r#"
[[probe]]
id = "ant.a"
protocol = "anthropic"
models = ["m"]
after = ["ant.b"]
[probe.body]
x = 1
[[probe]]
id = "ant.b"
protocol = "anthropic"
models = ["m"]
after = ["ant.a"]
[probe.body]
x = 1
"#;
        let ps = Catalogue::parse_file(toml).unwrap();
        assert!(Catalogue::from_probes(ps, &models()).unwrap_err().to_string().contains("cycle"));
    }

    #[test]
    fn expansion_model_and_stream_product() {
        let toml = r#"
[[probe]]
id = "ant.x"
protocol = "anthropic"
models = "anthropic_all"
stream = "both"
[probe.body]
model = "$MODEL"
"#;
        let ps = Catalogue::parse_file(toml).unwrap();
        let cat = Catalogue::from_probes(ps, &models()).unwrap();
        let all = cat.expand(&models(), &ExpandOpts { cheap: false, ..Default::default() }).unwrap();
        // 2 models x 2 stream modes.
        assert_eq!(all.len(), 4);
        let cheap = cat.expand(&models(), &ExpandOpts { cheap: true, ..Default::default() }).unwrap();
        // 1 model x 2 stream modes.
        assert_eq!(cheap.len(), 2);
    }

    #[test]
    fn expansion_tag_filter() {
        let toml = r#"
[[probe]]
id = "ant.tagged"
protocol = "anthropic"
tags = ["explore"]
models = ["m"]
[probe.body]
x = 1
[[probe]]
id = "ant.untagged"
protocol = "anthropic"
tags = ["example"]
models = ["m"]
[probe.body]
x = 1
"#;
        let ps = Catalogue::parse_file(toml).unwrap();
        let cat = Catalogue::from_probes(ps, &models()).unwrap();
        let opts = ExpandOpts { tags: vec!["explore".into()], cheap: true, ..Default::default() };
        let ex = cat.expand(&models(), &opts).unwrap();
        assert_eq!(ex.len(), 1);
        assert_eq!(ex[0].probe.id, "ant.tagged");
    }

    #[test]
    fn rel_dir_names() {
        let p = Catalogue::parse_file(
            "[[probe]]\nid=\"ant.x\"\nprotocol=\"anthropic\"\nmodels=[\"m\"]\n[probe.body]\nx=1\n",
        )
        .unwrap()
        .remove(0);
        let e1 = Expansion { probe: p.clone(), model: "claude-opus-4-8".into(), stream: false };
        let e2 = Expansion { probe: p, model: "gpt/weird".into(), stream: true };
        assert_eq!(e1.rel_dir(), "ant.x/claude-opus-4-8");
        assert_eq!(e2.rel_dir(), "ant.x/gpt_weird.stream");
    }

    #[test]
    fn substitute_model_run_id_and_asset() {
        let body = serde_json::json!({
            "model": "$MODEL",
            "user": "user-$RUN_ID",
            "image": "$ASSET:png_1px",
            "data_url": "$ASSET_DATA_URL:png_1px"
        });
        let out = substitute(&body, &ctx()).unwrap();
        assert_eq!(out["model"], serde_json::json!("claude-opus-4-8"));
        assert_eq!(out["user"], serde_json::json!("user-run-xyz"));
        assert_eq!(out["image"], serde_json::json!("AQIDBA=="));
        assert_eq!(out["data_url"], serde_json::json!("data:image/png;base64,AQIDBA=="));
    }

    #[test]
    fn substitute_whole_ref_yields_native_type() {
        let body = serde_json::json!({
            "tool_id": "${ref:ant.tools.call:/content/1/id}",
            "n": "${ref:ant.tools.call:/usage/output_tokens}"
        });
        let out = substitute(&body, &ctx()).unwrap();
        assert_eq!(out["tool_id"], serde_json::json!("toolu_real"));
        // whole-ref to an integer yields an integer, not a string
        assert_eq!(out["n"], serde_json::json!(42));
    }

    #[test]
    fn substitute_embedded_ref_is_textual() {
        let body = serde_json::json!({ "note": "id is ${ref:ant.tools.call:/content/1/id} ok" });
        let out = substitute(&body, &ctx()).unwrap();
        assert_eq!(out["note"], serde_json::json!("id is toolu_real ok"));
    }

    #[test]
    fn substitute_missing_ref_errors() {
        let body = serde_json::json!({ "x": "${ref:ant.nope:/id}" });
        assert!(substitute(&body, &ctx()).is_err());
    }

    #[test]
    fn clamp_reduces_only_over_cap() {
        let body = serde_json::json!({ "max_tokens": 4096, "max_output_tokens": 100 });
        let out = clamp_tokens(body, false, Protocol::Anthropic);
        assert_eq!(out["max_tokens"], serde_json::json!(256));
        assert_eq!(out["max_output_tokens"], serde_json::json!(100));
    }

    #[test]
    fn clamp_respects_allow_long() {
        let body = serde_json::json!({ "max_tokens": 4096 });
        let out = clamp_tokens(body, true, Protocol::Anthropic);
        assert_eq!(out["max_tokens"], serde_json::json!(4096));
    }

    #[test]
    fn clamp_injects_cap_when_absent_per_protocol() {
        // Anthropic → max_tokens.
        let out = clamp_tokens(serde_json::json!({ "model": "m" }), false, Protocol::Anthropic);
        assert_eq!(out["max_tokens"], serde_json::json!(256));
        assert!(out.get("max_output_tokens").is_none());
        // Responses → max_output_tokens.
        let out = clamp_tokens(serde_json::json!({ "model": "m" }), false, Protocol::Responses);
        assert_eq!(out["max_output_tokens"], serde_json::json!(256));
        // Chat → max_completion_tokens.
        let out = clamp_tokens(serde_json::json!({ "model": "m" }), false, Protocol::Chat);
        assert_eq!(out["max_completion_tokens"], serde_json::json!(256));
        // allow_long suppresses injection entirely.
        let out = clamp_tokens(serde_json::json!({ "model": "m" }), true, Protocol::Anthropic);
        assert!(out.get("max_tokens").is_none());
    }
}
