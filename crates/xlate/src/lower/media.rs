//! Pass 4: media (plan §7.5).
//!
//! Resolves foreign-family file references via the router's [`Resolutions`], rejects audio the
//! backend cannot accept, and enforces image/PDF byte-size and count limits. Byte data is never
//! rescaled — an over-limit item is a hard [`XlateError::unsupported`].

use llm_xlate_core::caps::Capabilities;
use llm_xlate_core::degrade::Degradations;
use llm_xlate_core::error::XlateError;
use llm_xlate_core::ir::{IrRequest, Item, MediaSource, Part, Protocol};

use crate::requirements::{FileRef, Resolutions};

pub(crate) fn run(
    req: &mut IrRequest,
    caps: &Capabilities,
    target: Protocol,
    res: &Resolutions,
    degr: &mut Degradations,
) -> Result<(), XlateError> {
    let _ = degr; // this pass rejects rather than degrades; kept for a uniform signature
    let target_family = target.family();

    // (a) Foreign file refs and audio support, per part.
    let audio_ok = caps.media.audio.sources.as_ref().is_some_and(|s| !s.is_empty());
    for_each_part_mut(req, &mut |p: &mut Part| {
        if matches!(p, Part::Audio(_)) && !audio_ok {
            return Err(XlateError::unsupported("audio", "audio input is not supported by the backend"));
        }
        if let Some(ms) = media_source_mut(p) {
            if let MediaSource::FileRef { family, id } = ms {
                if *family != target_family {
                    let key = FileRef::new(family.clone(), id.clone());
                    match res.files.get(&key) {
                        Some(resolved) => {
                            *ms = MediaSource::FileRef {
                                family: resolved.family.clone(),
                                id: resolved.id.clone(),
                            };
                        }
                        None => {
                            return Err(XlateError::unsupported(
                                "file_id",
                                format!(
                                    "file id {:?} from family {} could not be resolved for the \
                                     {} backend",
                                    id, family, target_family
                                ),
                            ));
                        }
                    }
                }
            }
        }
        Ok(())
    })?;

    // (b) Image / PDF byte-size and count limits.
    let mut image_count: u32 = 0;
    let mut pdf_count: u32 = 0;
    for_each_part_ref(req, |p| {
        match p {
            Part::Image(ms) => {
                image_count = image_count.saturating_add(1);
                if let (MediaSource::Base64 { data, .. }, Some(max)) =
                    (ms, caps.media.image.max_bytes)
                {
                    if data.len() as u64 > max {
                        return Err(XlateError::unsupported(
                            "image",
                            format!("image exceeds max_bytes ({} > {max})", data.len()),
                        ));
                    }
                }
            }
            Part::Document { source, media_type, .. } if media_type == "application/pdf" => {
                pdf_count = pdf_count.saturating_add(1);
                if let (MediaSource::Base64 { data, .. }, Some(max)) =
                    (source, caps.media.pdf.max_bytes)
                {
                    if data.len() as u64 > max {
                        return Err(XlateError::unsupported(
                            "document",
                            format!("PDF exceeds max_bytes ({} > {max})", data.len()),
                        ));
                    }
                }
            }
            _ => {}
        }
        Ok(())
    })?;

    if let Some(max) = caps.media.image.max_count {
        if image_count > max {
            return Err(XlateError::unsupported(
                "image",
                format!("too many images ({image_count} > max_count {max})"),
            ));
        }
    }
    if let Some(max) = caps.media.pdf.max_count {
        if pdf_count > max {
            return Err(XlateError::unsupported(
                "document",
                format!("too many PDFs ({pdf_count} > max_count {max})"),
            ));
        }
    }

    Ok(())
}

/// The mutable [`MediaSource`] inside a media-bearing part, if any.
fn media_source_mut(p: &mut Part) -> Option<&mut MediaSource> {
    match p {
        Part::Image(ms) | Part::Audio(ms) => Some(ms),
        Part::Document { source, .. } => Some(source),
        _ => None,
    }
}

/// Visit every part in a request's instructions and items mutably, short-circuiting on error.
fn for_each_part_mut(
    req: &mut IrRequest,
    f: &mut impl FnMut(&mut Part) -> Result<(), XlateError>,
) -> Result<(), XlateError> {
    for ins in &mut req.instructions {
        for p in &mut ins.content {
            f(p)?;
        }
    }
    for item in &mut req.items {
        match item {
            Item::Message { content, .. } | Item::ToolResult { content, .. } => {
                for p in content {
                    f(p)?;
                }
            }
            _ => {}
        }
    }
    Ok(())
}

/// Visit every part immutably, short-circuiting on error.
fn for_each_part_ref(
    req: &IrRequest,
    mut f: impl FnMut(&Part) -> Result<(), XlateError>,
) -> Result<(), XlateError> {
    for ins in &req.instructions {
        for p in &ins.content {
            f(p)?;
        }
    }
    for item in &req.items {
        match item {
            Item::Message { content, .. } | Item::ToolResult { content, .. } => {
                for p in content {
                    f(p)?;
                }
            }
            _ => {}
        }
    }
    Ok(())
}
