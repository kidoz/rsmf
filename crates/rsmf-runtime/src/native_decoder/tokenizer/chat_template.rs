use super::{
    CHAT_TEMPLATE_FUEL, MAX_CHAT_RENDERED_BYTES, MAX_CHAT_TEMPLATE_BYTES, NativeDecoderChatMessage,
};
use crate::{Result, RuntimeError};
use minijinja::{Environment, ErrorKind as JinjaErrorKind, value::Value as JinjaValue};
use serde_json::json;
use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct NativeDecoderChatTemplate {
    template: String,
    globals: HashMap<String, serde_json::Value>,
}

impl NativeDecoderChatTemplate {
    pub(crate) fn from_assets(
        tokenizer_config: Option<&[u8]>,
        chat_template_asset: Option<&[u8]>,
    ) -> Result<Option<Self>> {
        let config = if let Some(bytes) = chat_template_asset {
            Some(NativeDecoderChatTemplateConfig {
                template: chat_template_from_asset(bytes)?,
                globals: tokenizer_config
                    .map(chat_template_globals_from_tokenizer_config)
                    .transpose()?
                    .unwrap_or_default(),
            })
        } else if let Some(bytes) = tokenizer_config {
            chat_template_from_tokenizer_config(bytes)?
        } else {
            None
        };
        Ok(config.map(|config| Self {
            template: config.template,
            globals: config.globals,
        }))
    }

    pub(crate) fn render(
        &self,
        messages: &[NativeDecoderChatMessage],
        add_generation_prompt: bool,
    ) -> Result<String> {
        if messages.is_empty() {
            return Err(RuntimeError::NativeDecoderTokenizerInvalid {
                reason: "chat template requires at least one message".to_string(),
            });
        }
        render_chat_template(
            &self.template,
            &self.globals,
            messages,
            add_generation_prompt,
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct NativeDecoderChatTemplateConfig {
    template: String,
    globals: HashMap<String, serde_json::Value>,
}

pub(crate) fn chat_template_from_tokenizer_config(
    bytes: &[u8],
) -> Result<Option<NativeDecoderChatTemplateConfig>> {
    let value: serde_json::Value = serde_json::from_slice(bytes).map_err(|error| {
        RuntimeError::NativeDecoderTokenizerInvalid {
            reason: format!("tokenizer_config.json is not valid JSON: {error}"),
        }
    })?;
    let Some(template) = value.get("chat_template") else {
        return Ok(None);
    };
    let Some(template) = template.as_str() else {
        return Err(RuntimeError::NativeDecoderTokenizerInvalid {
            reason: "tokenizer_config.json chat_template must be a string".to_string(),
        });
    };
    validate_chat_template_size(template)?;
    Ok(Some(NativeDecoderChatTemplateConfig {
        template: template.to_string(),
        globals: chat_template_globals_from_value(&value),
    }))
}

pub(crate) fn chat_template_from_asset(bytes: &[u8]) -> Result<String> {
    if bytes.len() > MAX_CHAT_TEMPLATE_BYTES {
        return Err(RuntimeError::NativeDecoderTokenizerInvalid {
            reason: format!(
                "chat template asset is too large: {} bytes exceeds {}",
                bytes.len(),
                MAX_CHAT_TEMPLATE_BYTES
            ),
        });
    }
    let text = std::str::from_utf8(bytes).map_err(|error| {
        RuntimeError::NativeDecoderTokenizerInvalid {
            reason: format!("chat_template.json is not UTF-8: {error}"),
        }
    })?;
    if let Ok(value) = serde_json::from_str::<serde_json::Value>(text) {
        if let Some(template) = value.as_str() {
            validate_chat_template_size(template)?;
            return Ok(template.to_string());
        }
        if let Some(template) = value
            .get("chat_template")
            .and_then(serde_json::Value::as_str)
        {
            validate_chat_template_size(template)?;
            return Ok(template.to_string());
        }
        return Err(RuntimeError::NativeDecoderTokenizerInvalid {
            reason: "chat_template.json must be a JSON string or object with chat_template"
                .to_string(),
        });
    }
    validate_chat_template_size(text)?;
    Ok(text.to_string())
}

pub(crate) fn render_chat_template(
    template: &str,
    globals: &HashMap<String, serde_json::Value>,
    messages: &[NativeDecoderChatMessage],
    add_generation_prompt: bool,
) -> Result<String> {
    match render_chat_template_jinja(template, globals, messages, add_generation_prompt) {
        Ok(rendered) => Ok(rendered),
        Err(jinja_error) => {
            if let Ok(rendered) =
                render_chat_template_legacy(template, messages, add_generation_prompt)
            {
                return Ok(rendered);
            }
            Err(jinja_error)
        }
    }
}

pub(crate) fn render_chat_template_jinja(
    template: &str,
    globals: &HashMap<String, serde_json::Value>,
    messages: &[NativeDecoderChatMessage],
    add_generation_prompt: bool,
) -> Result<String> {
    validate_chat_template_size(template)?;
    let mut env = Environment::new();
    env.set_fuel(Some(CHAT_TEMPLATE_FUEL));
    env.add_function(
        "raise_exception",
        |message: String| -> std::result::Result<String, minijinja::Error> {
            Err(minijinja::Error::new(
                JinjaErrorKind::InvalidOperation,
                message,
            ))
        },
    );
    let messages = messages
        .iter()
        .map(|message| json!({ "role": message.role, "content": message.content }))
        .collect::<Vec<_>>();
    let mut context = serde_json::Map::new();
    context.insert("messages".to_string(), serde_json::Value::Array(messages));
    context.insert(
        "add_generation_prompt".to_string(),
        serde_json::Value::Bool(add_generation_prompt),
    );
    context.insert("tools".to_string(), serde_json::Value::Array(Vec::new()));
    context.insert(
        "documents".to_string(),
        serde_json::Value::Array(Vec::new()),
    );
    for (key, value) in globals {
        context.insert(key.clone(), value.clone());
    }
    let rendered = env
        .template_from_str(template)
        .and_then(|template| template.render(JinjaValue::from_serialize(&context)))
        .map_err(|error| RuntimeError::NativeDecoderTokenizerInvalid {
            reason: format!("chat template render failed: {error}"),
        })?;
    if rendered.len() > MAX_CHAT_RENDERED_BYTES {
        return Err(RuntimeError::NativeDecoderTokenizerInvalid {
            reason: format!(
                "chat template rendered {} bytes, maximum supported is {}",
                rendered.len(),
                MAX_CHAT_RENDERED_BYTES
            ),
        });
    }
    Ok(rendered)
}

pub(crate) fn render_chat_template_legacy(
    template: &str,
    messages: &[NativeDecoderChatMessage],
    add_generation_prompt: bool,
) -> Result<String> {
    let Some((before, for_body, after)) = split_jinja_for_messages(template)? else {
        return render_chat_template_segment(template, None, add_generation_prompt);
    };
    let mut rendered = render_chat_template_segment(before, None, add_generation_prompt)?;
    for message in messages {
        rendered.push_str(&render_chat_template_segment(
            for_body,
            Some(message),
            add_generation_prompt,
        )?);
    }
    rendered.push_str(&render_chat_template_segment(
        after,
        None,
        add_generation_prompt,
    )?);
    Ok(rendered)
}

pub(crate) fn validate_chat_template_size(template: &str) -> Result<()> {
    if template.len() > MAX_CHAT_TEMPLATE_BYTES {
        return Err(RuntimeError::NativeDecoderTokenizerInvalid {
            reason: format!(
                "chat template is too large: {} bytes exceeds {}",
                template.len(),
                MAX_CHAT_TEMPLATE_BYTES
            ),
        });
    }
    Ok(())
}

pub(crate) fn chat_template_globals_from_tokenizer_config(
    bytes: &[u8],
) -> Result<HashMap<String, serde_json::Value>> {
    let value: serde_json::Value = serde_json::from_slice(bytes).map_err(|error| {
        RuntimeError::NativeDecoderTokenizerInvalid {
            reason: format!("tokenizer_config.json is not valid JSON: {error}"),
        }
    })?;
    Ok(chat_template_globals_from_value(&value))
}

pub(crate) fn chat_template_globals_from_value(
    value: &serde_json::Value,
) -> HashMap<String, serde_json::Value> {
    let mut globals = HashMap::new();
    for key in [
        "bos_token",
        "eos_token",
        "unk_token",
        "sep_token",
        "pad_token",
        "cls_token",
        "mask_token",
    ] {
        if let Some(value) = value.get(key) {
            globals.insert(key.to_string(), value.clone());
        }
    }
    globals
}

pub(crate) fn split_jinja_for_messages(template: &str) -> Result<Option<(&str, &str, &str)>> {
    let Some((for_start, for_end, for_tag)) = find_jinja_tag(template, 0, "for") else {
        return Ok(None);
    };
    if for_tag != "for message in messages" {
        return Err(RuntimeError::NativeDecoderTokenizerInvalid {
            reason: format!("unsupported chat template for block {for_tag:?}"),
        });
    }
    let Some((end_start, end_end, end_tag)) = find_jinja_tag(template, for_end, "endfor") else {
        return Err(RuntimeError::NativeDecoderTokenizerInvalid {
            reason: "chat template for block is missing endfor".to_string(),
        });
    };
    if end_tag != "endfor" {
        return Err(RuntimeError::NativeDecoderTokenizerInvalid {
            reason: format!("unsupported chat template end block {end_tag:?}"),
        });
    }
    Ok(Some((
        &template[..for_start],
        &template[for_end..end_start],
        &template[end_end..],
    )))
}

pub(crate) fn render_chat_template_segment(
    segment: &str,
    message: Option<&NativeDecoderChatMessage>,
    add_generation_prompt: bool,
) -> Result<String> {
    let mut rendered = String::new();
    let mut rest = segment;
    while let Some(index) = rest.find('{') {
        rendered.push_str(&rest[..index]);
        rest = &rest[index..];
        if rest.starts_with("{{") {
            let end =
                rest.find("}}")
                    .ok_or_else(|| RuntimeError::NativeDecoderTokenizerInvalid {
                        reason: "chat template expression is missing }}".to_string(),
                    })?;
            let expression = rest[2..end].trim().trim_matches('-').trim();
            rendered.push_str(&eval_chat_expression(expression, message)?);
            rest = &rest[end + 2..];
        } else if rest.starts_with("{%") {
            let end =
                rest.find("%}")
                    .ok_or_else(|| RuntimeError::NativeDecoderTokenizerInvalid {
                        reason: "chat template block is missing %}".to_string(),
                    })?;
            let tag = rest[2..end].trim().trim_matches('-').trim();
            if let Some(condition) = tag.strip_prefix("if ") {
                let body_start = end + 2;
                let (branch, next_index) = render_chat_if_block(
                    rest,
                    body_start,
                    condition,
                    message,
                    add_generation_prompt,
                )?;
                rendered.push_str(&branch);
                rest = &rest[next_index..];
            } else {
                return Err(RuntimeError::NativeDecoderTokenizerInvalid {
                    reason: format!("unsupported chat template block {tag:?}"),
                });
            }
        } else {
            rendered.push('{');
            rest = &rest[1..];
        }
    }
    rendered.push_str(rest);
    Ok(rendered)
}

pub(crate) fn render_chat_if_block(
    rest: &str,
    body_start: usize,
    first_condition: &str,
    message: Option<&NativeDecoderChatMessage>,
    add_generation_prompt: bool,
) -> Result<(String, usize)> {
    let mut branches = Vec::new();
    let mut current_condition = Some(first_condition.trim());
    let mut current_body_start = body_start;
    loop {
        let Some((tag_start, tag_end, tag)) = find_next_jinja_control_tag(rest, current_body_start)
        else {
            return Err(RuntimeError::NativeDecoderTokenizerInvalid {
                reason: "chat template if block is missing endif".to_string(),
            });
        };
        if let Some(condition) = tag.strip_prefix("elif ") {
            branches.push((current_condition, current_body_start, tag_start));
            current_condition = Some(condition.trim());
            current_body_start = tag_end;
        } else if tag == "else" {
            branches.push((current_condition, current_body_start, tag_start));
            current_condition = None;
            current_body_start = tag_end;
        } else if tag == "endif" {
            branches.push((current_condition, current_body_start, tag_start));
            for (condition, start, end) in branches {
                let selected = match condition {
                    Some(condition) => {
                        eval_chat_condition(condition, message, add_generation_prompt)?
                    }
                    None => true,
                };
                if selected {
                    return Ok((
                        render_chat_template_segment(
                            &rest[start..end],
                            message,
                            add_generation_prompt,
                        )?,
                        tag_end,
                    ));
                }
            }
            return Ok((String::new(), tag_end));
        } else {
            return Err(RuntimeError::NativeDecoderTokenizerInvalid {
                reason: format!("unsupported chat template if control block {tag:?}"),
            });
        }
    }
}

pub(crate) fn find_next_jinja_control_tag(
    template: &str,
    start: usize,
) -> Option<(usize, usize, &str)> {
    let mut search_start = start;
    while let Some(relative_start) = template[search_start..].find("{%") {
        let tag_start = search_start + relative_start;
        let tag_body_start = tag_start + 2;
        let relative_end = template[tag_body_start..].find("%}")?;
        let tag_end = tag_body_start + relative_end + 2;
        let tag = template[tag_body_start..tag_body_start + relative_end]
            .trim()
            .trim_matches('-')
            .trim();
        if tag.starts_with("elif ") || tag == "else" || tag == "endif" {
            return Some((tag_start, tag_end, tag));
        }
        search_start = tag_end;
    }
    None
}

pub(crate) fn eval_chat_condition(
    condition: &str,
    message: Option<&NativeDecoderChatMessage>,
    add_generation_prompt: bool,
) -> Result<bool> {
    match condition {
        "add_generation_prompt" => return Ok(add_generation_prompt),
        "not add_generation_prompt" => return Ok(!add_generation_prompt),
        _ => {}
    }
    if let Some((left, right)) = condition.split_once("==") {
        return Ok(eval_chat_condition_value(left.trim(), message)?
            == eval_chat_condition_literal(right.trim())?);
    }
    if let Some((left, right)) = condition.split_once("!=") {
        return Ok(eval_chat_condition_value(left.trim(), message)?
            != eval_chat_condition_literal(right.trim())?);
    }
    Err(RuntimeError::NativeDecoderTokenizerInvalid {
        reason: format!("unsupported chat template condition {condition:?}"),
    })
}

pub(crate) fn eval_chat_condition_value(
    expression: &str,
    message: Option<&NativeDecoderChatMessage>,
) -> Result<String> {
    match expression {
        "message['role']" | "message[\"role\"]" | "message.role" => message
            .map(|message| message.role.clone())
            .ok_or_else(|| RuntimeError::NativeDecoderTokenizerInvalid {
                reason: "chat template message role used outside message loop".to_string(),
            }),
        "message['content']" | "message[\"content\"]" | "message.content" => message
            .map(|message| message.content.clone())
            .ok_or_else(|| RuntimeError::NativeDecoderTokenizerInvalid {
                reason: "chat template message content used outside message loop".to_string(),
            }),
        other => Err(RuntimeError::NativeDecoderTokenizerInvalid {
            reason: format!("unsupported chat template condition value {other:?}"),
        }),
    }
}

pub(crate) fn eval_chat_condition_literal(value: &str) -> Result<String> {
    if (value.starts_with('\'') && value.ends_with('\''))
        || (value.starts_with('"') && value.ends_with('"'))
    {
        Ok(unescape_chat_literal(&value[1..value.len() - 1]))
    } else {
        Err(RuntimeError::NativeDecoderTokenizerInvalid {
            reason: format!("unsupported chat template condition literal {value:?}"),
        })
    }
}

pub(crate) fn find_jinja_tag<'a>(
    template: &'a str,
    start: usize,
    expected_prefix: &str,
) -> Option<(usize, usize, &'a str)> {
    let mut search_start = start;
    while let Some(relative_start) = template[search_start..].find("{%") {
        let tag_start = search_start + relative_start;
        let tag_body_start = tag_start + 2;
        let relative_end = template[tag_body_start..].find("%}")?;
        let tag_end = tag_body_start + relative_end + 2;
        let tag = template[tag_body_start..tag_body_start + relative_end]
            .trim()
            .trim_matches('-')
            .trim();
        if tag.starts_with(expected_prefix) {
            return Some((tag_start, tag_end, tag));
        }
        search_start = tag_end;
    }
    None
}

pub(crate) fn eval_chat_expression(
    expression: &str,
    message: Option<&NativeDecoderChatMessage>,
) -> Result<String> {
    let mut output = String::new();
    for part in split_chat_concat(expression) {
        output.push_str(&eval_chat_expression_part(part.trim(), message)?);
    }
    Ok(output)
}

pub(crate) fn split_chat_concat(expression: &str) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut start = 0usize;
    let mut quote = None;
    let mut escape = false;
    for (index, ch) in expression.char_indices() {
        if escape {
            escape = false;
            continue;
        }
        if ch == '\\' {
            escape = true;
            continue;
        }
        if let Some(active_quote) = quote {
            if ch == active_quote {
                quote = None;
            }
            continue;
        }
        if ch == '\'' || ch == '"' {
            quote = Some(ch);
        } else if ch == '+' {
            parts.push(&expression[start..index]);
            start = index + ch.len_utf8();
        }
    }
    parts.push(&expression[start..]);
    parts
}

pub(crate) fn eval_chat_expression_part(
    part: &str,
    message: Option<&NativeDecoderChatMessage>,
) -> Result<String> {
    if (part.starts_with('\'') && part.ends_with('\''))
        || (part.starts_with('"') && part.ends_with('"'))
    {
        return Ok(unescape_chat_literal(&part[1..part.len() - 1]));
    }
    if matches!(
        part,
        "message['role']" | "message[\"role\"]" | "message.role"
    ) {
        return message.map(|message| message.role.clone()).ok_or_else(|| {
            RuntimeError::NativeDecoderTokenizerInvalid {
                reason: "chat template message role used outside message loop".to_string(),
            }
        });
    }
    if matches!(
        part,
        "message['content']" | "message[\"content\"]" | "message.content"
    ) {
        return message
            .map(|message| message.content.clone())
            .ok_or_else(|| RuntimeError::NativeDecoderTokenizerInvalid {
                reason: "chat template message content used outside message loop".to_string(),
            });
    }
    Err(RuntimeError::NativeDecoderTokenizerInvalid {
        reason: format!("unsupported chat template expression part {part:?}"),
    })
}

pub(crate) fn unescape_chat_literal(literal: &str) -> String {
    let mut output = String::new();
    let mut chars = literal.chars();
    while let Some(ch) = chars.next() {
        if ch != '\\' {
            output.push(ch);
            continue;
        }
        match chars.next() {
            Some('n') => output.push('\n'),
            Some('r') => output.push('\r'),
            Some('t') => output.push('\t'),
            Some('\\') => output.push('\\'),
            Some('\'') => output.push('\''),
            Some('"') => output.push('"'),
            Some(other) => {
                output.push('\\');
                output.push(other);
            }
            None => output.push('\\'),
        }
    }
    output
}
