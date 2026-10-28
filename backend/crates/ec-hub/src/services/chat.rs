//! Python: backend/src/engine_console/services/chat.py
//! (tests/core/test_nodehttp_chat_bench.py, tests/core/test_api_smoke.py chat/arena parts).
//!
//! Chat completions relayed to an instance's engine through its node (`NodeHttp`, SSE passthrough with persistence and
//! usage), conversations and prompts in SQLite, and the arena (blind A/B matches + Elo ratings). The routes use the
//! methods and `ChatStream` below.

use std::sync::Arc;
use std::time::Instant;

use futures::stream::BoxStream;
use serde_json::{Map, Value, json};

use crate::domain::errors::{
    AppResult, Problem, bad_request, conflict, internal, not_found, upstream_error,
};
use crate::domain::models::{
    ArenaMatchIn, Conversation, ConversationIn, Instance, Page, Prompt, PromptIn,
};
use crate::services::common::{new_id, now};
use crate::services::gguf_info::py;
use crate::services::hf::py_round;
use crate::services::lifecycle::LifecycleService;
use crate::services::nodehttp::{HttpOpts, JSON_HEADERS, NodeHttp};
use crate::services::store::{SqlRow, Store, paginate};
use crate::services::usage::UsageService;
use crate::sync::Recover as _;

pub const ELO_K: f64 = 32.0;
pub const ELO_START: f64 = 1000.0;
pub const CHAT_PATH: &str = "/v1/chat/completions";
/// a long generation may stream for a long time; the node enforces this bound
pub const STREAM_TIMEOUT_S: f64 = 3600.0;
pub const COMPLETE_TIMEOUT_S: f64 = 600.0;
/// an error answer is read under this cap; only the first 500 characters are shown
pub const ERROR_BODY_CAP: usize = 64 * 1024;

/// An opened streamed reply: raw SSE lines of the engine (each ending in "\n"), relayed as they come. Dropping it
/// (a browser that went away) cancels the engine request on the node and still records what was received (on the
/// blocking pool, never in `Drop` itself).
pub struct ChatStream {
    pub lines: BoxStream<'static, String>,
}

impl std::fmt::Debug for ChatStream {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("ChatStream")
    }
}

/// What a streamed reply writes when it ends (Python's `on_done`): its usage row and, in a conversation, the user's
/// last message and the answer received so far.
struct Reply {
    db: Arc<Store>,
    usage: Arc<UsageService>,
    conv: Option<String>,
    sent: Vec<Value>,
    inst: Instance,
    text: String,
    reasoning: String,
    usage_obj: Map<String, Value>,
    latency_ms: f64,
}

impl Reply {
    fn write(self) {
        persist(
            &self.db,
            &self.usage,
            self.conv.as_deref(),
            &self.sent,
            &self.inst,
            &self.text,
            &self.reasoning,
            &self.usage_obj,
            self.latency_ms,
        );
    }
}

/// Writes replies off the async workers, on the runtime's blocking pool, and counts the ones not written yet so the
/// lifespan's end can wait for them ([`ChatService::flush`]): a reply handed off by a stream dropped during the
/// shutdown's drain is still written before the runtime goes away.
#[derive(Default)]
struct ReplyWriter {
    inflight: std::sync::Mutex<usize>,
    idle: tokio::sync::Notify,
}

/// Counts a handed-off reply out when its job ends, however it ends (written, panicked, or dropped unrun).
struct Written(Arc<ReplyWriter>);

impl Drop for Written {
    fn drop(&mut self) {
        let mut n = self.0.inflight.guard();
        *n = n.saturating_sub(1);
        if *n == 0 {
            self.0.idle.notify_waiters();
        }
    }
}

impl ReplyWriter {
    /// Write `reply` on the blocking pool; the handle ends once it is written. Outside a runtime (no async worker to
    /// stall: only synchronous code gets here) it is written at once and there is no handle.
    fn submit(self: &Arc<Self>, reply: Reply) -> Option<tokio::task::JoinHandle<()>> {
        let Ok(rt) = tokio::runtime::Handle::try_current() else {
            reply.write();
            return None;
        };
        *self.inflight.guard() += 1;
        let written = Written(Arc::clone(self));
        Some(rt.spawn_blocking(move || {
            let _written = written;
            reply.write();
        }))
    }

    /// Wait until every reply handed off so far is written.
    async fn flush(&self) {
        loop {
            // registered before the count is read: a notify between the two is not missed
            let idle = self.idle.notified();
            if *self.inflight.guard() == 0 {
                return;
            }
            idle.await;
        }
    }
}

/// Collects a streamed reply and hands it to the [`ReplyWriter`] when the stream ends, however it ends (Python's
/// `on_done` in the stream's `finally`).
struct Recorder {
    /// None once handed off
    reply: Option<Reply>,
    writer: Arc<ReplyWriter>,
    started: Instant,
}

impl Recorder {
    /// `ChatStream._observe`: collect the text, the reasoning and the usage of one SSE line.
    fn observe(&mut self, line: &str) {
        let Some(r) = self.reply.as_mut() else {
            return;
        };
        if !line.starts_with("data:") || py::strip(line) == "data: [DONE]" {
            return;
        }
        let Ok(Value::Object(ev)) = serde_json::from_str::<Value>(&line[5..]) else {
            return;
        };
        if let Some(Value::Object(u)) = ev.get("usage").filter(|u| py::truthy(u)) {
            r.usage_obj = u.clone();
        }
        if let Some(Value::Array(choices)) = ev.get("choices") {
            for ch in choices {
                let Some(d) = ch.get("delta").and_then(Value::as_object) else {
                    continue;
                };
                if let Some(Value::String(c)) = d.get("content").filter(|v| py::truthy(v)) {
                    r.text.push_str(c);
                }
                if let Some(x) = py::or_chain(&[d.get("reasoning_content"), d.get("reasoning")])
                    .filter(|v| py::truthy(v))
                {
                    r.reasoning.push_str(&ec_proto::pyfmt::str_value(x));
                }
            }
        }
    }

    /// The reply, once (later calls: None), with its latency.
    fn take(&mut self) -> Option<Reply> {
        let mut r = self.reply.take()?;
        r.latency_ms = self.started.elapsed().as_secs_f64() * 1000.0;
        Some(r)
    }

    /// The stream's own end: the reply is written off the async worker, and the stream ends once it is written, as
    /// Python's `finally` persists before the response ends.
    async fn finish(&mut self) {
        if let Some(job) = self.take().and_then(|r| self.writer.submit(r)) {
            let _ = job.await;
        }
    }
}

impl Drop for Recorder {
    /// A stream dropped before its end (a browser that went away). `Drop` runs on an async worker, inside the
    /// connection's poll, so the reply received so far is only handed to the writer here, never written.
    fn drop(&mut self) {
        if let Some(r) = self.take() {
            drop(self.writer.submit(r));
        }
    }
}

/// `int(usage.get(k, 0) or 0)`
fn token_count(u: &Map<String, Value>, k: &str) -> i64 {
    match u.get(k) {
        Some(Value::Number(n)) => n.as_i64().unwrap_or_else(|| {
            #[allow(clippy::cast_possible_truncation)]
            let v = n.as_f64().unwrap_or(0.0).trunc() as i64;
            v
        }),
        Some(Value::String(s)) => crate::services::store::py_int(s).unwrap_or(0),
        Some(Value::Bool(b)) => i64::from(*b),
        _ => 0,
    }
}

fn add_message(
    db: &Store,
    conv: &str,
    role: &str,
    content: &str,
    reasoning: Option<&str>,
    usage: Option<&Map<String, Value>>,
) {
    let usage_text = usage
        .filter(|u| !u.is_empty())
        .map(|u| crate::pyjson::dumps(&Value::Object(u.clone())));
    let _ = db.execute(
        "INSERT INTO messages(conversation_id,role,content,reasoning,usage,ts) VALUES(?,?,?,?,?,?)",
        &[&conv, &role, &content, &reasoning, &usage_text, &now()],
    );
    let _ = db.execute(
        "UPDATE conversations SET updated_at=? WHERE id=?",
        &[&now(), &conv],
    );
}

#[allow(clippy::too_many_arguments)]
fn persist(
    db: &Store,
    usage: &UsageService,
    conv: Option<&str>,
    sent: &[Value],
    inst: &Instance,
    text: &str,
    reasoning: &str,
    usage_obj: &Map<String, Value>,
    latency_ms: f64,
) {
    let (pt, ct) = (
        token_count(usage_obj, "prompt_tokens"),
        token_count(usage_obj, "completion_tokens"),
    );
    let _ = usage.record(&inst.repo_id, Some(&inst.id), pt, ct, latency_ms, "chat");
    if let Some(conv) = conv {
        let last_user = sent
            .iter()
            .rev()
            .find(|m| m.get("role").and_then(Value::as_str) == Some("user"));
        if let Some(m) = last_user {
            let c = m.get("content").cloned().unwrap_or(Value::Null);
            let content = match &c {
                Value::String(s) => s.clone(),
                other => crate::pyjson::dumps(other),
            };
            add_message(db, conv, "user", &content, None, None);
        }
        add_message(
            db,
            conv,
            "assistant",
            text,
            (!reasoning.is_empty()).then_some(reasoning),
            (!usage_obj.is_empty()).then_some(usage_obj),
        );
    }
}

fn engine_error(status: i64, detail: &[u8]) -> Problem {
    let text = py::take(&String::from_utf8_lossy(detail), 500);
    let mut p = upstream_error(format!("engine returned HTTP {status}: {text}"));
    if matches!(status, 400 | 404 | 422) {
        p.status = u16::try_from(status).unwrap_or(502);
    }
    p
}

pub struct ChatService {
    db: Arc<Store>,
    lifecycle: Arc<LifecycleService>,
    http: Arc<NodeHttp>,
    usage: Arc<UsageService>,
    writer: Arc<ReplyWriter>,
}

impl std::fmt::Debug for ChatService {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("ChatService")
    }
}

fn s_of(v: &Value) -> String {
    ec_proto::pyfmt::str_value(v)
}

impl ChatService {
    pub fn new(
        db: Arc<Store>,
        lifecycle: Arc<LifecycleService>,
        http: Arc<NodeHttp>,
        usage: Arc<UsageService>,
    ) -> Self {
        Self {
            db,
            lifecycle,
            http,
            usage,
            writer: Arc::default(),
        }
    }

    /// Wait until every streamed reply handed off so far (by a stream that ended or was dropped) is written. The
    /// lifespan's end calls it after the connections are closed.
    pub async fn flush(&self) {
        self.writer.flush().await;
    }

    // -- proxy ------------------------------------------------------------------------------------------------
    fn ready(&self, iid: &str) -> AppResult<Instance> {
        let inst = self.lifecycle.get(iid)?;
        if inst.state != "ready" || inst.port.is_none_or(|p| p == 0) {
            return Err(
                conflict(format!("instance {} is {}", inst.name, inst.state))
                    .code("instance_not_ready")
                    .into(),
            );
        }
        Ok(inst)
    }

    /// `_prepare`: the instance, the payload sent to the engine and the conversation.
    fn prepare(&self, body: &Value) -> AppResult<(Instance, Map<String, Value>, Option<String>)> {
        let mut body = body.as_object().cloned().unwrap_or_default();
        let iid = body.shift_remove("instance_id");
        let conv = body.shift_remove("conversation_id").filter(py::truthy);
        let Some(iid) = iid.filter(py::truthy) else {
            return Err(bad_request("instance_id is required")
                .code("missing_instance_id")
                .into());
        };
        match body.get("messages") {
            Some(Value::Array(a)) if !a.is_empty() => {}
            _ => {
                return Err(bad_request("messages must be a non-empty list")
                    .code("invalid_messages")
                    .into());
            }
        }
        let inst = self.ready(&s_of(&iid))?;
        let conv = conv.map(|c| s_of(&c));
        if let Some(c) = &conv {
            self.get_conversation(c, false)?;
        }
        body.insert("model".into(), Value::String(inst.repo_id.clone())); // served_name = repo_id
        Ok((inst, body, conv))
    }

    fn recorder(
        &self,
        conv: Option<String>,
        payload: &Map<String, Value>,
        inst: &Instance,
    ) -> Recorder {
        Recorder {
            reply: Some(Reply {
                db: Arc::clone(&self.db),
                usage: Arc::clone(&self.usage),
                conv,
                sent: payload
                    .get("messages")
                    .and_then(Value::as_array)
                    .cloned()
                    .unwrap_or_default(),
                inst: inst.clone(),
                text: String::new(),
                reasoning: String::new(),
                usage_obj: Map::new(),
                latency_ms: 0.0,
            }),
            writer: Arc::clone(&self.writer),
            started: Instant::now(),
        }
    }

    /// `stream: true`: open the engine response (refusals happen here, before any byte is sent).
    pub async fn open_stream(&self, body: &Value) -> AppResult<ChatStream> {
        let (inst, mut payload, conv) = self.prepare(body)?;
        payload.insert("stream".into(), Value::Bool(true));
        match payload.get_mut("stream_options") {
            None => {
                payload.insert("stream_options".into(), json!({"include_usage": true}));
            }
            Some(Value::Object(o)) => {
                o.insert("include_usage".into(), Value::Bool(true));
            }
            // `payload.setdefault(...)["include_usage"] = True` on a non-dict raises in Python
            Some(_) => return Err(internal("unexpected server error").into()),
        }
        self.lifecycle.touch(&inst.id);
        let mut headers: Vec<(String, String)> = JSON_HEADERS
            .iter()
            .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
            .collect();
        if let Some(a) = headers.iter_mut().find(|(k, _)| k == "accept") {
            a.1 = "text/event-stream".into();
        }
        let mut resp = self
            .http
            .open(
                &inst.host_id,
                inst.port.unwrap_or(0),
                "POST",
                CHAT_PATH,
                HttpOpts {
                    body: Some(crate::pyjson::dumps(&Value::Object(payload.clone())).into_bytes()),
                    headers,
                    timeout_s: Some(STREAM_TIMEOUT_S),
                    max_bytes: None,
                },
            )
            .await?;
        if resp.status >= 400 {
            let status = resp.status;
            // `try: read finally: aclose()`: read() closes the response however it ends
            let detail = resp.read(ERROR_BODY_CAP).await?;
            return Err(engine_error(status, &detail).into());
        }
        let mut rec = self.recorder(conv, &payload, &inst);
        let lines = async_stream::stream! {
            while let Ok(Some(line)) = resp.line().await {
                rec.observe(&line);
                yield format!("{line}\n");
            }
            resp.close().await;
            rec.finish().await;
        };
        Ok(ChatStream {
            lines: Box::pin(lines),
        })
    }

    /// `stream: false`: the engine's JSON answer.
    pub async fn complete(&self, body: &Value) -> AppResult<Value> {
        let (inst, mut payload, conv) = self.prepare(body)?;
        payload.insert("stream".into(), Value::Bool(false));
        self.lifecycle.touch(&inst.id);
        let t0 = Instant::now();
        let r = self
            .http
            .request(
                &inst.host_id,
                inst.port.unwrap_or(0),
                "POST",
                CHAT_PATH,
                HttpOpts {
                    body: Some(crate::pyjson::dumps(&Value::Object(payload.clone())).into_bytes()),
                    headers: JSON_HEADERS
                        .iter()
                        .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
                        .collect(),
                    timeout_s: Some(COMPLETE_TIMEOUT_S),
                    max_bytes: None,
                },
            )
            .await?;
        if r.status >= 400 {
            let cap = r.body.len().min(ERROR_BODY_CAP);
            return Err(engine_error(r.status, &r.body[..cap]).into());
        }
        let data: Value = serde_json::from_slice(&r.body)
            .map_err(|_| upstream_error("the engine did not return JSON"))?;
        let Value::Object(obj) = &data else {
            return Err(upstream_error("the engine returned an unexpected answer").into());
        };
        let first = match obj.get("choices") {
            Some(Value::Array(a)) if !a.is_empty() => a[0].clone(),
            _ => json!({}),
        };
        let msg = first
            .get("message")
            .filter(|m| py::truthy(m))
            .and_then(Value::as_object)
            .cloned()
            .unwrap_or_default();
        let text = msg
            .get("content")
            .filter(|v| py::truthy(v))
            .map(s_of)
            .unwrap_or_default();
        let reasoning = py::or_chain(&[msg.get("reasoning_content"), msg.get("reasoning")])
            .filter(|v| py::truthy(v))
            .map(s_of)
            .unwrap_or_default();
        let usage = obj
            .get("usage")
            .filter(|u| py::truthy(u))
            .and_then(Value::as_object)
            .cloned()
            .unwrap_or_default();
        let sent = payload
            .get("messages")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        persist(
            &self.db,
            &self.usage,
            conv.as_deref(),
            &sent,
            &inst,
            &text,
            &reasoning,
            &usage,
            t0.elapsed().as_secs_f64() * 1000.0,
        );
        Ok(data)
    }

    // -- conversations ------------------------------------------------------------------------------------------
    pub fn create_conversation(&self, c: &ConversationIn) -> AppResult<Conversation> {
        let (cid, ts) = (new_id("conv_"), now());
        self.db.execute(
            "INSERT INTO conversations VALUES(?,?,?,?,?,?)",
            &[&cid, &c.title, &c.system_prompt, &c.instance_id, &ts, &ts],
        )?;
        self.get_conversation(&cid, false)
    }

    pub fn get_conversation(&self, cid: &str, with_messages: bool) -> AppResult<Conversation> {
        let Some(r) = self
            .db
            .one("SELECT * FROM conversations WHERE id=?", &[&cid])?
        else {
            return Err(not_found(format!("no such conversation: {cid}")).into());
        };
        let messages = if with_messages {
            let rows = self.db.all(
                "SELECT * FROM messages WHERE conversation_id=? ORDER BY id",
                &[&cid],
            )?;
            Some(
                rows.iter()
                    .map(|m| {
                        let usage = m
                            .text("usage")
                            .filter(|u| !u.is_empty())
                            .and_then(|u| serde_json::from_str::<Value>(&u).ok())
                            .unwrap_or(Value::Null);
                        json!({"role": m.json("role"), "content": m.json("content"), "reasoning": m.json("reasoning"),
                               "ts": m.json("ts"), "usage": usage})
                        .as_object()
                        .cloned()
                        .unwrap_or_default()
                    })
                    .collect(),
            )
        } else {
            None
        };
        Ok(Conversation {
            id: r.str_or("id", ""),
            title: r.str_or("title", ""),
            system_prompt: r.text("system_prompt"),
            instance_id: r.text("instance_id"),
            created_at: r.f64("created_at").unwrap_or(0.0),
            updated_at: r.f64("updated_at").unwrap_or(0.0),
            messages,
        })
    }

    pub fn list_conversations(
        &self,
        limit: i64,
        cursor: Option<&str>,
    ) -> AppResult<Page<Conversation>> {
        let (limit, off) = paginate(limit, cursor);
        let rows = self.db.all(
            "SELECT id FROM conversations ORDER BY updated_at DESC LIMIT ? OFFSET ?",
            &[&(limit + 1), &off],
        )?;
        let lim = usize::try_from(limit).unwrap_or(0);
        let items = rows
            .iter()
            .take(lim)
            .map(|r| self.get_conversation(&r.str_or("id", ""), false))
            .collect::<AppResult<Vec<_>>>()?;
        Ok(Page::new(
            items,
            (rows.len() > lim).then(|| (off + limit).to_string()),
        ))
    }

    pub fn delete_conversation(&self, cid: &str) -> AppResult<()> {
        self.get_conversation(cid, false)?;
        self.db
            .execute("DELETE FROM conversations WHERE id=?", &[&cid])?;
        Ok(())
    }

    // -- prompts --------------------------------------------------------------------------------------------------
    fn prompt(r: &SqlRow) -> Prompt {
        let tags = r
            .text("tags")
            .and_then(|t| serde_json::from_str::<Vec<String>>(&t).ok())
            .unwrap_or_default();
        Prompt {
            title: r.str_or("title", ""),
            content: r.str_or("content", ""),
            tags,
            id: r.str_or("id", ""),
            created_at: r.f64("created_at").unwrap_or(0.0),
            updated_at: r.f64("updated_at").unwrap_or(0.0),
        }
    }

    pub fn create_prompt(&self, p: &PromptIn) -> AppResult<Prompt> {
        let (pid, ts) = (new_id("prm_"), now());
        let tags = crate::pyjson::dumps(&Value::from(p.tags.clone()));
        self.db.execute(
            "INSERT INTO prompts VALUES(?,?,?,?,?,?)",
            &[&pid, &p.title, &p.content, &tags, &ts, &ts],
        )?;
        self.get_prompt(&pid)
    }

    pub fn get_prompt(&self, pid: &str) -> AppResult<Prompt> {
        match self.db.one("SELECT * FROM prompts WHERE id=?", &[&pid])? {
            Some(r) => Ok(Self::prompt(&r)),
            None => Err(not_found(format!("no such prompt: {pid}")).into()),
        }
    }

    pub fn list_prompts(&self, limit: i64, cursor: Option<&str>) -> AppResult<Page<Prompt>> {
        let (limit, off) = paginate(limit, cursor);
        let rows = self.db.all(
            "SELECT * FROM prompts ORDER BY title LIMIT ? OFFSET ?",
            &[&(limit + 1), &off],
        )?;
        let lim = usize::try_from(limit).unwrap_or(0);
        Ok(Page::new(
            rows.iter().take(lim).map(Self::prompt).collect(),
            (rows.len() > lim).then(|| (off + limit).to_string()),
        ))
    }

    pub fn update_prompt(&self, pid: &str, p: &PromptIn) -> AppResult<Prompt> {
        self.get_prompt(pid)?;
        let tags = crate::pyjson::dumps(&Value::from(p.tags.clone()));
        self.db.execute(
            "UPDATE prompts SET title=?, content=?, tags=?, updated_at=? WHERE id=?",
            &[&p.title, &p.content, &tags, &now(), &pid],
        )?;
        self.get_prompt(pid)
    }

    pub fn delete_prompt(&self, pid: &str) -> AppResult<()> {
        self.get_prompt(pid)?;
        self.db.execute("DELETE FROM prompts WHERE id=?", &[&pid])?;
        Ok(())
    }

    // -- arena ------------------------------------------------------------------------------------------------------
    async fn ask(&self, inst: &Instance, msgs: &[Value], max_tokens: i64) -> AppResult<String> {
        let data = self
            .complete(&json!({"instance_id": inst.id, "messages": msgs, "max_tokens": max_tokens}))
            .await?;
        let first = match data.get("choices") {
            Some(Value::Array(a)) if !a.is_empty() => a[0].clone(),
            _ => json!({}),
        };
        let content = first
            .get("message")
            .filter(|m| py::truthy(m))
            .and_then(|m| m.get("content").cloned())
            .filter(py::truthy);
        Ok(content.map(|c| s_of(&c)).unwrap_or_default())
    }

    pub async fn create_match(&self, m: &ArenaMatchIn) -> AppResult<Value> {
        if m.instance_a == m.instance_b {
            return Err(bad_request("pick two different instances")
                .code("same_instance")
                .into());
        }
        let (ia, ib) = (self.ready(&m.instance_a)?, self.ready(&m.instance_b)?);
        let mut msgs = Vec::new();
        if let Some(sp) = m.system_prompt.as_ref().filter(|s| !s.is_empty()) {
            msgs.push(json!({"role": "system", "content": sp}));
        }
        msgs.push(json!({"role": "user", "content": m.prompt}));
        let (ra, rb) = tokio::join!(
            self.ask(&ia, &msgs, m.max_tokens),
            self.ask(&ib, &msgs, m.max_tokens)
        );
        let (ra, rb) = (ra?, rb?);
        let (mid, ts) = (new_id("match_"), now());
        self.db.execute(
            "INSERT INTO arena_matches(id,prompt,system_prompt,blind,model_a,model_b,response_a,response_b,created_at) \
             VALUES(?,?,?,?,?,?,?,?,?)",
            &[
                &mid,
                &m.prompt,
                &m.system_prompt,
                &i64::from(m.blind),
                &ia.repo_id,
                &ib.repo_id,
                &ra,
                &rb,
                &ts,
            ],
        )?;
        self.match_(&mid)
    }

    fn match_(&self, mid: &str) -> AppResult<Value> {
        let Some(r) = self
            .db
            .one("SELECT * FROM arena_matches WHERE id=?", &[&mid])?
        else {
            return Err(not_found(format!("no such match: {mid}")).into());
        };
        let reveal = !r.bool("blind") || !r.json("winner").is_null();
        Ok(
            json!({"id": r.json("id"), "prompt": r.json("prompt"), "blind": r.bool("blind"),
                  "response_a": r.json("response_a"), "response_b": r.json("response_b"), "winner": r.json("winner"),
                  "model_a": if reveal { r.json("model_a") } else { Value::Null },
                  "model_b": if reveal { r.json("model_b") } else { Value::Null }}),
        )
    }

    fn rating(&self, model: &str) -> AppResult<f64> {
        Ok(self
            .db
            .one("SELECT rating FROM arena_ratings WHERE model=?", &[&model])?
            .and_then(|r| r.f64("rating"))
            .unwrap_or(ELO_START))
    }

    pub fn vote(&self, mid: &str, winner: &str) -> AppResult<Value> {
        let Some(r) = self
            .db
            .one("SELECT * FROM arena_matches WHERE id=?", &[&mid])?
        else {
            return Err(not_found(format!("no such match: {mid}")).into());
        };
        if !r.json("winner").is_null() {
            return Err(conflict("this match already has a vote")
                .code("already_voted")
                .into());
        }
        let (a, b) = (r.str_or("model_a", ""), r.str_or("model_b", ""));
        self.db.execute(
            "UPDATE arena_matches SET winner=?, voted_at=? WHERE id=?",
            &[&winner, &now(), &mid],
        )?;
        let with_ratings = |m: Value, ratings: Value| -> Value {
            let mut o = m.as_object().cloned().unwrap_or_default();
            o.insert("ratings".into(), ratings);
            Value::Object(o)
        };
        if a == b {
            // two instances of one model: a vote says nothing about relative strength
            let mut rt = Map::new();
            rt.insert(a.clone(), Value::from(py_round(self.rating(&a)?, 1)));
            return Ok(with_ratings(self.match_(mid)?, Value::Object(rt)));
        }
        let (ra, rb) = (self.rating(&a)?, self.rating(&b)?);
        let ea = 1.0 / (1.0 + 10f64.powf((rb - ra) / 400.0));
        let sa = match winner {
            "a" => 1.0,
            "b" => 0.0,
            "tie" => 0.5,
            _ => return Err(internal("unexpected server error").into()),
        };
        let na = ra + ELO_K * (sa - ea);
        let nb = rb + ELO_K * ((1.0 - sa) - (1.0 - ea));
        #[allow(clippy::float_cmp)]
        for (model, new, score) in [(&a, na, sa), (&b, nb, 1.0 - sa)] {
            self.db.execute(
                "INSERT INTO arena_ratings(model,rating,games,wins,losses,ties) VALUES(?,?,1,?,?,?) ON CONFLICT(model) \
                 DO UPDATE SET rating=excluded.rating, games=games+1, wins=wins+excluded.wins, \
                 losses=losses+excluded.losses, ties=ties+excluded.ties",
                &[
                    model,
                    &new,
                    &i64::from(score == 1.0),
                    &i64::from(score == 0.0),
                    &i64::from(score == 0.5),
                ],
            )?;
        }
        let mut rt = Map::new();
        rt.insert(a, Value::from(py_round(na, 1)));
        rt.insert(b, Value::from(py_round(nb, 1)));
        Ok(with_ratings(self.match_(mid)?, Value::Object(rt)))
    }

    pub fn leaderboard(&self) -> AppResult<Vec<Value>> {
        Ok(self
            .db
            .all("SELECT * FROM arena_ratings ORDER BY rating DESC", &[])?
            .iter()
            .map(|r| {
                json!({"model": r.json("model"), "rating": py_round(r.f64("rating").unwrap_or(0.0), 1),
                       "games": r.json("games"), "wins": r.json("wins"), "losses": r.json("losses"), "ties": r.json("ties")})
            })
            .collect())
    }
}
