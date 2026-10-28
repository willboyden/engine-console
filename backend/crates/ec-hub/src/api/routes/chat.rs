//! /chat/completions, /conversations, /prompts, /arena (port of api/routers/chat.py).

use serde_json::Value;

use crate::api::deps::{slot_stream, take_stream_slot};
use crate::api::http::{Ctx, Fail, HResult, json, no_content, ok};
use crate::api::schemas::{ARENA_MATCH_IN, ARENA_VOTE, CONVERSATION_IN, PROMPT_IN};
use crate::api::validate::{Params, T};
use crate::domain::errors::internal;
use crate::domain::models::{ArenaMatchIn, ArenaVote, ConversationIn, PromptIn};

fn finish(p: Params) -> Result<(), Fail> {
    if p.ok() {
        Ok(())
    } else {
        Err(Fail::Validation(p.errs))
    }
}

async fn body_of<D: serde::de::DeserializeOwned>(ctx: &mut Ctx, t: &T) -> Result<D, Fail> {
    let body = ctx.body().await?;
    let mut p = Params::default();
    let v = p.body(&body, t, true);
    finish(p)?;
    serde_json::from_value(v).map_err(|_| Fail::from(internal("unexpected server error")))
}

/// OpenAI-shaped body plus `instance_id` and optional `conversation_id`; SSE passthrough when stream=true, at most
/// MAX_CHAT_STREAMS at once and MAX_CHAT_STREAMS_PER_KEY per key (429 too_many_chat_streams).
pub async fn chat_completions(mut ctx: Ctx) -> HResult {
    let body: Value = body_of(&mut ctx, &T::AnyDict).await?;
    if body.get("stream").is_some_and(ec_proto::sanitize::truthy) {
        let slot = take_stream_slot(
            &ctx,
            &ctx.c.chat_streams,
            "streamed chat replies",
            "too_many_chat_streams",
        )?;
        // refused or failed before streaming: the slot is given back as `slot` drops
        let stream = ctx.c.chat.open_stream(&body).await?;
        return slot_stream(stream.lines, slot);
    }
    ok(&ctx.c.chat.complete(&body).await?)
}

pub async fn list_conversations(ctx: Ctx) -> HResult {
    let mut p = Params::default();
    let limit = p.int_q(&ctx.query, "limit", 50);
    let cursor = p.opt_str_q(&ctx.query, "cursor", &T::str());
    finish(p)?;
    ok(&ctx.c.chat.list_conversations(limit, cursor.as_deref())?)
}

pub async fn create_conversation(mut ctx: Ctx) -> HResult {
    let body: ConversationIn = body_of(&mut ctx, &T::Model(&CONVERSATION_IN)).await?;
    json(201, &ctx.c.chat.create_conversation(&body)?)
}

pub async fn get_conversation(ctx: Ctx) -> HResult {
    ok(&ctx.c.chat.get_conversation(&ctx.param("cid"), true)?)
}

pub async fn delete_conversation(ctx: Ctx) -> HResult {
    ctx.c.chat.delete_conversation(&ctx.param("cid"))?;
    no_content()
}

pub async fn list_prompts(ctx: Ctx) -> HResult {
    let mut p = Params::default();
    let limit = p.int_q(&ctx.query, "limit", 100);
    let cursor = p.opt_str_q(&ctx.query, "cursor", &T::str());
    finish(p)?;
    ok(&ctx.c.chat.list_prompts(limit, cursor.as_deref())?)
}

pub async fn create_prompt(mut ctx: Ctx) -> HResult {
    let body: PromptIn = body_of(&mut ctx, &T::Model(&PROMPT_IN)).await?;
    json(201, &ctx.c.chat.create_prompt(&body)?)
}

pub async fn update_prompt(mut ctx: Ctx) -> HResult {
    let body: PromptIn = body_of(&mut ctx, &T::Model(&PROMPT_IN)).await?;
    ok(&ctx.c.chat.update_prompt(&ctx.param("pid"), &body)?)
}

pub async fn delete_prompt(ctx: Ctx) -> HResult {
    ctx.c.chat.delete_prompt(&ctx.param("pid"))?;
    no_content()
}

pub async fn create_match(mut ctx: Ctx) -> HResult {
    let body: ArenaMatchIn = body_of(&mut ctx, &T::Model(&ARENA_MATCH_IN)).await?;
    json(201, &ctx.c.chat.create_match(&body).await?)
}

pub async fn vote(mut ctx: Ctx) -> HResult {
    let body: ArenaVote = body_of(&mut ctx, &T::Model(&ARENA_VOTE)).await?;
    ok(&ctx.c.chat.vote(&ctx.param("mid"), &body.winner)?)
}

pub async fn leaderboard(ctx: Ctx) -> HResult {
    ok(&ctx.c.chat.leaderboard()?)
}
