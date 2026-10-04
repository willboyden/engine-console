"""Lifecycle scripts registered in a node's policy (contract v2, management class `script`).

A lab often already starts its engines with its own script that encodes ordering rules (start A before B); the
console must use that script and never bypass it. The hub sees only a script's name, label and the NAMES of its
actions (argv stays on the node); running one is `scripts.run` on that host's node, whose own policy decides.

Rules enforced here before anything is sent: the script and action exist in the node-reported policy, the one
argument (if any) is from the action's allowlist, the node is connected and in manage mode, and only one job per
(host, script) runs at a time. Output lines are masked with `common.scrub` before they are stored or streamed."""
from __future__ import annotations

import asyncio
import contextlib
import logging
import re
from collections import deque
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Any

from engine_console.domain.errors import BadRequest, Conflict, NotFound
from engine_console.domain.fleet import FleetPort, NodeError, ScriptAction, ScriptInfo, ScriptJob
from engine_console.services import nodeview as nv
from engine_console.services.common import new_id, now, scrub
from engine_console.services.nodeerr import NodeCaller, shielded

log = logging.getLogger(__name__)
TAIL_CHARS = 20_000      # ScriptJob.output_tail: the last ~20 kB
MAX_LINES = 2000         # lines kept for late SSE subscribers
MAX_JOBS = 50            # finished jobs kept in memory
LINE_CAP = 4000
JOB_MAX_S = 4 * 3600      # a job that outlives this is failed (the node's own per-script timeout normally ends it long before)
_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f\u2028\u2029]")   # everything but tab: a line is one line


@dataclass
class _Job:
    info: ScriptJob
    steps: list[tuple[str, str | None]]
    lines: deque[str] = field(default_factory=lambda: deque(maxlen=MAX_LINES))
    base: int = 0                                     # lines dropped from the front of `lines`
    step: str = ""                                    # the action running right now
    changed: asyncio.Condition = field(default_factory=asyncio.Condition)
    task: asyncio.Task[None] | None = None

    @property
    def finished(self) -> bool:
        return self.info.state != "running"

    def tail(self) -> str:
        return "\n".join(self.lines)[-TAIL_CHARS:]


class ScriptService:
    def __init__(self, fleet: FleetPort, *, clock: Callable[[], float] = now) -> None:
        self._fleet = fleet
        self._nodes = NodeCaller(fleet)
        self._clock = clock
        self._jobs: dict[str, _Job] = {}
        self._running: dict[tuple[str, str], str] = {}   # (host, script) -> job id

    # -- what the policies register ----------------------------------------------------------------------------
    def scripts(self, host_id: str | None = None) -> list[ScriptInfo]:
        out: list[ScriptInfo] = []
        for h in self._fleet.hosts():
            if host_id is not None and h.id != host_id:
                continue
            node = self._fleet.node(h.id)
            connected = nv.link_up(node)
            status = {str(s.get("name")): s for s in nv.inventory(node).get("scripts") or [] if isinstance(s, dict)}
            for s in nv.scripts(node):
                st = status.get(str(s["name"])) if connected else None
                code = st.get("status") if st else None
                out.append(ScriptInfo(
                    host_id=h.id, name=str(s["name"]), label=str(s.get("label") or s["name"]),
                    engine=s.get("engine") or None, description=str(s.get("description") or ""),
                    status=code if code in ("up", "down", "unknown", "error") else "unknown",
                    detail=(st.get("detail") if st else None) or (None if connected else "this machine is not connected"),
                    checked_at=nv.num(st.get("checked_at")) if st else None,
                    actions=[ScriptAction(name=str(a["name"]), label=str(a.get("label") or a["name"]),
                                          args=[str(x) for x in a.get("args") or []], confirm=bool(a.get("confirm")),
                                          danger=bool(a.get("danger")))
                             for a in s.get("actions") or [] if isinstance(a, dict) and a.get("name")],
                    containers=[str(c) for c in s.get("containers") or []]))
        return out

    def script(self, host_id: str, name: str) -> ScriptInfo:
        for s in self.scripts(host_id):
            if s.name == name:
                return s
        raise NotFound(f"no script '{name}' is registered in the policy of host {host_id}", code="unknown_script")

    def busy(self, host_id: str) -> dict[str, str]:
        """Script name -> the action running right now, for instances to show 'starting' / 'stopping'."""
        return {n: self._jobs[j].step for (h, n), j in self._running.items() if h == host_id and j in self._jobs}

    # -- running -------------------------------------------------------------------------------------------------
    def _validate(self, host_id: str, name: str, action: str, arg: str | None) -> None:
        node = self._nodes.node(host_id)
        info = self.script(host_id, name)
        act = next((a for a in info.actions if a.name == action), None)
        if act is None:
            raise NotFound(f"script '{name}' has no action '{action}' (it offers: "
                           f"{', '.join(a.name for a in info.actions) or 'none'})", code="unknown_action")
        if act.args and arg is None:
            raise BadRequest(f"action '{action}' needs one of: {', '.join(act.args)}", code="arg_required")
        if arg is not None and arg not in act.args:
            raise BadRequest(f"'{arg}' is not an allowed argument for '{action}'"
                             + (f" (allowed: {', '.join(act.args)})" if act.args else " (it takes none)"), code="invalid_arg")
        if not nv.link_up(node):
            raise self._nodes.problem(NodeError(host_id, "node_unavailable", node.link().error or
                                                f"the node is {node.link().state}"))
        if nv.mode(node) != "manage":
            raise self._nodes.problem(NodeError(host_id, "monitor_only", "this machine's node policy is in monitor "
                                                "mode: scripts cannot run until its operator switches the mode to manage"))

    def _start(self, host_id: str, name: str, label: str, arg: str | None, steps: list[tuple[str, str | None]]) -> ScriptJob:
        key = (host_id, name)
        if key in self._running:
            raise Conflict(f"script '{name}' on host {host_id} is already running (job {self._running[key]}); "
                           "wait for it to finish", code="script_busy")
        job = _Job(ScriptJob(id=new_id("job_"), host_id=host_id, name=name, action=label, arg=arg, state="running",
                             started_at=self._clock()), steps)
        job.step = steps[0][0]
        self._jobs[job.info.id] = job
        self._running[key] = job.info.id
        job.task = asyncio.create_task(self._execute(job), name=f"script-{job.info.id}")
        self._evict()
        return job.info.model_copy()

    async def run(self, host_id: str, name: str, action: str, arg: str | None = None) -> ScriptJob:
        self._validate(host_id, name, action, arg)
        return self._start(host_id, name, action, arg, [(action, arg)])

    async def restart(self, host_id: str, name: str) -> ScriptJob:
        """`down` then `up` as ONE job (one id to follow). Stops at the first failure: never starts what failed to stop."""
        self._validate(host_id, name, "down", None)
        self._validate(host_id, name, "up", None)
        return self._start(host_id, name, "restart", None, [("down", None), ("up", None)])

    async def _emit(self, job: _Job, line: str) -> None:
        text = _CONTROL.sub(" ", scrub(line))[:LINE_CAP]
        async with job.changed:
            if len(job.lines) == job.lines.maxlen:
                job.base += 1
            job.lines.append(text)
            job.changed.notify_all()

    async def _execute(self, job: _Job) -> None:
        info = job.info
        rc: int | None = 0
        try:
            for action, arg in job.steps:
                job.step = action
                if len(job.steps) > 1:
                    await self._emit(job, f"== {action} ==")
                rc = await self._one(job, action, arg)
                if rc != 0:
                    break
            info.rc = rc
            info.state = "completed" if rc == 0 else "failed"
        except NodeError as e:
            await self._emit(job, f"[{e.code}] {e.message}")
            info.state = "failed"
        except asyncio.CancelledError:
            info.state = "failed"
            await shielded(self._emit(job, "[cancelled] the console is shutting down"))
            raise
        except Exception as e:   # noqa: BLE001 - a job must always end in a terminal state
            log.exception("script job %s crashed", info.id)
            await self._emit(job, f"[internal] {type(e).__name__}")
            info.state = "failed"
        finally:
            info.finished_at = self._clock()
            info.output_tail = job.tail()
            self._running.pop((info.host_id, info.name), None)
            async with job.changed:
                job.changed.notify_all()

    async def _one(self, job: _Job, action: str, arg: str | None) -> int | None:
        args: dict[str, Any] = {"name": job.info.name, "action": action}
        if arg is not None:
            args["arg"] = arg
        stream = self._nodes.open(job.info.host_id, "scripts.run", args)
        try:
            async with asyncio.timeout(JOB_MAX_S):
                async for item in stream:
                    ev = item.event if item.kind == "event" else None
                    if ev and isinstance(ev.get("line"), str):
                        await self._emit(job, ev["line"])
                        job.info.output_tail = job.tail()
                res = await stream.result()
        except TimeoutError:
            await shielded(stream.cancel())
            raise NodeError(job.info.host_id, "timeout", f"the script ran longer than {JOB_MAX_S // 3600} hours") from None
        except BaseException:
            await shielded(stream.cancel())
            raise
        rc = res.get("rc")
        return rc if isinstance(rc, int) else None

    # -- reading -------------------------------------------------------------------------------------------------
    def job(self, job_id: str) -> ScriptJob:
        j = self._jobs.get(job_id)
        if j is None:
            raise NotFound(f"no such script job: {job_id}", code="unknown_job")
        j.info.output_tail = j.tail()
        return j.info.model_copy()

    def jobs(self) -> list[ScriptJob]:
        return [self.job(j) for j in reversed(self._jobs)]

    async def stream(self, job_id: str, *, once: bool = False) -> AsyncIterator[tuple[str, str]]:
        """(event, data) pairs for SSE: every `line` (the whole history, then live), then one `end` with the job."""
        j = self._jobs.get(job_id)
        if j is None:
            raise NotFound(f"no such script job: {job_id}", code="unknown_job")
        pos = 0   # absolute line number
        while True:
            async with j.changed:
                pos = max(pos, j.base)
                pending = list(j.lines)[pos - j.base:]
                finished = j.finished
                if not pending and not finished and not once:
                    await j.changed.wait()
                    continue
            for line in pending:
                yield "line", line
            pos += len(pending)
            if finished or once:
                break
        yield "end", self.job(job_id).model_dump_json()

    async def wait(self, job_id: str) -> None:
        j = self._jobs.get(job_id)
        if j is not None and j.task is not None:
            await asyncio.gather(j.task, return_exceptions=True)

    def _evict(self) -> None:
        done = [i for i, j in self._jobs.items() if j.finished]
        for i in done[: max(0, len(done) - MAX_JOBS)]:
            del self._jobs[i]

    async def shutdown(self) -> None:
        tasks = [j.task for j in self._jobs.values() if j.task is not None and not j.task.done()]
        for t in tasks:
            t.cancel()
        with contextlib.suppress(Exception):
            await asyncio.gather(*tasks, return_exceptions=True)
