"""How long does the /metrics handler's loop (`async for stat in llm.get_stats_async(2)`) take with the engine's own
IterationResult class? Component-level timing: no server, no model."""
import asyncio
import json
import time

from tensorrt_llm.executor.result import IterationResult


async def read_all(result):
    t = time.monotonic()
    got = [x async for x in result]
    return got, round(time.monotonic() - t, 2)


async def main():
    result = IterationResult()
    result.set_timeout(2)
    for i in range(3):
        result.queue.put(json.dumps({"iter": i}))
    print("first read (3 items queued):", await read_all(result), flush=True)
    print("second read, no new prompt:", await read_all(result), flush=True)
    result.mark_undone()                       # what aget_stats does on every call
    print("after mark_undone, queue empty:", await read_all(result), flush=True)


asyncio.run(main())
