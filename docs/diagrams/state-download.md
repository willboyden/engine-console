# State machine: download

States from `DownloadState` in `domain/models.py`; transitions from `services/downloads.py`.

```mermaid
stateDiagram-v2
  [*] --> queued: POST /downloads
  queued --> running: slot free (concurrency 2)
  running --> paused: pause
  queued --> paused: pause
  paused --> queued: resume
  failed --> queued: resume
  running --> completed: all files done, sha256 ok, refs written
  running --> failed: error / disk / gated / hash mismatch
  running --> cancelled: cancel
  queued --> cancelled: cancel
  paused --> cancelled: cancel
  completed --> [*]
  cancelled --> [*]
```

On console restart `recover()` parks rows left `running` or `queued` as `paused` so they can be resumed.
