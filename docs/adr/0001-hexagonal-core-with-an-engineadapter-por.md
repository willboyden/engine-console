# ADR-0001: Hexagonal core with an EngineAdapter port

Status: accepted (contract v1, see `docs/ARCHITECTURE.md` section 1)

## Context

The console must manage vLLM and SGLang now and plausibly others (llama.cpp, TensorRT-LLM, Dynamo) later. Engines differ in CLI flags, metrics names, log formats and memory behaviour, but the control-plane logic (fit, lifecycle, downloads, profiles) is identical.

## Decision

Define one abstract port, `EngineAdapter` (`adapters/base.py`, locked): `param_catalog`, `presets`, `validate`, `build_launch`, `compatibility`, `memory_model`, `parse_metrics`, `parse_startup_log`. Services depend only on the port; `AdapterRegistry` discovers `vllm` and `sglang` modules by inspection and tolerates a broken module. Other ports (`HubClient`, `CommandRunner`, `GpuProbe` in `domain/ports.py`) are injected through `container.py`, which lets tests substitute fakes.

## Consequences

- Adding an engine is one module plus registry entry; the UI form is generated from `ParamSpec[]`, so no frontend change.
- Offline tests are straightforward (fakes at every port).
- Cost: the port is a lowest common denominator. Engine-specific features (e.g. a vLLM-only knob) must be expressed as a `ParamSpec`, and cross-engine concepts (`memory_model` returning a flat float dict) are loosely typed.
- The port is locked, so changing it needs a contract revision, not a drive-by edit.

## Alternatives considered

- Per-engine code paths inside services (`if engine == 'vllm'`): rejected, it spreads across every service.
- A generic 'run any container' abstraction with no engine knowledge: rejected, it forfeits fit estimation and parameter validation, the main value.
