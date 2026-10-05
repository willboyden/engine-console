# ADR-0018: Downloads and copies stream through the hub

Status: accepted (contract v2, `docs/ARCHITECTURE.md` section 10.1, decision 18, and section 10.6); implemented in
`backend/src/engine_console/services/downloads.py` and the `fs.*` operations of the node. Extends ADR-0008 and ADR-0012.
**Verified live:** one real Hugging Face download through the allowlisting egress proxy into a writable root (the Hugging
Face cache layout), and a library delete of a model. **Fakes only:** copies between machines (the tests run two real
`ecnode --stdio` processes on one machine; the opt-in test is `backend/tests/data/test_live_node.py` with
`ENGINE_CONSOLE_LIVE_NODE=1`), resume after an interrupted transfer against a real network, and any transfer to a
remote machine.

## Context

Models must be able to land in any machine's folders and on any of its drives. Nodes need no internet access, and the
project's rule is deny-by-default egress: the hub is already the one place with the Hugging Face allowlist, the optional
egress proxy and the token. A file must be checked where it ends up, not where it started.

## Decision

- **Download.** The hub lists the repository's files (with a sha256 for each large file) through its own allowlisted client,
  asks the target node which blobs and partial files exist (`fs.stat`), and streams each missing file from Hugging Face into
  the node's `fs.write`, resuming from the partial's size. The node always checks the declared size and, when the hub knows
  the expected sha256 (a Hugging Face file stored with LFS), hashes the **whole** file and compares; only then does it rename
  `<file>.incomplete` into place, and on a size or hash mismatch it deletes the partial. It never fills a disk below 1 GiB
  free. The Hugging Face token stays on the hub.
- **Copy.** `fs.read` on the source node, through the hub, into `fs.write` on the target node, with the same resume and
  verification.
- **Layout by the target root's kind.** `hf-home` and `hf-hub` roots receive the cache layout (`blobs/<sha256>`,
  `snapshots/<commit>/<path>` as relative links, `refs/<revision>`), written with `fs.write`, `fs.symlink` and
  `fs.put_small`, with the links and the ref created only after every blob is in place, so a half-downloaded repository
  never looks like a model. A `gguf` root receives the files under `<org>--<name>/<path>`. An `auto` root takes what its
  last scan showed. Roots of kind `plain` take only a copy of a plain model directory; `ollama` and `other` roots take
  nothing.
- **What can be copied.** A Hugging Face cache folder, a loose `.gguf` file or split set, and a plain model directory whose
  scan counted every file. The node protocol has no directory listing, so a plain directory is found by its usual file
  names and then proven complete (file count and byte total must equal the scan); an Ollama store or an `other` folder is
  refused (`copy_unsupported`).
- **Copies that start in a Hugging Face cache still need Hugging Face.** The hub asks Hugging Face for the repository's file
  listing (paths, sizes, sha256) to know what the cache holds. A cache names small, non-LFS files after git object ids, so
  in a copy of a whole repository the large files go machine to machine and the few small ones (config, tokenizer) are
  fetched from Hugging Face by the hub. A loose `.gguf` file or a plain model directory is copied without any Hugging Face
  access.
- **Bookkeeping.** `POST /downloads` takes `host_id` and `root_id` (default: the machine's first writable Hugging Face
  root); `POST /library/copies/{id}/copy` creates a row of kind `copy`. Two downloads run at once, at most 50 rows are
  unfinished, the hub checks free space on the target mount with a 2 % margin before it starts, and a hub restart parks
  running and queued rows as `paused`. Pause keeps the partial; cancel asks the node to delete the file in flight only if
  that node's policy allows deletes.

## Consequences

- One egress chokepoint for model files, one place for the token, nodes without internet, integrity checked on the
  storing machine, and resume that survives a hub restart because the partial lives on the node.
- Integrity has a gap: only files with an expected sha256 are hash-checked. A loose `.gguf` file or a plain model directory
  copied between machines, and the small non-LFS files of a repository, are checked for size only.
- The hub is on the data path. Every byte of a machine-to-machine copy crosses the hub twice (read from the source node,
  written to the target node), so throughput is bounded by the hub's CPU and by two ssh sessions. This has not been
  measured. NFS shared storage (see below) avoids a copy where it is available.
- With an egress proxy configured, the proxy carries every downloaded byte and sees the token in each request header.
- A copy that starts in a Hugging Face cache fails when Hugging Face is unreachable from the hub, even though the large
  files are local to the fleet.
- A cancelled or failed transfer leaves its `.incomplete` partial on a machine whose policy forbids deletes; downloading
  the same file again resumes from it.
- Shared storage awareness (contract section 10.5): when a machine's model root is an NFS mount of another configured
  machine's disk, the library lists the bytes once and says both machines can load them, and a copy onto the same
  storage is refused (`copy_same_location`).

## Alternatives considered

- **Nodes download directly from Hugging Face.** Every node would need internet, an egress proxy setting, and a copy of the
  token.
- **Node-to-node transfer (rsync or scp).** Needs trust and keys between nodes, bypasses the hub's integrity check and the
  receiving node's policy checks.
- **NFS or another shared filesystem only.** It helps where it exists (and is recognised), but not for machines that do not
  share storage or for drives that are local.
- **The hub keeps a staging copy of each model.** Needs hub disk as large as the models and writes every byte twice.
