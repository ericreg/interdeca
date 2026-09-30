# Local application protocol, version 1

The shared `ApplicationSession` in `application.py` owns `ApplicationController`,
`ApplicationState`, pipeline construction, and `SerialExecutionManager`. Slicer
uses this session through the WebSocket worker; the [Click CLI](cli.md) hosts the
same session in process, passing validated command envelopes through a local queue.
Slicer owns widgets, MRML access, world-coordinate
scene snapshots, result rendering, and the worker process lifetime. A standalone
Qt host can use `ApplicationClient` from `interdeca_client.client`
with its own asynchronous UI dispatcher and view. The client package imports
NumPy and `websockets`, but never imports `interdeca`, Pydantic, SciPy, scikit-learn,
VTK, or an estimator implementation. `client/pyproject.toml` builds the separate
`interdeca-client` distribution; Slicer does not install the computational package.

## Startup and connection

The host starts its configured external Python with
`-I -u -m interdeca.worker --session-directory <private-directory>`. The worker
must be installed in that environment; it does not inherit Slicer's Python paths
or shared-library paths. The host captures stdout and stderr from process creation,
including import failures that happen before the server exists.

The worker binds an ephemeral port on `127.0.0.1` and atomically publishes
`endpoint.json` containing `protocol_version` and `port`. The host connects once,
with `Authorization: Bearer <session-secret>`; the launcher supplies the secret
through `INTERDECA_SESSION_TOKEN`, not a command-line argument or endpoint file.
Browser Origins and additional clients are rejected. Proxy routing and WebSocket
compression are disabled. Messages are JSON text, with a 1 MiB message limit.
The implementation uses the [websockets 15 threading API](https://websockets.readthedocs.io/en/15.0/reference/sync/client.html).

The worker reports `capabilities`, including PCA, ICA, UMAP, and the local
diagnostic log path. The host waits for this message before enabling analysis.
UMAP is a required core dependency imported only in the computational environment,
so its binary dependencies do not affect Slicer. A missing or broken core dependency
is a startup failure captured by the launcher.

There is one session and one connection per client. Disconnect closes the session;
there is no reconnect/replay or persisted estimator restoration in version 1.
Restart creates fresh identifiers and a fresh private directory.

## Commands

Pydantic models in `interdeca/serde.py` validate the envelope and then the schema
selected by its action. Unknown actions, additional payload fields, unsupported
versions, stale revisions, and invalid numerical metadata fail explicitly.
Commands describe user actions, never tasks, class names, or pickled objects.

```json
{
  "protocol_version": 1,
  "type": "command",
  "request_id": "request-42",
  "input_revision": 3,
  "action": "fit_population",
  "payload": {
    "dataset_id": "worker-issued-dataset-id",
    "options": {"method": "umap", "components": 2}
  }
}
```

| Action | Payload |
| --- | --- |
| `run_atlas` | `AtlasRequest` fields: directories, optional override/Blender paths, and bake/UV/cleanup options |
| `process_colors` | `mesh` references and `options` (`ColorRequest`) |
| `fit_population` | `dataset_id` and `options` (`ModelRequest`) |
| `show_specimen` | `dataset_id`, `specimen_id` |
| `select_region` | `mesh`, `points`, `same_side_only`, optional `vertex_colors` |
| `export_region` | `mesh` |
| `set_axes` | `model_id`, zero-based `x_axis`, `y_axis` |
| `reconstruct` | `model_id`, `specimen_id`, and `positions`: two slider fractions in [0, 1], or null to start at the specimen |
| `invalidate` | `scope`: colors, population, selection, mesh, curve, or all |
| `clear_selection` / `cancel` | Empty object |

The worker retains one current processed dataset and one fitted population model.
Identifiers are opaque and valid only while those results remain current. A new
fit or invalidation makes older model identifiers unusable. Requests have unique
IDs within their session; duplicates are rejected, not retried.

## Events and revisions

All events contain `protocol_version`, `request_id`, `run_id`, and
`input_revision`. Session-level events use null identifiers. An accepted pipeline
command receives its run ID before execution; immediate actions use a null run ID.
A command rejected before acceptance receives `failed` with its request ID when
that identity can be validated. Malformed identity fields produce an uncorrelated
failure.

| Type | Meaning |
| --- | --- |
| `capabilities` | Estimator availability and diagnostic log location |
| `accepted` | Validated action accepted, with its allocated run ID |
| `progress` | Operation label, completed/total tasks, current/total stages |
| `log` | Batched severity/message entries; each entry retains request/run identity |
| `presentation` | Small control snapshot, current dataset/model IDs, selection counts |
| `result` | Atlas display data, scores, plot data, morphospace controls, face colors, selection indices, or export geometry |
| `completed` / `failed` / `cancelled` | Terminal acknowledgement; failures include diagnostic details |

The client increments its input revision **before** sending `invalidate` or
`clear_selection`. These actions must carry a strictly newer revision. Other
commands use the current revision. Results from older revisions are discarded
before touching the scene. Control snapshots carry their own revision because an
older run can finish after an invalidation and still release the current busy
indicator. The host also rechecks the source mesh signature immediately before
rendering, including vertex coordinates, face order, and corner UVs.

The worker computes slider ranges, full model coordinates, and the plot marker.
Its `morph_controls` result supplies normalized slider positions and the two
selected coordinates for direct display. Population control metadata includes a
component count; numerical scores are sent only for plotting. The Slicer form
retains no base coordinate vector or fitted ranges.

Rapid reconstruction commands keep only the newest pending request in the worker.
Each superseded request receives `cancelled`; the retained request keeps its
original request ID and reserved run ID when it starts. The client also suppresses
older reconstruction results already queued for UI delivery. Clicking Cancel
suppresses queued local results as well as requesting worker cancellation.

## Numerical artifacts and ownership

Each launch creates a private `interdeca-session-*` directory. The client publishes
inputs under `inputs/`; the worker publishes outputs under `outputs/`. Each file
has a random name and is written to a temporary file before an atomic rename.
Only the completed filename may be sent to the peer:

```json
{
  "path": "inputs/0123456789abcdef0123456789abcdef.npy",
  "dtype": "<f8",
  "shape": [1000, 3]
}
```

References are relative to that session root. Readers check ownership, path
containment, dtype, and shape, reject object arrays, and use `allow_pickle=False`.
Mesh payloads contain separate `vertices`, `faces`, and nullable `uv` references.
UV shape is `(face_count, 3, 2)`: every corner retains its original place. The
worker validates numerical geometry before it reaches an algorithm. Neither side
sorts or triangulates meshes during transport.

Every artifact has one consumer. The worker copies input arrays before use; the
client deletes input files after the request's terminal acknowledgement. The
client copies result arrays into presentation records and deletes those output
files, including results discarded as stale. Plot scores remain display data;
slider interpolation, fitted estimators, and analytical features remain in the
worker. Control snapshots contain IDs and counts, so they never reuse a
consumed artifact reference.

The Slicer client removes both artifact directories when closing the session.
The CLI retains output files as saved presentation results for testing, while
removing its temporary inputs. Diagnostic `worker.log` is retained for both hosts;
the socket launcher also retains subprocess `stdout.log` and `stderr.log`.
A host crash may leave its temporary directory for manual deletion. Persistent
atlas outputs and texture paths are outside the exchange directories and are
never removed by this cleanup.

## Scheduling, diagnostics, and shutdown

A session thread serializes controller calls and execution callbacks. A separate
serial thread performs numerical operations. WebSocket reading and writing run
independently, so native numerical calls do not prevent receipt of cancellation.
Progress and control snapshots are coalesced at approximately 20 Hz. Log batches
are bounded; dropped UI entries and abbreviated long messages remain complete
in `worker.log`. Result and terminal events are retained in order.

Cancellation takes effect between operations; it does not interrupt a running
native call or Blender operation. Stop Worker closes the connection and detaches
UI callbacks immediately, then waits briefly and terminates the process if needed.
On POSIX, the launcher also terminates the worker's process group, including active
Blender children. Windows process-tree teardown still needs platform validation.
The application does not delete partial persistent atlas outputs.

Slicer's timer delivers all view callbacks on the UI thread before widgets or MRML
are accessed. A brief timer yield lets the client socket thread run under PythonQt;
no numerical analysis runs in that host thread or process.
