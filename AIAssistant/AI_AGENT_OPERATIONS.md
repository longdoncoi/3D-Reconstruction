# AI Agent operations

## Runtime modes

The desktop default is local `llama_cpp`, in-memory checkpoints and an enabled
allow-list sandbox.  It requires no external services.  Set
`AGENT_CHECKPOINT_BACKEND=postgres` or `redis` together with
`AGENT_CHECKPOINT_URL` only after installing the matching optional LangGraph
package and provisioning the service.

`/v1/agent/execute` accepts normal JSON for the Qt client or SSE when the
request has `Accept: text/event-stream`. SSE events are `status`, `step` and
`done`. Prometheus is exposed at `/metrics` only with
`AGENT_OBSERVABILITY=1`; traces use the OpenTelemetry SDK exporter configured
by the deployment.

## Desktop actions

`Config/agent_action_manifest.json` is the only action/alias/intent contract.
The server validates it before dispatch; Qt validates it again before emitting
the action. Each action receives a `request_id`, and Qt must post the result to
`/v1/agent/ui-action-result`. A dispatch is not reported as successful until
that acknowledgement arrives.

## Scale and privacy

Do not move inference to vLLM/TGI merely for model quality. Use it when shared
GPU batching/concurrent users are a measured bottleneck. Configure an
OpenAI-compatible endpoint with `AGENT_INFERENCE_*` after that decision.

The hybrid policy defaults to `local_only`. A cloud route may only be enabled
for `public`/`non_sensitive` requests; private project files, DICOM and user
data must stay local unless product policy explicitly changes.

## Tool sandbox

Approved `run_command` and `write_file` calls are deny-by-default. Commands
must use an allow-listed executable and cannot use shell chaining; writes are
atomic, size-limited and restricted to `AGENT_WRITE_ALLOWLIST`. For stronger
isolation set `AGENT_SANDBOX_RUNTIME=docker`: commands run with no network,
one CPU, 1 GiB memory and 128 process IDs. Use an organization-approved image
that contains only the allowed build/test tools before enabling this mode.
