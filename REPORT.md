# API vs. SDK Comparison

## Measurement Context

Both checked-in result files were generated in offline mode from the same six-turn scenario. They are useful for verifying the runners and tool workflows, but they are **not a live performance or cost benchmark**: the API mock estimates tokens from serialized text, while the SDK offline runner makes no model calls and therefore reports zero tokens and cost. Neither run received a cache hit.

## Scenario Metrics

| Metric | Raw Messages API | Claude Agent SDK | Interpretation |
|---|---:|---:|---|
| User turns | 6 | 6 | Same scenario input |
| Model/API calls | 10 | 0 | API mock call count; SDK result is offline and has no model calls |
| Tool calls | 7 | 10 | SDK offline runner also records task/memory reads needed for recommendations and the final summary |
| Input tokens | 9,861 estimated | 0 | SDK live usage is reported by `ResultMessage` |
| Output tokens | 410 estimated | 0 | Not an observed SDK token count |
| Cache reads / hits | 0 | 0 | No live service requests occurred |
| Estimated cost | $0.035733 estimated | $0.00 | API estimate is not a bill; SDK zero means no model request |

Sources: [results/api.json](results/api.json) and [results/sdk.json](results/sdk.json). API input/output tokens and cost use the harness fallback estimator because the offline client has no API usage payload. SDK's offline result deliberately labels usage as estimated and leaves counts at zero.

## Harness Code Size

LOC counts include non-empty lines except standalone comment lines; docstrings and inline comments count. SDK LOC measures project-owned code only and excludes the installed SDK and bundled Claude Code runtime. CLI runners are listed separately and are excluded from the harness total.

| Harness section | API files | API LOC | SDK implementation | SDK LOC |
|---|---|---:|---|---:|
| Execution loop and session handling | `API/harness/loop.py` | 111 | `SDK/agent.py` (client execution, session resume, result parsing) | 148 |
| Tool schemas/dispatch and persistent-memory tools | `API/harness/tools.py` + `API/harness/memory.py` | 186 | `SDK/tools.py` (in-process MCP domain and memory tools) | 163 |
| Append-only conversation history | `API/harness/history.py` | 36 | SDK-managed session transcript and resume | 0 |
| Context construction | `API/harness/context.py` | 51 | SDK-managed context | 0 |
| Context compaction | `API/harness/compaction.py` | 61 | SDK automatic compaction | 0 |
| Prompt caching | `API/harness/caching.py` | 25 | SDK automatic prompt caching | 0 |
| Token and cost tracking | `API/harness/tokens.py` | 84 | SDK result usage/cost fields, parsed in `SDK/agent.py` above | 0 additional |
| **Harness total** |  | **554** |  | **311** |
| CLI scenario runner (not included above) | `API/main.py` | 113 | `SDK/main.py` | 348 |

The SDK LOC column is not a measure of the SDK package itself. Usage/cost parsing is included in `SDK/agent.py` rather than counted twice in the table.

## Trade-Offs

- **Raw Messages API:** The harness exposes request construction, tool dispatch, append-only history, memory persistence, context filtering/compaction, cache breakpoints, and cost estimation as project code. This makes behavior inspectable and individually replaceable, but creates a larger maintenance surface and requires careful handling of tool cycles, transcript validity, caching, and usage accounting.
- **Claude Agent SDK:** The runner delegates session transcripts/resumption, prompt caching, and context compaction to SDK/Claude Code built-ins. The project supplies a custom system prompt and a restricted set of in-process MCP tools, with `strict_mcp_config=True`. This reduces custom harness code, but behavior and some metrics depend on the SDK/runtime version and are less directly controlled by this repository.
- **Persistence:** API history is an application-owned append-only JSONL log, while the SDK stores its transcript under the SDK session and resumes it using the persisted SDK session ID. Both implementations keep user preferences in an application-owned JSON file through their memory tools.
- **Runtime requirements:** The raw implementation runs with the workspace's Python 3.9 interpreter. The current `claude-agent-sdk` package requires Python 3.10 or newer and bundles/uses the Claude Code CLI, so live SDK execution needs a newer interpreter and valid Claude credentials.
- **Benchmarking:** Compare model calls, cache hits, tokens, and cost only after running both implementations live with the same model, credentials/provider, prompt, scenario state, and cache/session starting conditions. The current offline numbers validate plumbing, not relative efficiency.

## Offline Test Coverage

Run `py -3 -m unittest discover -v`. The dedicated mock-client suite in [tests/test_state_history.py](tests/test_state_history.py) has five independent checks: retaining multiple memory keys, writing and retrieving memory across fresh agent instances, persisting user/assistant history in append-only form, restoring messages in order, and isolating histories by session. The broader suite also checks tool cycles, SDK option restrictions, scenario execution, and persistent state.
