# Free Model Relay

[![Make.fun bounty](https://make.fun/badge/free-model-aggregator-02.svg)](https://make.fun/bounty/free-model-aggregator-02)
[![Rank](https://make.fun/badge/free-model-aggregator-02/bujangkasep100-hash/makefun-live-model-aggregator/rank.svg)](https://make.fun/bounty/free-model-aggregator-02/leaderboard)
[![Reward](https://make.fun/badge/free-model-aggregator-02/bujangkasep100-hash/makefun-live-model-aggregator/reward.svg)](https://make.fun/bounty/free-model-aggregator-02)

One Node.js 22 server exposes OpenAI Chat Completions and Anthropic Messages over Make.fun's OpenRouter gateway. Both APIs support JSON replies and server-sent events. The two model-list formats and the trial registration share the free catalog from the trial brief.

Send `model: "auto"` to rank eligible models by live availability and the brief's quality ratings. Use a model ID from `/v1/models` to pin a request. The relay forwards the client's output cap, system instructions, tools and JSON format. A retryable refusal can move an `auto` request to another model. The response reports the actual model and any earlier gateway refusals.

## Run it

From the unpacked Make.fun kit, run the practice grade with Node.js 22. Set `ENTRY_REPO` to this repository's absolute path. `OPENROUTER_ENV_FILE` names a private env file outside the repository containing `OPENROUTER_API_KEY`.

```sh
node cli/makefun.mjs grade-local --kit . --bot "$ENTRY_REPO" \
  --driver process --env-file "$OPENROUTER_ENV_FILE"
```

The kit supplies the trial brief, the environment service and the gateway. To print the catalog without starting the server, run `node main.ts --list` in this repository with `MAKEFUN_PROMPT_FILE` set to the brief.
