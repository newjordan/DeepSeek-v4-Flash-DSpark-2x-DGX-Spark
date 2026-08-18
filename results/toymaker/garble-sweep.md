# Context Garble Sweep — 2026-08-18 16:56

model: deepseek-v4-flash-dspark | endpoint: http://127.0.0.1:18888/v1 | runs/length: 2 | cold prefill: forced (unique nonce)

| ctx_len | run | verdict | finish | secs | reasoning_ch | tool_calls | flags | content |
|---|---|---|---|---|---|---|---|---|
| 2048 | 0 | CLEAN | stop | 7.6 | 77 | - | - | 'The capital of Idaho is Boise. Its metro area population is roughly 80' |
| 2048 | 1 | CLEAN | stop | 6.3 | 77 | - | - | 'The capital of Idaho is **Boise**. Its metro area population is roughl' |
| 8192 | 0 | CLEAN | stop | 10.3 | 77 | - | - | 'The capital of Idaho is Boise. Its rough population is about 235,000 (' |
| 8192 | 1 | CLEAN | stop | 1.3 | 77 | - | - | 'The capital of Idaho is Boise. Its population is roughly 236,000 (city' |
| 32768 | 0 | CLEAN | stop | 16.7 | 77 | - | - | 'The capital of Idaho is Boise. Its population is roughly 236,000 (city' |
| 32768 | 1 | CLEAN | stop | 2.0 | 187 | - | - | 'The capital of Idaho is **Boise**. Its population is roughly **235,000' |
| 131072 | 0 | CLEAN | stop | 71.4 | 77 | - | - | 'The capital of Idaho is Boise. Its population is roughly 236,000 (city' |
| 131072 | 1 | CLEAN | tool_calls | 3.9 | 263 | search_web | - | '' |
