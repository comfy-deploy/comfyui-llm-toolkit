// Safe API Provider Selector Node - Model list updates without auto-fetch
import { app } from "../../../scripts/app.js";

// Hardcoded model lists for immediate updates (no fetch on load)
const PROVIDER_MODELS = {
    "openai": [
        "gpt-4o-mini",
        "babbage-002",
        "chatgpt-4o-latest",
        "chatgpt-5-latest",
        "codex-mini-latest",
        "computer-use-preview",
        "computer-use-preview-2025-03-11",
        "dall-e-2",
        "dall-e-3",
        "davinci-002",
        "gpt-3.5-turbo",
        "gpt-3.5-turbo-0125",
        "gpt-3.5-turbo-1106",
        "gpt-3.5-turbo-16k",
        "gpt-3.5-turbo-instruct",
        "gpt-3.5-turbo-instruct-0914",
        "gpt-4",
        "gpt-4-0125-preview",
        "gpt-4-0613",
        "gpt-4-1106-preview",
        "gpt-4-turbo",
        "gpt-4-turbo-2024-04-09",
        "gpt-4-turbo-preview",
        "gpt-4.1",
        "gpt-4.1-2025-04-14",
        "gpt-4.1-mini",
        "gpt-4.1-mini-2025-04-14",
        "gpt-4.1-nano",
        "gpt-4.1-nano-2025-04-14",
        "gpt-4.5-preview",
        "gpt-4.5-preview-2025-02-27",
        "gpt-4o",
        "gpt-4o-2024-05-13",
        "gpt-4o-2024-08-06",
        "gpt-4o-2024-11-20",
        "gpt-4o-audio-preview",
        "gpt-4o-audio-preview-2024-10-01",
        "gpt-4o-audio-preview-2024-12-17",
        "gpt-4o-audio-preview-2025-06-03",
        "gpt-4o-mini-2024-07-18",
        "gpt-4o-mini-audio-preview",
        "gpt-4o-mini-audio-preview-2024-12-17",
        "gpt-4o-mini-realtime-preview",
        "gpt-4o-mini-realtime-preview-2024-12-17",
        "gpt-4o-mini-search-preview",
        "gpt-4o-mini-search-preview-2025-03-11",
        "gpt-4o-mini-transcribe",
        "gpt-4o-mini-tts",
        "gpt-4o-realtime-preview",
        "gpt-4o-realtime-preview-2024-10-01",
        "gpt-4o-realtime-preview-2024-12-17",
        "gpt-4o-realtime-preview-2025-06-03",
        "gpt-4o-search-preview",
        "gpt-4o-search-preview-2025-03-11",
        "gpt-4o-transcribe",
        "gpt-5",
        "gpt-5-2025-08-07",
        "gpt-5-chat-latest",
        "gpt-5-mini",
        "gpt-5-mini-2025-08-07",
        "gpt-5-nano",
        "gpt-5-nano-2025-08-07",
        "gpt-audio",
        "gpt-audio-2025-08-28",
        "gpt-image-1",
        "gpt-realtime",
        "gpt-realtime-2025-08-28",
        "gpt40-0806-loco-vm",
        "o1",
        "o1-2024-12-17",
        "o1-mini",
        "o1-mini-2024-09-12",
        "o1-preview",
        "o1-preview-2024-09-12",
        "o1-pro",
        "o1-pro-2025-03-19",
        "o3",
        "o3-2025-04-16",
        "o3-deep-research",
        "o3-deep-research-2025-06-26",
        "o3-mini",
        "o3-mini-2025-01-31",
        "o3-pro",
        "o3-pro-2025-06-10",
        "o4-mini",
        "o4-mini-2025-04-16",
        "o4-mini-deep-research",
        "o4-mini-deep-research-2025-06-26",
        "omni-moderation-2024-09-26",
        "omni-moderation-latest",
        "text-embedding-3-large",
        "text-embedding-3-small",
        "text-embedding-ada-002",
        "tts-1",
        "tts-1-1106",
        "tts-1-hd",
        "tts-1-hd-1106",
        "tts-l-hd",
        "whisper-1",
        "whisper-I"
    ],
    "gemini": [
        "gemini-2.5-flash-image-preview",
        "aqa",
        "embedding-001",
        "embedding-gecko-001",
        "gemini-1.5-flash",
        "gemini-1.5-flash-002",
        "gemini-1.5-flash-8b",
        "gemini-1.5-flash-8b-001",
        "gemini-1.5-flash-8b-latest",
        "gemini-1.5-flash-latest",
        "gemini-1.5-pro",
        "gemini-1.5-pro-002",
        "gemini-1.5-pro-latest",
        "gemini-2.0-flash",
        "gemini-2.0-flash-001",
        "gemini-2.0-flash-exp",
        "gemini-2.0-flash-lite",
        "gemini-2.0-flash-lite-001",
        "gemini-2.0-flash-lite-preview",
        "gemini-2.0-flash-lite-preview-02-05",
        "gemini-2.0-flash-thinking-exp",
        "gemini-2.0-flash-thinking-exp-01-21",
        "gemini-2.0-flash-thinking-exp-1219",
        "gemini-2.0-pro-exp",
        "gemini-2.0-pro-exp-02-05",
        "gemini-2.5-flash",
        "gemini-2.5-flash-image-preview",
        "gemini-2.5-flash-lite",
        "gemini-2.5-flash-lite-preview-06-17",
        "gemini-2.5-flash-preview-05-20",
        "gemini-2.5-flash-preview-tts",
        "gemini-2.5-pro",
        "gemini-2.5-pro-preview-03-25",
        "gemini-2.5-pro-preview-05-06",
        "gemini-2.5-pro-preview-06-05",
        "gemini-2.5-pro-preview-tts",
        "gemini-embedding-001",
        "gemini-embedding-exp",
        "gemini-embedding-exp-03-07",
        "gemini-exp-1206",
        "gemma-3-12b-it",
        "gemma-3-1b-it",
        "gemma-3-27b-it",
        "gemma-3-4b-it",
        "gemma-3n-e2b-it",
        "gemma-3n-e4b-it",
        "imagen-3.0-generate-002",
        "imagen-4.0-generate-preview-06-06",
        "imagen-4.0-ultra-generate-preview-06-06",
        "learnlm-2.0-flash-experimental",
        "text-embedding-004"
    ],
    "groq": [
        "moonshotai/kimi-k2-instruct",
        "allam-2-7b",
        "compound-beta",
        "compound-beta-mini",
        "deepseek-r1-distill-llama-70b",
        "gemma2-9b-it",
        "llama-3.1-8b-instant",
        "llama-3.3-70b-versatile",
        "meta-llama/llama-4-maverick-17b-128e-instruct",
        "meta-llama/llama-4-scout-17b-16e-instruct",
        "meta-llama/llama-guard-4-12b",
        "meta-llama/llama-prompt-guard-2-22m",
        "meta-llama/llama-prompt-guard-2-86m",
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b",
        "playai-tts",
        "playai-tts-arabic",
        "qwen/qwen3-32b",
        "whisper-large-v3",
        "whisper-large-v3-turbo"
    ],
    "anthropic": [
        "claude-3-5-sonnet-20241022", "claude-3-5-haiku-20241022", 
        "claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"
    ],
    "mistral": [
        "mistral-large-latest", "mistral-medium-latest", "mistral-small-latest", 
        "mistral-tiny-latest", "codestral-latest", "pixtral-12b-latest",
        "open-mixtral-8x22b", "open-mixtral-8x7b", "open-mistral-7b"
    ],
    "deepseek": [
        "deepseek-chat",
        "deepseek-coder"
    ],
    "bfl": ["flux-kontext-max"],
    "suno": ["V3_5", "V4", "V4_5"],
    "wavespeed": [
        "wavespeed-ai/flux-kontext-dev-ultra-fast", "wavespeed-ai/flux-kontext-dev/multi-ultra-fast",
        "bytedance/portrait", "bytedance/seededit-v3",
        "minimax/hailuo-02/t2v-standard", "minimax/hailuo-02/t2v-pro",
        "minimax/hailuo-02/i2v-pro", "minimax/hailuo-02/i2v-standard",
        "kwaivgi/kling-v2.1-i2v-standard", "kwaivgi/kling-v2.1-i2v-pro", "kwaivgi/kling-v2.1-i2v-master",
        "google/veo3-fast", "wavespeed-ai/veo2-t2v", "wavespeed-ai/veo2-i2v"
    ],
    "openrouter": [
        "google/gemini-2.5-flash-image-preview",
        "agentica-org/deepcoder-14b-preview",
        "agentica-org/deepcoder-14b-preview:free",
        "ai21/jamba-large-1.7",
        "ai21/jamba-mini-1.7",
        "aion-labs/aion-1.0",
        "aion-labs/aion-1.0-mini",
        "aion-labs/aion-rp-llama-3.1-8b",
        "alfredpros/codellama-7b-instruct-solidity",
        "alpindale/goliath-120b",
        "amazon/nova-lite-v1",
        "amazon/nova-micro-v1",
        "amazon/nova-pro-v1",
        "anthracite-org/magnum-v2-72b",
        "anthracite-org/magnum-v4-72b",
        "anthropic/claude-3-haiku",
        "anthropic/claude-3-opus",
        "anthropic/claude-3.5-haiku",
        "anthropic/claude-3.5-haiku-20241022",
        "anthropic/claude-3.5-sonnet",
        "anthropic/claude-3.5-sonnet-20240620",
        "anthropic/claude-3.7-sonnet",
        "anthropic/claude-3.7-sonnet:thinking",
        "anthropic/claude-opus-4",
        "anthropic/claude-opus-4.1",
        "anthropic/claude-sonnet-4",
        "arcee-ai/coder-large",
        "arcee-ai/maestro-reasoning",
        "arcee-ai/spotlight",
        "arcee-ai/virtuoso-large",
        "arliai/qwq-32b-arliai-rpr-v1",
        "arliai/qwq-32b-arliai-rpr-v1:free",
        "baidu/ernie-4.5-21b-a3b",
        "baidu/ernie-4.5-300b-a47b",
        "baidu/ernie-4.5-vl-28b-a3b",
        "baidu/ernie-4.5-vl-424b-a47b",
        "bytedance/ui-tars-1.5-7b",
        "cognitivecomputations/dolphin-mistral-24b-venice-edition:free",
        "cognitivecomputations/dolphin-mixtral-8x22b",
        "cognitivecomputations/dolphin3.0-mistral-24b",
        "cognitivecomputations/dolphin3.0-mistral-24b:free",
        "cognitivecomputations/dolphin3.0-r1-mistral-24b",
        "cognitivecomputations/dolphin3.0-r1-mistral-24b:free",
        "cohere/command",
        "cohere/command-a",
        "cohere/command-r",
        "cohere/command-r-03-2024",
        "cohere/command-r-08-2024",
        "cohere/command-r-plus",
        "cohere/command-r-plus-04-2024",
        "cohere/command-r-plus-08-2024",
        "cohere/command-r7b-12-2024",
        "deepseek/deepseek-chat",
        "deepseek/deepseek-chat-v3-0324",
        "deepseek/deepseek-chat-v3-0324:free",
        "deepseek/deepseek-chat-v3.1",
        "deepseek/deepseek-chat-v3.1:free",
        "deepseek/deepseek-prover-v2",
        "deepseek/deepseek-r1",
        "deepseek/deepseek-r1-0528",
        "deepseek/deepseek-r1-0528-qwen3-8b",
        "deepseek/deepseek-r1-0528-qwen3-8b:free",
        "deepseek/deepseek-r1-0528:free",
        "deepseek/deepseek-r1-distill-llama-70b",
        "deepseek/deepseek-r1-distill-llama-70b:free",
        "deepseek/deepseek-r1-distill-llama-8b",
        "deepseek/deepseek-r1-distill-qwen-14b",
        "deepseek/deepseek-r1-distill-qwen-14b:free",
        "deepseek/deepseek-r1-distill-qwen-32b",
        "deepseek/deepseek-r1:free",
        "deepseek/deepseek-v3.1-base",
        "eleutherai/llemma_7b",
        "google/gemini-2.0-flash-001",
        "google/gemini-2.0-flash-exp:free",
        "google/gemini-2.0-flash-lite-001",
        "google/gemini-2.5-flash",
        "google/gemini-2.5-flash-image-preview",
        "google/gemini-2.5-flash-image-preview:free",
        "google/gemini-2.5-flash-lite",
        "google/gemini-2.5-flash-lite-preview-06-17",
        "google/gemini-2.5-pro",
        "google/gemini-2.5-pro-exp-03-25",
        "google/gemini-2.5-pro-preview",
        "google/gemini-2.5-pro-preview-05-06",
        "google/gemini-flash-1.5",
        "google/gemini-flash-1.5-8b",
        "google/gemini-pro-1.5",
        "google/gemma-2-27b-it",
        "google/gemma-2-9b-it",
        "google/gemma-2-9b-it:free",
        "google/gemma-3-12b-it",
        "google/gemma-3-12b-it:free",
        "google/gemma-3-27b-it",
        "google/gemma-3-27b-it:free",
        "google/gemma-3-4b-it",
        "google/gemma-3-4b-it:free",
        "google/gemma-3n-e2b-it:free",
        "google/gemma-3n-e4b-it",
        "google/gemma-3n-e4b-it:free",
        "gryphe/mythomax-l2-13b",
        "inception/mercury",
        "inception/mercury-coder",
        "infermatic/mn-inferor-12b",
        "inflection/inflection-3-pi",
        "inflection/inflection-3-productivity",
        "liquid/lfm-3b",
        "liquid/lfm-7b",
        "mancer/weaver",
        "meta-llama/llama-3-70b-instruct",
        "meta-llama/llama-3-8b-instruct",
        "meta-llama/llama-3.1-405b",
        "meta-llama/llama-3.1-405b-instruct",
        "meta-llama/llama-3.1-405b-instruct:free",
        "meta-llama/llama-3.1-70b-instruct",
        "meta-llama/llama-3.1-8b-instruct",
        "meta-llama/llama-3.2-11b-vision-instruct",
        "meta-llama/llama-3.2-1b-instruct",
        "meta-llama/llama-3.2-3b-instruct",
        "meta-llama/llama-3.2-3b-instruct:free",
        "meta-llama/llama-3.2-90b-vision-instruct",
        "meta-llama/llama-3.3-70b-instruct",
        "meta-llama/llama-3.3-70b-instruct:free",
        "meta-llama/llama-3.3-8b-instruct:free",
        "meta-llama/llama-4-maverick",
        "meta-llama/llama-4-maverick:free",
        "meta-llama/llama-4-scout",
        "meta-llama/llama-4-scout:free",
        "meta-llama/llama-guard-2-8b",
        "meta-llama/llama-guard-3-8b",
        "meta-llama/llama-guard-4-12b",
        "microsoft/mai-ds-r1",
        "microsoft/mai-ds-r1:free",
        "microsoft/phi-3-medium-128k-instruct",
        "microsoft/phi-3-mini-128k-instruct",
        "microsoft/phi-3.5-mini-128k-instruct",
        "microsoft/phi-4",
        "microsoft/phi-4-multimodal-instruct",
        "microsoft/phi-4-reasoning-plus",
        "microsoft/wizardlm-2-8x22b",
        "minimax/minimax-01",
        "minimax/minimax-m1",
        "mistralai/codestral-2501",
        "mistralai/codestral-2508",
        "mistralai/devstral-medium",
        "mistralai/devstral-small",
        "mistralai/devstral-small-2505",
        "mistralai/devstral-small-2505:free",
        "mistralai/magistral-medium-2506",
        "mistralai/magistral-medium-2506:thinking",
        "mistralai/magistral-small-2506",
        "mistralai/ministral-3b",
        "mistralai/ministral-8b",
        "mistralai/mistral-7b-instruct",
        "mistralai/mistral-7b-instruct-v0.1",
        "mistralai/mistral-7b-instruct-v0.3",
        "mistralai/mistral-7b-instruct:free",
        "mistralai/mistral-large",
        "mistralai/mistral-large-2407",
        "mistralai/mistral-large-2411",
        "mistralai/mistral-medium-3",
        "mistralai/mistral-medium-3.1",
        "mistralai/mistral-nemo",
        "mistralai/mistral-nemo:free",
        "mistralai/mistral-saba",
        "mistralai/mistral-small",
        "mistralai/mistral-small-24b-instruct-2501",
        "mistralai/mistral-small-24b-instruct-2501:free",
        "mistralai/mistral-small-3.1-24b-instruct",
        "mistralai/mistral-small-3.1-24b-instruct:free",
        "mistralai/mistral-small-3.2-24b-instruct",
        "mistralai/mistral-small-3.2-24b-instruct:free",
        "mistralai/mistral-tiny",
        "mistralai/mixtral-8x22b-instruct",
        "mistralai/mixtral-8x7b-instruct",
        "mistralai/pixtral-12b",
        "mistralai/pixtral-large-2411",
        "moonshotai/kimi-dev-72b",
        "moonshotai/kimi-dev-72b:free",
        "moonshotai/kimi-k2",
        "moonshotai/kimi-k2:free",
        "moonshotai/kimi-vl-a3b-thinking",
        "moonshotai/kimi-vl-a3b-thinking:free",
        "morph/morph-v3-fast",
        "morph/morph-v3-large",
        "neversleep/llama-3-lumimaid-70b",
        "neversleep/llama-3.1-lumimaid-8b",
        "neversleep/noromaid-20b",
        "nousresearch/deephermes-3-llama-3-8b-preview:free",
        "nousresearch/deephermes-3-mistral-24b-preview",
        "nousresearch/hermes-2-pro-llama-3-8b",
        "nousresearch/hermes-3-llama-3.1-405b",
        "nousresearch/hermes-3-llama-3.1-70b",
        "nousresearch/hermes-4-405b",
        "nousresearch/hermes-4-70b",
        "nvidia/llama-3.1-nemotron-70b-instruct",
        "nvidia/llama-3.1-nemotron-ultra-253b-v1",
        "nvidia/llama-3.1-nemotron-ultra-253b-v1:free",
        "nvidia/llama-3.3-nemotron-super-49b-v1",
        "openai/chatgpt-4o-latest",
        "openai/codex-mini",
        "openai/gpt-3.5-turbo",
        "openai/gpt-3.5-turbo-0613",
        "openai/gpt-3.5-turbo-16k",
        "openai/gpt-3.5-turbo-instruct",
        "openai/gpt-4",
        "openai/gpt-4-0314",
        "openai/gpt-4-1106-preview",
        "openai/gpt-4-turbo",
        "openai/gpt-4-turbo-preview",
        "openai/gpt-4.1",
        "openai/gpt-4.1-mini",
        "openai/gpt-4.1-nano",
        "openai/gpt-4o",
        "openai/gpt-4o-2024-05-13",
        "openai/gpt-4o-2024-08-06",
        "openai/gpt-4o-2024-11-20",
        "openai/gpt-4o-audio-preview",
        "openai/gpt-4o-mini",
        "openai/gpt-4o-mini-2024-07-18",
        "openai/gpt-4o-mini-search-preview",
        "openai/gpt-4o-search-preview",
        "openai/gpt-4o:extended",
        "openai/gpt-5",
        "openai/gpt-5-chat",
        "openai/gpt-5-mini",
        "openai/gpt-5-nano",
        "openai/gpt-oss-120b",
        "openai/gpt-oss-120b:free",
        "openai/gpt-oss-20b",
        "openai/gpt-oss-20b:free",
        "openai/o1",
        "openai/o1-mini",
        "openai/o1-mini-2024-09-12",
        "openai/o1-pro",
        "openai/o3",
        "openai/o3-mini",
        "openai/o3-mini-high",
        "openai/o3-pro",
        "openai/o4-mini",
        "openai/o4-mini-high",
        "opengvlab/internvl3-14b",
        "openrouter/auto",
        "perplexity/r1-1776",
        "perplexity/sonar",
        "perplexity/sonar-deep-research",
        "perplexity/sonar-pro",
        "perplexity/sonar-reasoning",
        "perplexity/sonar-reasoning-pro",
        "pygmalionai/mythalion-13b",
        "qwen/qwen-2.5-72b-instruct",
        "qwen/qwen-2.5-72b-instruct:free",
        "qwen/qwen-2.5-7b-instruct",
        "qwen/qwen-2.5-coder-32b-instruct",
        "qwen/qwen-2.5-coder-32b-instruct:free",
        "qwen/qwen-2.5-vl-7b-instruct",
        "qwen/qwen-max",
        "qwen/qwen-plus",
        "qwen/qwen-turbo",
        "qwen/qwen-vl-max",
        "qwen/qwen-vl-plus",
        "qwen/qwen2.5-vl-32b-instruct",
        "qwen/qwen2.5-vl-32b-instruct:free",
        "qwen/qwen2.5-vl-72b-instruct",
        "qwen/qwen2.5-vl-72b-instruct:free",
        "qwen/qwen3-14b",
        "qwen/qwen3-14b:free",
        "qwen/qwen3-235b-a22b",
        "qwen/qwen3-235b-a22b-2507",
        "qwen/qwen3-235b-a22b-thinking-2507",
        "qwen/qwen3-235b-a22b:free",
        "qwen/qwen3-30b-a3b",
        "qwen/qwen3-30b-a3b-instruct-2507",
        "qwen/qwen3-30b-a3b-thinking-2507",
        "qwen/qwen3-30b-a3b:free",
        "qwen/qwen3-32b",
        "qwen/qwen3-4b:free",
        "qwen/qwen3-8b",
        "qwen/qwen3-8b:free",
        "qwen/qwen3-coder",
        "qwen/qwen3-coder-30b-a3b-instruct",
        "qwen/qwen3-coder:free",
        "qwen/qwq-32b",
        "qwen/qwq-32b-preview",
        "qwen/qwq-32b:free",
        "raifle/sorcererlm-8x22b",
        "rekaai/reka-flash-3:free",
        "sao10k/l3-euryale-70b",
        "sao10k/l3-lunaris-8b",
        "sao10k/l3.1-euryale-70b",
        "sao10k/l3.3-euryale-70b",
        "scb10x/llama3.1-typhoon2-70b-instruct",
        "shisa-ai/shisa-v2-llama3.3-70b",
        "shisa-ai/shisa-v2-llama3.3-70b:free",
        "sophosympatheia/midnight-rose-70b",
        "switchpoint/router",
        "tencent/hunyuan-a13b-instruct",
        "tencent/hunyuan-a13b-instruct:free",
        "thedrummer/anubis-70b-v1.1",
        "thedrummer/anubis-pro-105b-v1",
        "thedrummer/rocinante-12b",
        "thedrummer/skyfall-36b-v2",
        "thedrummer/unslopnemo-12b",
        "thudm/glm-4-32b",
        "thudm/glm-4.1v-9b-thinking",
        "thudm/glm-z1-32b",
        "tngtech/deepseek-r1t-chimera",
        "tngtech/deepseek-r1t-chimera:free",
        "tngtech/deepseek-r1t2-chimera:free",
        "undi95/remm-slerp-l2-13b",
        "x-ai/grok-2-1212",
        "x-ai/grok-2-vision-1212",
        "x-ai/grok-3",
        "x-ai/grok-3-beta",
        "x-ai/grok-3-mini",
        "x-ai/grok-3-mini-beta",
        "x-ai/grok-4",
        "x-ai/grok-code-fast-1",
        "x-ai/grok-vision-beta",
        "z-ai/glm-4-32b",
        "z-ai/glm-4.5",
        "z-ai/glm-4.5-air",
        "z-ai/glm-4.5-air:free",
        "z-ai/glm-4.5v",
    ]
};

app.registerExtension({
    name: "ComfyLLMToolkit.APIProviderSelector",
    
    async nodeCreated(node) {
        if (node.comfyClass === "APIProviderSelectorNode") {
        // Find the widgets safely
        const providerWidget = node.widgets?.find(w => w.name === "provider");
        const modelWidget = node.widgets?.find(w => w.name === "llm_model");
        
        if (!providerWidget || !modelWidget) {
            console.warn("APIProviderSelector: Required widgets not found");
            return;
        }
        
        // Function to update model list based on provider
        const updateModelList = (provider) => {
            // Skip if graph is loading to prevent freezes
            if (app.graph?.isLoading) {
                return;
            }
            
            const models = PROVIDER_MODELS[provider] || ["No models available"];
            
            // Update the widget options
            modelWidget.options.values = models;
            
            // Set to first model if current selection is invalid
            if (!models.includes(modelWidget.value)) {
                modelWidget.value = models[0];
            }
            
            // Mark node as dirty to trigger re-render
            node.setDirtyCanvas(true);
            
            console.log(`APIProviderSelector: Updated models for ${provider}: ${models.length} models`);
        };
        
        // Store original callback
        const originalProviderCallback = providerWidget.callback;
        
        // Fetch models from JSON files or hardcoded fallback
        const updateLLMModels = async () => {
            const provider = providerWidget.value;
            
            try {
                // Use GET request to fetch from JSON files
                const response = await fetch(`/ComfyLLMToolkit/get_provider_models?provider=${encodeURIComponent(provider)}`);

                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }

                const data = await response.json();
                console.log(`APIProviderSelector: Loaded models from ${data.source}:`, data);

                if (data.models && Array.isArray(data.models) && data.models.length > 0) {
                    // Update both the cache and widget
                    PROVIDER_MODELS[provider] = data.models;
                    modelWidget.options.values = data.models;
                    
                    // Set to first model if current selection is invalid
                    if (!data.models.includes(modelWidget.value)) {
                        modelWidget.value = data.models[0];
                    }
                    
                    node.setDirtyCanvas(true);
                    console.log(`APIProviderSelector: Updated ${data.models.length} models for ${provider} from ${data.source}`);
                } else {
                    throw new Error("No models available");
                }
            } catch (error) {
                console.error("Error updating models:", error);
                
                // Use fallback models
                const models = PROVIDER_MODELS[provider] || ["No models available"];
                modelWidget.options.values = models;
                modelWidget.value = models[0];
                node.setDirtyCanvas(true);
            }
        };
        
        // Add manual refresh button as a widget
        if (!node.widgets.find(w => w.name === "_refresh_models")) {
            const refreshButton = node.addWidget("button", "_refresh_models", "Refresh Models", () => {
                updateLLMModels();
            });
            refreshButton.serialize = false; // Don't save button in workflow
        }
        
        // Update callback to use new function
        providerWidget.callback = function() {
            // Call original callback if it exists
            if (originalProviderCallback) {
                originalProviderCallback.apply(this, arguments);
            }
            
            // Fetch fresh models when provider changes
            updateLLMModels();
        };
        
        // Initial model fetch
        // updateLLMModels();
        }
    },
    
    async loadedGraphNode(node) {
        if (node.comfyClass === "APIProviderSelectorNode") {
        // Re-initialize model list when loading a saved workflow
        const providerWidget = node.widgets?.find(w => w.name === "provider");
        const modelWidget = node.widgets?.find(w => w.name === "llm_model");
        
        if (providerWidget && modelWidget) {
            const models = PROVIDER_MODELS[providerWidget.value] || ["No models available"];
            modelWidget.options.values = models;
            
            // Ensure saved model is still valid
            if (!models.includes(modelWidget.value)) {
                modelWidget.value = models[0];
            }
            
            console.log(`APIProviderSelector: Restored ${providerWidget.value} with model ${modelWidget.value}`);
            }
        }
    }
});