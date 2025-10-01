# ComfyUI LLM Toolkit

A custom node collection for integrating various LLM (Large Language Model) providers with ComfyUI.
![llmtoolkit_thum](https://github.com/user-attachments/assets/610a2135-3a03-4cc3-b75a-3ee44f70f0d8)
## NEW GENERAL IMAGE AND VIDEO CFGs
<img width="1367" height="869" alt="image" src="https://github.com/user-attachments/assets/ed85c580-c981-4e36-be72-c4e00192391b" />

## Features
- True Context-to-Context node connections one-input / one-output design
<img width="1107" alt="Screenshot 2025-04-28 at 16 51 13" src="https://github.com/user-attachments/assets/6613b013-18c4-4a7d-b40e-6ae025ad21d0" />

- Generators run even if nothing is connected, and they are the only node in the workflow 
![imagerun](https://github.com/user-attachments/assets/bac9f8a0-6342-4000-b3ef-66fc45c1bdef)

- Streaming Output directly on the node UI
![stream](https://github.com/user-attachments/assets/eca923ae-f3f2-4014-8937-c8eda53eb680)

- Runs openai latest image model GPT-image-1 and we include various templates 

- Text generation using various LLM providers (OpenAI and local models, etc.)
- Provider selection and configuration with dynamic model fetching
- API key management
- Seamless integration with ComfyUI workflows

## Installation

	@@ -33,6 +43,7 @@ The following Python packages are required:
- pyyaml
- python-dotenv
- requests
- openai

## Installation

1. Clone this repository into your ComfyUI custom_nodes directory:
   ```bash
   cd /path/to/ComfyUI/custom_nodes
   git clone https://github.com/comfy-deploy/comfyui-llm-toolkit.git
   ```

2. Install the required dependencies:
   ```bash
   cd comfyui-llm-toolkit
   pip install -r requirements.txt
   ```

3. Restart ComfyUI

## Dependencies

The following Python packages are required:
- aiohttp
- pyyaml
- python-dotenv
- requests

## Configuration

1. API keys for various providers can be stored in a `.env` file in the root directory:
   ```
   OPENAI_API_KEY=your_openai_key
   # Add other provider keys as needed
   ```

2. Alternatively, you can provide API keys directly in the node interface.

## Usage

### LLM Provider Selector Node
Use this node to select the LLM provider and model. It outputs provider configuration within the wildcard "context" output.

- When you change the provider, the model dropdown updates dynamically with available models
- IP and Port fields appear/disappear based on whether the provider needs them
- The output is a single "context" type that contains the provider configuration

### Generate Text Node
Basic text generation with the selected provider and model.

- Connect it to the LLM Provider Selector by connecting their "context" ports
- The node automatically detects if provider config is present in the input
- You can override the model with the dropdown
- The "context" output contains both the original input data and the LLM response

## Unified ONE-INPUT / ONE-OUTPUT Architecture

The LLM Toolkit uses a single "context" input/output approach for maximum flexibility:

1. **Single Connection Point**: Each node has just one wildcard input and one wildcard output named "context"
2. **Smart Data Handling**: 
   - Provider config is embedded within the "context" data structure
   - Each node intelligently extracts the data it needs from the "context" input
   - Nodes preserve all input data and add their own data to the "context" output
3. **Cascading Data Flow**: As data flows through nodes, it accumulates in the "context" structure

For example, with nodes A → B → C:
- Node A creates an "context" with provider config
- Node B receives A's "context", extracts the provider config, and adds LLM response to the "context"
- Node C receives B's "context" which now contains both provider config and LLM response

This allows you to:
- Chain multiple LLM operations with a single connection
- Preserve and accumulate data throughout the workflow
- Easily integrate with other ComfyUI nodes

## Supported Providers

- OpenAI (Default: gpt-4o-mini)
- Ollama (local)


## Troubleshooting

If you encounter model list update issues:
1. Make sure ComfyUI is running with the correct server configuration
2. Check that JavaScript is enabled in your browser
3. Verify that your API keys are correctly set in the .env file or provided in the node

If you encounter import errors:
1. Make sure you've installed all dependencies: `pip install -r requirements.txt`
2. Verify that you've placed the custom node in the correct directory
3. Restart ComfyUI after installation

## Testing Your Installation

Run the included test script to verify your setup:
```bash
cd /path/to/ComfyUI/custom_nodes/comfyui-llm-toolkit
python test_js_integration.py
```


