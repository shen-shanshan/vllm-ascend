# Ascend plugin for vLLM

vLLM Ascend plugin (vllm-ascend) is a community maintained hardware plugin for running vLLM on the Ascend NPU.

This plugin is the recommended approach for supporting the Ascend backend within the vLLM community. It adheres to the principles outlined in the [<u>[RFC]: Hardware pluggable</u>](https://github.com/vllm-project/vllm/issues/11162), providing a hardware-pluggable interface that decouples the integration of the Ascend NPU with vLLM.

By using vLLM Ascend plugin, popular open-source models, including Transformer-like, Mixture-of-Expert, Embedding, Multi-modal LLMs can run seamlessly on the Ascend NPU.

## Contents

- [<u>Quick Start</u>](./quick_start.md)
- [<u>Installation</u>](./installation.md)
- Usage
  - [<u>Running vLLM with Ascend</u>](./usage/running_vllm_with_ascend.md)
  - [<u>Feature Support</u>](./usage/feature_support.md)
  - [<u>Supported Models</u>](./usage/supported_models.md)
