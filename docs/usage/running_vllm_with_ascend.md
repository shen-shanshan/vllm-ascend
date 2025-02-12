# Running vLLM with Ascend

## Preparation

### Check CANN Environment

Check your CANN environment:

```bash
cd /usr/local/Ascend/ascend-toolkit/latest/<arch>-linux  # <arch>: aarch64 or x86_64
cat ascend_toolkit_install.info
```

The cann version should >= `8.0.RC2`, for example:

```bash
package_name=Ascend-cann-toolkit
version=8.0.RC3
```

### Check NPU Device

Check your available NPU chips:

```bash
npu-smi info
```

### Download Model

Install modelscope:

```bash
pip install modelscope
```

Download model with modelscope python sdk:

```python
# /root/models/model_download.py
from modelscope import snapshot_download

model_dir = snapshot_download('Qwen/Qwen2.5-7B-Instruct', cache_dir='/root/models')
```

Start downloading:

```bash
python model_download.py
```

To use models from ModelScope instead of HuggingFace Hub, set an environment variable:

```bash
export VLLM_USE_MODELSCOPE=True
```

## Offline Inference

### Install vllm and vllm-ascend

Install vllm and vllm-ascend directly with pip:

```bash
pip install vllm vllm-ascend
```

### Offline Inference on a Single NPU

Run the following script to execute offline inference on a single NPU:

```python
from vllm import LLM, SamplingParams

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
llm = LLM(model="Qwen/Qwen2.5-7B-Instruct")

outputs = llm.generate(prompts, sampling_params)
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```

> [!TIP]
> You can use your local models for offline inference by replacing the value of `model` in `LLM()` with `path/to/model`, e.g. `/root/models/Qwen/Qwen2.5-7B-Instruct`.

> [!NOTE]
>
> - `temperature`: Float that controls the randomness of the sampling. Lower values make the model more deterministic, while higher values make the model more random.
> - `top_p`: Float that controls the cumulative probability of the top tokens to consider.

You can find more information about the sampling parameters [<u>here</u>](https://docs.vllm.ai/en/stable/api/inference_params.html#sampling-params).

If you run this script successfully, you can see the info shown below:

```bash
Processed prompts: 100%|███████████████████████| 4/4 [00:00<00:00,  4.10it/s, est. speed input: 22.56 toks/s, output: 65.62 toks/s]
Prompt: 'Hello, my name is', Generated text: ' Daniel and I am an 8th grade student at York Middle School. I'
Prompt: 'The president of the United States is', Generated text: ' Statesman A, and the vice president is Statesman B. If they are'
Prompt: 'The capital of France is', Generated text: ' the city of Paris. This is a fact that can be found in any geography'
Prompt: 'The future of AI is', Generated text: ' following you. As the technology advances, a new report from the Institute for the'
```

## Online Serving

### Run Docker Container

Build your docker image using `vllm-ascend/Dockerfile`:

```bash
docker build -t vllm-ascend:1.0 .
```

> [!NOTE]
> `.` is the dir of your Dockerfile.

Launch your container:

```bash
docker run \
    --name vllm-ascend \
    --device /dev/davinci0 \
    --device /dev/davinci_manager \
    --device /dev/devmm_svm \
    --device /dev/hisi_hdc \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
    -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -v /root/models:/root/models \
    -it vllm-ascend bash
```

> [!TIP]
> To use your local model, you should mount your model dir to container, e.g. `-v /root/models:/root/models`.

> [!NOTE]
> You can set `davinci0 ~ davinci7` to specify a different NPU. Find more info about your device using `npu-smi info`.

### Online Serving on a Single NPU

vLLM can be deployed as a server that implements the OpenAI API protocol. This allows vLLM to be used as a drop-in replacement for applications using OpenAI API. By default, it starts the server at `http://localhost:8000`. You can specify the address with `--host` and `--port` arguments.

Run the following command to start the vLLM server on a single NPU:

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct
```

Once your server is started, you can query the model with input prompts:

```bash
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "prompt": "San Francisco is a",
        "max_tokens": 7,
        "temperature": 0
    }'
```

> [!TIP]
> You can use your local models when launching the vllm server or quering the model by replacing the value of `model` with `path/to/model`, e.g. `/root/models/Qwen/Qwen2.5-7B-Instruct`.

If you query the server successfully, you can see the info shown below:

```bash
...
```

## Distributed Inference

vLLM supports distributed tensor-parallel and pipeline-parallel inference and serving. To run multi-GPU inference with the `LLM` class, set the `tensor_parallel_size` argument to the number of GPUs you want to use.

```python
from vllm import LLM

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
llm = LLM(model="Qwen/Qwen2.5-7B-Instruct", tensor_parallel_size=4)

outputs = llm.generate(prompts, sampling_params)
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```

If you run this script successfully, you can see the info shown below:

```bash
...
```
