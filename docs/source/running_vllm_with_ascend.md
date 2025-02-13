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

Find more information about the sampling parameters [<u>here</u>](https://docs.vllm.ai/en/stable/api/inference_params.html#sampling-params).

If you run this script successfully, you can see the info shown below:

```bash
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
-p 8000:8000 \
-it vllm-ascend:1.0 bash
```

> [!TIP]
> To use your local model, you should mount your model dir into the container, e.g. `-v /root/models:/root/models`.

> [!NOTE]
> You can set `davinci0 ~ davinci7` to specify a different NPU. Get more info about your device using `npu-smi info`.

### Online Serving on a Single NPU

vLLM can be deployed as a server that implements the OpenAI API protocol. This allows vLLM to be used as a drop-in replacement for applications using OpenAI API. By default, it starts the server at `http://localhost:8000`. You can specify the address with `--host` and `--port` arguments.

Run the following command to start the vLLM server on a single NPU:

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct
```

> [!TIP]
> You can use your local models when launching the vllm server by replacing the value of `model` with `path/to/model`, e.g. `/root/models/Qwen/Qwen2.5-7B-Instruct`.

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
> You can use your local models when quering the server by replacing the value of `model` with `path/to/model`, e.g. `/root/models/Qwen/Qwen2.5-7B-Instruct`.

If you query the server successfully, you can see the info shown below (client):

```bash
{"id":"cmpl-574f00e342904692a73fb6c1c986c521","object":"text_completion","created":1739435675,"model":"/root/models/Qwen/Qwen2.5-7B-Instruct","choices":[{"index":0,"text":" city of neighborhoods, each with its","logprobs":null,"finish_reason":"length","stop_reason":null,"prompt_logprobs":null}],"usage":{"prompt_tokens":4,"total_tokens":11,"completion_tokens":7,"prompt_tokens_details":null}}
```

Logs of the vllm server:

```bash
INFO:     172.17.0.1:49518 - "POST /v1/completions HTTP/1.1" 200 OK
...
INFO 02-13 08:34:35 logger.py:39] Received request cmpl-574f00e342904692a73fb6c1c986c521-0: prompt: 'San Francisco is a', params: SamplingParams(n=1, presence_penalty=0.0, frequency_penalty=0.0, repetition_penalty=1.0, temperature=0.0, top_p=1.0, top_k=-1, min_p=0.0, seed=None, stop=[], stop_token_ids=[], bad_words=[], include_stop_str_in_output=False, ignore_eos=False, max_tokens=7, min_tokens=0, logprobs=None, prompt_logprobs=None, skip_special_tokens=True, spaces_between_special_tokens=True, truncate_prompt_tokens=None, guided_decoding=None), prompt_token_ids: [23729, 12879, 374, 264], lora_request: None, prompt_adapter_request: None.
```

## Distributed Inference

vLLM supports distributed tensor-parallel and pipeline-parallel inference and serving. To run multi-GPU inference with the `LLM` class, set the `tensor_parallel_size` argument to the number of NPUs you want to use.

```python
from vllm import LLM

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
llm = LLM(model="Qwen/Qwen2.5-7B-Instruct",
          tensor_parallel_size=2,
          distributed_executor_backend="mp")

outputs = llm.generate(prompts, sampling_params)
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```

If you run this script successfully, you can see the info shown below:

```bash
Prompt: 'Hello, my name is', Generated text: ' Daniel and I am an 8th grade student at York Middle School. I'
Prompt: 'The president of the United States is', Generated text: ' Statesman A, and the vice president is Statesman B. If they are'
Prompt: 'The capital of France is', Generated text: ' the city of Paris. This is a fact that can be found in any geography'
Prompt: 'The future of AI is', Generated text: ' following you. As the technology advances, a new report from the Institute for the'
```

## FAQ

### 1. Can't Find CANN Driver Lib

Error info:

```bash
ImportError: libascend_hal.so: cannot open shared object file: No such file or directory
```

Run this command:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

### 2. NPU OOM

Error info:

```bash
RuntimeError: NPU out of memory. Tried to allocate 260.00 MiB (NPU 0; 28.50 GiB total capacity; 10.24 GiB already allocated; 10.24 GiB current active; 149.57 MiB free; 10.28 GiB reserved in total by PyTorch) 
```

> [!NOTE]
> `max_split_size_mb` prevents the native allocator from splitting blocks larger than this size (in MB). This can reduce fragmentation and may allow some borderline workloads to complete without running out of memory.

Run this command:

```bash
export PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:256
```

Set `max_split_size_mb` to any value lower than you need to allocate (`260.00 MiB` here).

Find more details [<u>here</u>](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/envref/envref_07_0061.html).

### 3. Model's Max Seq Length is too Large

Error info:

```bash
ValueError: The model's max seq len (32768) is larger than the maximum number of tokens that can be stored in KV cache (26240). Try increasing `gpu_memory_utilization` or decreasing `max_model_len` when initializing the engine.
```

Add `--max_model_len` option when launching the vllm server:

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct --max_model_len 26240
```

### 4. Can't Connect to vLLM Server

Error info:

```bash
curl: (7) Failed to connect to localhost port 8000: Connection refused
```

> [!NOTE]
> Are you in the same container or does the container expose the port (including `-p 8000:8000` in the Docker run command)? If you are using two containers, does the client container use the host network (including `--net=host` in the Docker run command)? This seems like a networking issue, so you want to make sure your ports are set up correctly.

Add `-p 8000:8000` option when launching the docker container:

```bash
docker run ... -p 8000:8000 ...
```
