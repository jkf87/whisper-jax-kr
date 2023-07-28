# Whisper JAX

이 리포지토리에는 OpenAI의 [Whisper Model]https://arxiv.org/abs/2212.04356), 대체로 구축됨를 사용하여 포옹하는 허깅페이스 트랜스포머 Whisper 구현을 살펴보세요.
OpenAI의 PyTorch 코드와 비교했을 때, Whisper JAX는 **70배** 이상 실행됩니다.
더 빨라져 가장 빠른 Whisper 구현이 가능해졌습니다.

JAX 코드는 CPU, GPU 및 TPU에서 호환됩니다, 
독립적으로 실행할 수 있습니다(파이프라인 사용(#pipeline-.사용법)) 
또는 추론 엔드포인트로 사용할 수 있습니다([엔드포인트 만들기(#creating-.엔드포인트)).

클라우드 TPU에서 Whisper JAX를 실행하는 빠른 시작 가이드는 30분 분량의 오디오를 약 30초 분량으로 트랜스크립션한 다음 Kaggle 노트북을 참조하세요:

[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/sgandhi99/whisper-jax-tpu)

위스퍼 JAX 모델도 허깅 페이스 허브에서 데모로 실행되고 있습니다:

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/sanchit-gandhi/whisper-jax)

## Installation

Whisper JAX는 Python 3.9 및 JAX 버전 0.4.5를 사용하여 테스트되었습니다. 
설치 시에는 이미 최신버전의 JAX 패키지를 설치해야 합니다. 공식 JAX 설치 가이드를 사용하여 설치할 수 있습니다.: https://github.com/google/jax#installation

적절한 버전의 JAX가 설치되면 pip를 통해 Whisper JAX를 설치할 수 있습니다.:
```
pip install git+https://github.com/sanchit-gandhi/whisper-jax.git
```

Whisper JAX 패키지를 최신 버전으로 업데이트하려면 다음을 실행하면 됩니다.:
```
pip install --upgrade --no-deps --force-reinstall git+https://github.com/sanchit-gandhi/whisper-jax.git
```

## Pipeline Usage

The recommended way of running Whisper JAX is through the [`FlaxWhisperPipline`](https://github.com/sanchit-gandhi/whisper-jax/blob/main/whisper_jax/pipeline.py#L57) abstraction class. 
이 클래스는 가속기 장치에서 데이터 병렬 처리를 위한 생성 메서드를 래핑할 뿐만 아니라 필요한 모든 전처리 및 후처리를 처리합니다.

Whisper JAX는 JAX의 [`pmap`](https://jax.readthedocs.io/ko/latest/_autosummary/jax.pmap.html) 함수는 GPU/TPU 장치 간 데이터 병렬 처리를 위한 함수입니다. 
이 함수는 _JIT(Just In Time)_입니다.함수가 처음 호출될 때 컴파일됩니다. 그 이후에는 함수가 _캐시_되어 매우 빠른 시간 내에 실행될 수 있습니다.:

```python
from whisper_jax import FlaxWhisperPipline

# instantiate pipeline
pipeline = FlaxWhisperPipline("openai/whisper-large-v2")

# JIT compile the forward call - slow, but we only do once
text = pipeline("audio.mp3")

# used cached function thereafter - super fast!!
text = pipeline("audio.mp3")
```

### Half-Precision

파이프라인을 인스턴스화할 때 dtype 인수를 전달하면 모델 계산을 반정밀로 실행할 수 있습니다. 
이렇게 하면 중간 텐서를 반정밀도로 저장하여 계산 속도를 상당히 높입니다. 정밀도에는 변화가 없습니다

대부분의 GPU의 경우, dtype은 `jnp.float16`로 설정해야 합니다. A100 GPU 또는 TPU의 경우, dtype을 `jnp.bfloat16`로 설정해야 합니다.:
```python
from whisper_jax import FlaxWhisperPipline
import jax.numpy as jnp

# instantiate pipeline in bfloat16
pipeline = FlaxWhisperPipline("openai/whisper-large-v2", dtype=jnp.bfloat16)
```

### Batching
Whisper JAX는 가속기 장치에서 단일 오디오 입력을 일괄 처리하는 _ 옵션도 제공합니다. 
오디오가 먼저를 30초 세그먼트로 분할한 다음 모델에 청크를 전송하여 병렬로 전사합니다. 
그 결과트랜스 크립 션은 경계에서 다시 스티칭되어 단일하고 균일 한 트랜스 크립 션을 제공합니다.
실제로 일괄 처리는 배치 크기를 충분히 크게 선택하면 오디오 샘플을 순차적으로 전사할 때보다 10배 빠른 속도를 제공하며, WER^1

일괄 처리를 활성화하려면 파이프라인을 인스턴스화할 때 `batch_size` 매개 변수를 전달하세요:

```python
from whisper_jax import FlaxWhisperPipline

# instantiate pipeline with batching
pipeline = FlaxWhisperPipline("openai/whisper-large-v2", batch_size=16)
```

### Task

기본적으로 파이프라인은 음성 파일이 사용된 언어로 오디오 파일을 트랜스크립션합니다. For speech translation, set the 
`task` argument to `"translate"`:

```python
# translate
text = pipeline("audio.mp3", task="translate")
```

### Timestamps

[`FlaxWhisperPipeline`](https://github.com/sanchit-gandhi/whisper-jax/blob/main/whisper_jax/pipeline.py#L57)도 타임스탬프 예측을 지원합니다. 
타임스탬프를 활성화하면 두 번째 JIT 컴파일이 필요합니다.착신 전환, 이번에는 타임스탬프 출력 포함:

```python
# transcribe and return timestamps
outputs = pipeline("audio.mp3",  task="transcribe", return_timestamps=True)
text = outputs["text"]  # transcription
chunks = outputs["chunks"]  # transcription + timestamps
```

### Putting it all together
다음 코드 스니펫에서는 배치가 활성화된 상태에서 bfloat16 정밀도로 모델을 인스턴스화하고 오디오 파일을 트랜스크립션합니다.
타임스탬프 토큰 반환: 

```python
from whisper_jax import FlaxWhisperPipline
import jax.numpy as jnp

# instantiate pipeline with bfloat16 and enable batching
pipeline = FlaxWhisperPipline("openai/whisper-large-v2", dtype=jnp.bfloat16, batch_size=16)

# transcribe and return timestamps
outputs = pipeline("audio.mp3",  task="transcribe", return_timestamps=True)
```

## Model Usage

위스퍼 JAX 모델은 원래의 허깅 페이스와 거의 동일한 방식으로 더 세분화된 수준에서 사용할 수 있습니다.트랜스포머 구현. 
This requires the Whisper processor to be loaded separately to the model to handle the
pre- and post-processing, and the generate function to be wrapped using `pmap` by hand:

```python
import jax.numpy as jnp
from datasets import load_dataset
from flax.jax_utils import replicate
from flax.training.common_utils import shard
from jax import device_get, pmap
from transformers import WhisperProcessor

from whisper_jax import FlaxWhisperForConditionalGeneration

# load the processor and model
processor = WhisperProcessor.from_pretrained("openai/whisper-large-v2")
model, params = FlaxWhisperForConditionalGeneration.from_pretrained(
    "openai/whisper-large-v2", dtype=jnp.bfloat16, _do_init=False,
)

def generate_fn(input_features):
    pred_ids = model.generate(
        input_features, task="transcribe", return_timestamps=False, max_length=model.config.max_length, params=params,
    )
    return pred_ids.sequences

# pmap the generate function for data parallelism
p_generate = pmap(generate_fn, "input_features")
# replicate the parameters across devices
params = replicate(params)

# load a dummy sample from the LibriSpeech dataset
ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
sample = ds[0]["audio"]

# pre-process: convert the audio array to log-mel input features
input_features = processor(sample["array"], sampling_rate=sample["sampling_rate"], return_tensors="np").input_features
# replicate the input features across devices for DP
input_features = shard(input_features)

# run the forward pass (JIT compiled the first time it is called)
pred_ids = p_generate(input_features)
output_ids = device_get(pred_ids.reshape(-1, model.config.max_length))

# post-process: convert tokens ids to text string
transcription = processor.batch_decode(pred_ids, skip_special_tokens=True)
```

## Available Models and Languages
아마 웨이트가 있는 허깅 페이스 허브의 모든 위스퍼 모델은 위스퍼 JAX와 호환됩니다. 여기에는 다음이 포함되지만 이에 국한되지는 않습니다,공식 OpenAI Whisper 체크 포인트:

| Size     | Parameters | English-only                                         | Multilingual                                        |
|----------|------------|------------------------------------------------------|-----------------------------------------------------|
| tiny     | 39 M       | [✓](https://huggingface.co/openai/whisper-tiny.en)   | [✓](https://huggingface.co/openai/whisper-tiny)     |
| base     | 74 M       | [✓](https://huggingface.co/openai/whisper-base.en)   | [✓](https://huggingface.co/openai/whisper-base)     |
| small    | 244 M      | [✓](https://huggingface.co/openai/whisper-small.en)  | [✓](https://huggingface.co/openai/whisper-small)    |
| medium   | 769 M      | [✓](https://huggingface.co/openai/whisper-medium.en) | [✓](https://huggingface.co/openai/whisper-medium)   |
| large    | 1550 M     | x                                                    | [✓](https://huggingface.co/openai/whisper-large)    |
| large-v2 | 1550 M     | x                                                    | [✓](https://huggingface.co/openai/whisper-large-v2) |

Whisper JAX에서 미세 조정된 Whisper 체크포인트를 사용하려면 먼저 PyTorch 가중치를 Flax로 변환해야 합니다.
이는 파이토치 상태 딕셔너리를 프로즌 아마로 변환하는 `from_pt` 인수를 사용하면 간단하게 처리할 수 있습니다.
매개변수 사전을 즉시 변환할 수 있습니다. 그런 다음 변환된 Flax 가중치를 Hub로 푸시하여 Flax에서 바로 사용할 수 있습니다.를 사용할 수 있습니다. 
파이토치에서 플랙스로 가중치를 변환하려면 파이토치와 플랙스를 모두 설치해야 한다는 점에 유의하세요.

For example, to convert the fine-tuned checkpoint [`sanchit-gandhi/whisper-small-hi`](https://huggingface.co/sanchit-gandhi/whisper-small-hi) from the blog post [Fine-Tuning Whisper](https://huggingface.co/blog/fine-tune-whisper):
```python
from whisper_jax import FlaxWhisperForConditionalGeneration, FlaxWhisperPipline
import jax.numpy as jnp

checkpoint_id = "sanchit-gandhi/whisper-small-hi"
# convert PyTorch weights to Flax
model = FlaxWhisperForConditionalGeneration.from_pretrained(checkpoint_id, from_pt=True)
# push converted weights to the Hub
model.push_to_hub(checkpoint_id)

# now we can load the Flax weights directly as required
pipeline = FlaxWhisperPipline(checkpoint_id, dtype=jnp.bfloat16, batch_size=16)
```

## Advanced Usage
More advanced users may wish to explore different parallelisation techniques. The Whisper JAX code is
built on-top of the [T5x codebase](https://github.com/google-research/t5x), meaning it can be run using model, activation, and data parallelism using the T5x 
partitioning convention. To use T5x partitioning, the logical axis rules and number of model partitions must be defined.
For more details, the user is referred to the official T5x partitioning guide: https://github.com/google-research/t5x/blob/main/docs/usage/partitioning.md

### Pipeline
The following code snippet demonstrates how data parallelism can be achieved using the pipeline `shard_params` method in 
an entirely equivalent way to `pmap`:

```python
from whisper_jax import FlaxWhisperPipline
import jax.numpy as jnp

# 2D parameter and activation partitioning for DP
logical_axis_rules_dp = (
    ("batch", "data"),
    ("mlp", None),
    ("heads", None),
    ("vocab", None),
    ("embed", None),
    ("embed", None),
    ("joined_kv", None),
    ("kv", None),
    ("length", None),
    ("num_mel", None),
    ("channels", None),
)

pipeline = FlaxWhisperPipline("openai/whisper-large-v2", dtype=jnp.bfloat16, batch_size=16)
pipeline.shard_params(num_mp_partitions=1, logical_axis_rules=logical_axis_rules_dp)
```

### Model
It is also possible to use the Whisper JAX model with T5x partitioning by defining a T5x inference state and T5x partitioner:

```python
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import freeze
from jax.sharding import PartitionSpec as P

from whisper_jax import FlaxWhisperForConditionalGeneration, InferenceState, PjitPartitioner


# 2D parameter and activation partitioning for DP
logical_axis_rules_dp = [
    ("batch", "data"),
    ("mlp", None),
    ("heads", None),
    ("vocab", None),
    ("embed", None),
    ("embed", None),
    ("joined_kv", None),
    ("kv", None),
    ("length", None),
    ("num_mel", None),
    ("channels", None),
]

model, params = FlaxWhisperForConditionalGeneration.from_pretrained(
    "openai/whisper-large-v2",
    _do_init=False,
    dtype=jnp.bfloat16,
)


def init_fn():
    input_shape = (1, 80, 3000)

    input_features = jnp.zeros(input_shape, dtype="f4")
    input_features = input_features.at[(..., -1)].set(model.config.eos_token_id)

    decoder_input_ids = jnp.zeros((input_shape[0], 1), dtype="i4")
    decoder_attention_mask = jnp.ones_like(decoder_input_ids)

    batch_size, sequence_length = decoder_input_ids.shape
    decoder_position_ids = jnp.broadcast_to(jnp.arange(sequence_length)[None, :], (batch_size, sequence_length))

    rng = jax.random.PRNGKey(0)
    init_params = model.module.init(
        rng,
        input_features=input_features,
        decoder_input_ids=decoder_input_ids,
        decoder_attention_mask=decoder_attention_mask,
        decoder_position_ids=decoder_position_ids,
        return_dict=False,
    )
    return init_params


# Axis names metadata
param_axes = jax.eval_shape(init_fn)["params_axes"]

# Create InferenceState, since the partitioner expects it
state = InferenceState(
    step=jnp.array(0),
    params=freeze(model.params_shape_tree),
    params_axes=freeze(param_axes),
    flax_mutables=None,
    flax_mutables_axes=param_axes,
)

# Define the pjit partitioner with 1 model partition
partitioner = PjitPartitioner(
    num_partitions=1,
    logical_axis_rules=logical_axis_rules_dp,
)

mesh_axes = partitioner.get_mesh_axes(state)
params_spec = mesh_axes.params

p_shard_params = partitioner.partition(model.to_bf16, (params_spec,), params_spec)


def generate(params, input_features):
    output_ids = model.generate(input_features, params=params, max_length=model.config.max_length).sequences
    return output_ids


p_generate = partitioner.partition(
    generate,
    in_axis_resources=(params_spec, P("data")),
    out_axis_resources=P("data"),
)

# This will auto-magically run in mesh context
params = p_shard_params(freeze(params))

# you can now run the forward pass with: 
# pred_ids = p_generate(input_features)
```

## Benchmarks

We compare Whisper JAX to the official [OpenAI implementation](https://github.com/openai/whisper) and the [🤗 Transformers 
implementation](https://huggingface.co/docs/transformers/model_doc/whisper). We benchmark the models on audio samples of 
increasing length and report the average inference time in seconds over 10 repeat runs. For all three systems, we pass a 
pre-loaded audio file to the model and measure the time for the forward pass. Leaving the task of loading the audio file 
to the systems adds an equal offset to all the benchmark times, so the actual time for loading **and** transcribing an 
audio file will be higher than the reported numbers.

OpenAI and Transformers both run in PyTorch on GPU. Whisper JAX runs in JAX on GPU and TPU. OpenAI transcribes the audio 
sequentially in the order it is spoken. Both Transformers and Whisper JAX use a batching algorithm, where chunks of audio 
are batched together and transcribed in parallel (see section [Batching](#batching)).

**Table 1:** Average inference time in seconds for audio files of increasing length. GPU device is a single A100 40GB GPU. 
TPU device is a single TPU v4-8.

<div align="center">

|           | OpenAI  | Transformers | Whisper JAX | Whisper JAX |
|-----------|---------|--------------|-------------|-------------|
|           |         |              |             |             |
| Framework | PyTorch | PyTorch      | JAX         | JAX         |
| Backend   | GPU     | GPU          | GPU         | TPU         |
|           |         |              |             |             |
| 1 min     | 13.8    | 4.54         | 1.72        | 0.45        |
| 10 min    | 108.3   | 20.2         | 9.38        | 2.01        |
| 1 hour    | 1001.0  | 126.1        | 75.3        | 13.8        |
|           |         |              |             |             |

</div>

## Creating an Endpoint

The Whisper JAX model is running as a demo on the Hugging Face Hub:

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/sanchit-gandhi/whisper-jax)

However, at peak times there may be a queue of users that limit how quickly your audio input is transcribed. In this case,
you may benefit from running the model yourself, such that you have unrestricted access to the Whisper JAX model.

If you are just interested in running the model in a standalone Python script, refer to the Kaggle notebook Whisper JAX TPU:

[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/sgandhi99/whisper-jax-tpu)

Otherwise, we provide all the necessary code for creating an inference endpoint. To obtain this code, first clone the 
repository on the GPU/TPU on which you want to host the endpoint:
```
git clone https://github.com/sanchit-gandhi/whisper-jax
```

And then install Whisper JAX from source, with the required additional endpoint dependencies:
```
cd whisper-jax
pip install -e .["endpoint"]
```

We recommend that you set-up an endpoint in the same zone/region as the one you are based in. This reduces the communication 
time between your local machine and the remote one, which can significantly reduce the overall request time.

### Gradio App

The Python script [`app.py`](app/app.py) contains the code to launch a Gradio app with the Whisper large-v2 model.
By default, it uses a batch size of 16 and bfloat16 half-precision. You should update these parameters depending on your 
GPU/TPU device (as explained in the sections on [Half-precision](#half-precision) and [Batching](#batching)).

We can launch the Gradio app on port 7860 (default) on our GPU/TPU device through the following command:
```
python app/app.py
```

This will launch a Gradio demo with the same interface as the official Whisper JAX demo. To view the Gradio app remotely, 
we have two options:

1. Open the port 7860 on the GPU/TPU device to listen to all requests
2. Start an ngrok server on the GPU/TPU that redirects requests to port 7860

To open the port 7860 on your GPU/TPU, refer to your hardware provider's firewall instructions (for GCP, these can be 
found [here](https://cloud.google.com/firewall/docs/using-firewalls)). Once you have opened port 7860, you should be able 
to access the gradio demo through the http address:
```
http://DEVICE-IP:7860
```
where `DEVICE-IP` is the public IP address of your GPU/TPU. We can verify this address is accessible by opening this 
http address in a browser window on our local machine.

Alternatively, we can direct network requests to the Gradio app using ngrok. By using ngrok, we don't need to open the 
port 7860 on our GPU/TPU - ngrok will provide us with a public http address that will automatically redirect requests to 
port 7860 on our accelerator. However, in our experience, using ngrok was less reliable than a direct tunnel to port 7860, 
thus we recommend option 1 here where possible.

To set-up ngrok on your GPU/TPU, first install ngrok according to the official [installation guide](https://ngrok.com/download).
You should authenticate your ngrok account if you have one, otherwise your ngrok server will be time-limited to 2 hours.
Once installed and authenticated, you can launch an ngrok server on port 7860:
```
ngrok http 7860
```
The ngrok http address will be of the form:
```
https://NGROK-ADDRESS.ngrok.io
```
which can be used to access the Gradio demo through a web browser.

### Sending Requests

Independent of whether you've chosen to open the port 7860 or use ngrok, we're now ready to send audio file requests to our
endpoint. To do this, we'll make use of the `gradio_client` library. If you already have a recent version of Gradio, 
then the `gradio_client` library is included as a dependency.

Otherwise, the lightweight `gradio_client` package can be installed from pip and is tested to work with Python versions 
3.9 or higher:
```
pip install --upgrade gradio_client
```

We can now send json requests to our endpoint using ngrok. The function `transcribe_audio` sends an audio file to our endpoint 
and returns the transcription:

```python
from gradio_client import Client

# make sure this URL matches your http web address
API_URL = "http://DEVICE-IP:7860/" # if using port 7860
API_URL = "https://NGROK-ADDRESS.ngrok.io/" # if using ngrok

# set up the Gradio client
client = Client(API_URL)

def transcribe_audio(audio_path, task="transcribe", return_timestamps=False):
    """Function to transcribe an audio file using our endpoint"""
    text, runtime = client.predict(
        audio_path,
        task,
        return_timestamps,
        api_name="/predict_1",
    )
    return text

# transcribe an audio file using our endpoint
output = transcribe_audio("audio.mp3")

# transcribe with timestamps
output_with_timestamps = transcribe_audio("audio.mp3", return_timestamps=True)
```

## Acknowledgements

* 🤗 Hugging Face Transformers for the base Whisper implementation, particularly to [andyehrenberg](https://github.com/andyehrenberg) for the [Flax Whisper PR](https://github.com/huggingface/transformers/pull/20479) and [ArthurZucker](https://github.com/ArthurZucker) for the batching algorithm 
* Gradio for their easy-to-use package for building ML demos, and [pcuenca](https://github.com/pcuenca) for the help in hooking the demo up to the TPU 
* Google's [TPU Research Cloud (TRC)](https://sites.research.google/trc/about/) programme for Cloud TPUs
* Google's [t5x Repository](https://github.com/google-research/t5x) for the model partitioning framework

[^1]: See WER results from Colab: https://colab.research.google.com/drive/1rS1L4YSJqKUH_3YxIQHBI982zso23wor?usp=sharing
