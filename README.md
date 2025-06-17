```shell
git clone --recurse-submodules git@gitlab.alibaba-inc.com:EconML/BeyondAgent.git
```


# Explorer Dataflow Rollout Example (GSM8K) for BeyondAgent

This example demonstrates how to perform **dataflow rollout** using ParallelEnvManager with AsyncLLMServerManager of multipe model server (e.g., Qwen2.5-3B) on the GSM8K dataset.

### 这种实现方式：

1. 目录结构: 对verl目录下的代码不做任何改动，所有与beyondagent的代码存放在recipe/beyond_agent目录下。
2. Trainer继承: 继承verl中RayTrainer类，对部分函数进行修改
3. ParallelEnvManager类：在Trainer中引入ParallelEnvManager类，可通过线程池，并行执行多个dataflow对象（为每个prompt创建一个dataflow对象），并对输出结果进行聚合（upcoming）。
4. AsyncLLMServerManager类：在ParallelEnvManager中使用verl中的LLMServerManager类，所有dataflow对象共用同一个LLMServerManager，由LLMServerManager同时管理多个vLLM server, 通过ChatScheduler对来自各个线程中dataflow的llm-call进行分发和等待。

## Usage

### Step 1: Download GSM8K Dataset

```bash
cd examples/data_preprocess
python3 gsm8k_multiturn_w_tool.py
```

This will download and preprocess the GSM8K dataset into ~/data/gsm8k/.

### Step 2: Run Multi-Turn Rollout

If you have 2 GPUs
Use the standard 2-GPU script:

```bash
cd your_verl_root_dir
bash recipe/beyond_agent/run_qwen2.5-3b_dataflow_2gpu.sh
```

## Notes

- The rollout supports multi-turn conversations with tool-calling capabilities.
- Current tools are used for GSM8K answer evaluation.
- Future versions may extend to search and code interpreter tools.
