#!/bin/bash

# 注意：
# 请先执行文件：EnvService/env_sandbox/environments/bfcl/bfcl_dataprocess.py
# 获取BFCL_DATA_PATH与BFCL_SPLID_ID_PATH，并对应设置以上两个变量

# 环境变量（请改成实际路径）


export ENV_PATH=/mnt/data/yunpeng.zyp/code/beyondagent/workspace/EnvService
export BFCL_DATA_PATH=$ENV_PATH/bfcl/multiturn_dataset/multi_turn_base_processed.jsonl
export BFCL_SPLID_ID_PATH=$ENV_PATH/bfcl/multiturn_dataset/multi_turn_base_split_ids.json
export BFCL_ANSWER_PATH=$ENV_PATH/bfcl/data/possible_answer
export OPENAI_API_KEY=xx

# only for multinode running
export RAY_ENV_NAME=bfcl 

# 获取脚本所在目录的绝对路径
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# 导航到项目根目录 (envservice)
PROJECT_ROOT="$SCRIPT_DIR/../../"
cd "$PROJECT_ROOT"

# 设置 PYTHONPATH
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# 打印当前工作目录和 PYTHONPATH 以进行调试
echo "Current working directory: $(pwd)"
echo "PYTHONPATH: $PYTHONPATH"

# 运行 Python 命令
exec python -m envservice.env_service --env bfcl --portal 127.0.0.1 --port 8000