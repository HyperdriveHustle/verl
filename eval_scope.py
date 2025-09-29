from evalscope import TaskConfig, run_task

task_config = TaskConfig(
    api_url='http://127.0.0.1:8998/v1',  # 推理服务地址
    model='Qwen3-8B',  # 模型名称（需要与部署时的模型名称一致）
    eval_type='openai_api',  # 评测类型
    datasets=['live_code_bench'],  # 数据集名称
    dataset_args={'live_code_bench': {
        'few_shot_num': 0,
        'subset_list': ['v5'],
        'extra_params': {
            'start_date': '2024-10-01',
            'end_date': '2025-02-28',
            'timeout': 30
        },
        },
    },  # 数据集参数
    eval_batch_size=256,  # 发送请求的并发数
    generation_config={
        'max_tokens': 18932,  # 最大生成token数，建议设置为较大值避免输出截断
        'temperature': 0.6,  # 采样温度（qwen 报告推荐值）
        'top_p': 0.95,  # top-p采样（qwen 报告推荐值）
        'top_k': 20,  # top-k采样（qwen 报告推荐值）
        'n': 5,  # 每个请求产生的回复数量
        # 'extra_body': {'chat_template_kwargs': {'enable_thinking': False}}
    },
    judge_worker_num=64,
    timeout=60000,  # Timeout
    # stream=True,  # Use streaming output
    # limit=100,
)

run_task(task_config)