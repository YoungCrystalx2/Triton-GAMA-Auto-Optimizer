# Triton GA Auto Optimizer
**团队成员**  
杨晶晶、刘艺璇、付嘉瑞  
## 项目背景
随着大模型推理与训练的算力需求持续激增，Triton 已成为高性能计算内核开发的主流框架，算子执行效率直接决定端到端模型的吞吐与延迟。当前 Triton 算子优化存在两条典型路径，但均存在明显短板：
- 人工调优：
  - 优化空间庞大：仅矩阵乘法（matmul）就存在数百种 block size、循环分块、内存访问模式的组合，人工穷举几乎不可能；
  - 硬件适配复杂：不同架构的 CPU（如 x86、ARM）对内存布局、向量化指令的适配要求差异显著，通用优化策略效果有限。
- 纯 LLM 自动优化：依靠大模型直接生成优化代码，虽提升效率，但缺乏全局搜索能力，易陷入局部最优；优化行为无策略引导，代码正确性、稳定性与性能一致性无法保证，难以落地生产环境。

## 项目简介
本项目实现了一个基于**遗传算法 (Genetic Algorithm)** 和**多智能体协作 (Multi-Agent System)** 的Triton算子自动优化框架。通过将遗传算法的全局搜索能力与大语言模型的代码生成能力相结合，实现了对Triton CPU算子的自动化性能优化。

### 核心创新点

1. **遗传算法驱动的搜索**：使用标准GA流程（选择-交叉-变异-评估）在庞大的优化空间中高效搜索
2. **多智能体专业化变异**：4个专业智能体针对不同优化维度进行精细化变异
3. **三级级联评估机制**：确保代码质量的同时提高评估效率
4. **硬件感知适应度函数**：综合考虑正确性、性能和硬件特征

### 支持的算子类型
- **矩阵乘法 (matmul)**：重点优化对象，支持多种block size配置

## 项目架构

```
triton/
├── main.py                         # 主程序入口，参数解析和优化流程控制
├── evolutionary_optimizer.py       # 遗传算法核心实现
├── evaluator.py                    # 评估器（级联评估,性能测试,适应度计算）
├── llm_client.py                   # LLM客户端（支持Qwen等）
├── config.py                       # 配置管理
├── example_config.json             # 配置示例
├── run_matmul_optimization.sh      # matmul优化一键脚本
├── agent/                          # 多智能体变异系统
│   ├── base_agent.py               # 智能体基类
│   ├── loop_agent.py               # 循环优化智能体
│   ├── memory_agent.py             # 内存访问优化智能体
│   ├── vector_agent.py             # 向量化优化智能体
│   └── schedule_agent.py           # 调度优化智能体
└── triton-cpu-main/
    └── triton-cpu-scripts/
        ├── matmul.py               # 基准matmul实现
        └── rmsnorm.py              # 基准RMSNorm实现
```

## 核心算法流程

### 1. 遗传算法框架

项目实现了完整的遗传算法流程，专门针对Triton算子优化定制：

#### 初始化阶段
- 以基准代码为基础
- 使用LLM生成初始种群变体
- 确保种群多样性

#### 进化迭代
- **选择 (Selection)**：锦标赛选择算法，选择适应度最高的个体
- **交叉 (Crossover)**：LLM驱动的代码语义融合，结合两个父代的优化特征
- **变异 (Mutation)**：多智能体随机变异，针对特定优化方向进行调整
- **评估 (Evaluation)**：三级级联评估，计算适应度分数
- **更新 (Update)**：精英保留策略，保证最优个体不丢失

#### 终止条件
- 达到最大迭代次数
- 超过时间或Token预算
- 适应度连续5代无改善（早停机制）

### 2. 多智能体变异系统

#### 2.1 智能体架构
项目设计了4个专业化智能体，每个智能体专注于特定的优化维度：

| 智能体名称 | 优化方向 | 核心能力 |
|-----------|---------|----------|
| **LoopAgent** | 循环优化 | 循环分块、展开、顺序重排 |
| **MemoryAgent** | 内存优化 | 缓存友好访问模式、减少访存延迟 |
| **VectorAgent** | 向量化优化 | SIMD指令适配、数据并行优化 |
| **ScheduleAgent** | 调度优化 | 并行任务分配、执行顺序优化 |

#### 2.2 协作机制
- **触发时机**：仅在GA的变异阶段随机调用
- **角色扮演**：为LLM提供专业化提示词，引导生成针对性优化
- **约束保证**：保持函数签名和算法逻辑不变，仅进行性能优化

### 3. 评估机制与适应度

#### 3.1 三级级联评估层次设计
**Level 1: 静态语法检查**  
   - Python语法编译验证
   - 过滤明显语法错误的代码

**Level 2: 快速功能验证**  
   - 使用简化参数进行功能测试
   - 快速验证基本正确性

**Level 3: 完整性能评估**  
   - 全参数范围的功能和性能测试
   - 计算准确的性能指标

#### 3.2 多维适应度函数
```
fitness = 0.4 × correctness + 0.4 × speedup + 0.2 × hardware_score
```

- **正确性 (40%)**：功能测试通过率
- **加速比 (40%)**：相对基准实现的性能提升
- **硬件得分 (20%)**：缓存友好性、内存对齐等硬件感知指标

## 安装与使用

### 1. 环境准备
请确保：  
- Python 3.9+
- 已安装 `torch`、`triton`
- 若用 Qwen：`pip install dashscope`
- 已设置 API Key

### 2. 运行优化

#### 2.1 基本用法

```bash
python main.py \
    --kernel-type matmul \
    --llm-provider qwen \
    --llm-model qwen3-coder-plus \
    --population-size 8 \
    --max-iterations 15 \
    --max-time 1800 \
    --max-tokens 200000 \
    --output-dir results/matmul_opt
```

- 参数说明
| 参数 | 说明 | 默认值 |
|------|------|--------|
| --kernel-type | 算子类型 (matmul/vector_add/rmsnorm) | matmul |
| --llm-provider | LLM提供商 | qwen |
| --population-size | 种群大小 | 5 |
| --max-iterations | 最大迭代次数 | 10 |
| --max-time | 时间限制(秒) | 1200 |
| --max-tokens | Token预算 | 200000 |
| --output-dir | 输出目录 | optimized_kernels |

#### 2.2 一键运行脚本

```bash
# 直接运行matmul优化
bash run_matmul_optimization.sh
```

### 3. 生成文件结构

```
optimized_kernels/
├── kernel_1.py          # 最优算子代码
├── kernel_2.py          # 次优算子代码
├── summary.json         # 优化统计数据(包含：每个版本的功能正确性;性能加速比;Token使用量)
```

## 技术特点

### 优势

1. **全局搜索能力**：GA框架避免陷入局部最优
2. **语义级优化**：LLM理解代码语义，进行智能修改
3. **专业化优化**：多智能体针对不同维度进行专项优化
4. **质量保证**：三级评估确保生成的代码功能正确
5. **资源控制**：时间和Token限制保证实用性

### 局限性

1. **依赖LLM质量**：优化效果受LLM代码生成能力影响
2. **评估开销**：三级评估增加总优化时间
3. **参数调优**：需要根据具体硬件调整GA参数
4. **扩展性**：添加新算子需要实现对应的评估器

## 参考文献

1. Guo, Q., et al. "Connecting large language models with evolutionary algorithms yields powerful prompt optimizers." arXiv preprint arXiv:2309.08532 (2023).

2. Novikov, A., et al. "AlphaEvolve: A coding agent for scientific and algorithmic discovery." arXiv preprint arXiv:2506.13131 (2025).

3. Lange, R. T., Imajuku, Y., & Cetin, E. "Shinkaevolve: Towards open-ended and sample-efficient program evolution." arXiv preprint arXiv:2509.19349 (2025).

4. Lange, R. T., et al. "Towards Robust Agentic CUDA Kernel Benchmarking, Verification, and Optimization." arXiv preprint arXiv:2509.14279 (2025).
