#!/bin/bash
# 使用 Qwen3-Coder-Plus 优化 matmul.py 中的 Triton 矩阵乘法算子
# 输出优化后算子 vs 原始 Triton 算子 vs PyTorch 参考实现的性能对比

# 确保已设置 API 密钥环境变量
# export DASHSCOPE_API_KEY="your-api-key"

echo "============================================================"
echo "使用 Qwen3-Coder-Plus 优化 Triton 矩阵乘法算子"
echo "============================================================"

# 检查 API 密钥
if [ -z "$DASHSCOPE_API_KEY" ] && [ -z "$QWEN_API_KEY" ]; then
    echo "错误: 未设置 DASHSCOPE_API_KEY 或 QWEN_API_KEY 环境变量"
    echo "请运行: export DASHSCOPE_API_KEY='your-api-key'"
    exit 1
fi

# 检查 matmul.py 文件是否存在
MATMUL_FILE="triton-cpu-main/triton-cpu-scripts/matmul.py"
if [ ! -f "$MATMUL_FILE" ]; then
    echo "警告: 未找到 $MATMUL_FILE，将使用默认 baseline"
    MATMUL_FILE=""
else
    echo "找到 kernel 文件: $MATMUL_FILE"
fi

# 运行优化
echo ""
echo "开始优化..."
echo ""

python main.py \
    --kernel-type matmul \
    --llm-provider qwen \
    --llm-model qwen3-coder-plus \
    --population-size 5 \
    --max-iterations 10 \
    --max-time 1200 \
    --max-tokens 200000 \
    --output-dir results/matmul_optimization \
    ${MATMUL_FILE:+--kernel-file "$MATMUL_FILE"}

EXIT_CODE=$?

echo ""
echo "============================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "优化完成！"
    echo "结果保存在: results/matmul_optimization/"
    echo ""
    echo "查看性能对比:"
    echo "  cat results/matmul_optimization/summary.json"
else
    echo "优化过程中出现错误 (退出码: $EXIT_CODE)"
    echo "请检查错误信息"
fi
echo "============================================================"

exit $EXIT_CODE
