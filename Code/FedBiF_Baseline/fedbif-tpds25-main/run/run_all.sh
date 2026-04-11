
SCRIPTS=(
    "fedbif.sh"
    "fedbif1.sh"
    "fedbif2.sh"
    "fedbif3.sh"
    "fedbif_femnist.sh"
    "fedbif_shakespeare.sh"
)

echo "🚀 启动自动化批处理实验流水线"



for script in "${SCRIPTS[@]}"; do
    echo ""
    echo "---------------------------------------------------"
    echo "▶️  正在运行: $script"
    echo "🕒 开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "---------------------------------------------------"


    bash "$script"

    status=$?
    
    if [ $status -eq 0 ]; then
        echo "✅ $script 完美执行完成！"
    else
        echo "⚠️ $script 执行结束，返回了非零状态码 ($status)。"
        echo "💡 提示: 联邦学习脚本末尾的 gRPC 断连报错属于正常现象，流水线将自动继续..."
    fi
    
    echo "🕒 结束时间: $(date '+%Y-%m-%d %H:%M:%S')"

    sleep 10
done

echo ""
echo "🎉 恭喜！所有 6 个实验脚本已全部执行完毕！"
