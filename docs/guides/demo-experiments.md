# 查看来源模型的实验检查点

三个来源模型共用实验浏览器；MF1 使用[独立媒体 Demo](minifrontier1.md#导出和诊断-demo)。先准备本地检查点，再启动：

```bash
uv run minifrontier demo --root outputs --gpu 0 --include-experiments --port 7860
```

浏览器打开 `http://127.0.0.1:7860`。`--root` 是训练输出根目录，`--include-experiments` 开放未通过能力验收的实验检查点；默认视图只展示通过验收的权重。

| 设备选择 | 参数 |
|---|---|
| 按 `nvidia-smi` 的物理卡号选择 | `--gpu 0` |
| 按 `CUDA_VISIBLE_DEVICES` 映射后的可见编号选择 | `--device cuda:0` |
| CPU | `--device cpu` |

`--gpu` 与 `--device` 互斥。Demo 单卡运行；选择 GPU 前需确认显存余量。

用 `--experiment-root my-experiment` 可将 `outputs/my-experiment/` 列为本次试验，其余归入历史试验；参数可重复。不指定时展示全部实验检查点。

页面按模型筛选，每个试验显示最新保存的 `model.pt` 或 `checkpoint.pt`，刷新和生成时重新发现权重。尚未首次保存的运行不会出现。详情中的阶段、step、CE token 和时间来自保存时的记录；“本次实验已结束”只表示该次运行结束。生成后另显示实际载入的 step 和 GPU。

预训练检查点默认使用文本续写，SFT 等阶段默认使用对话模板，也可手动切换。请求串行处理，每次加载一个模型，结束后释放权重和 CUDA 缓存。
