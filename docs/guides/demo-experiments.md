# 在指定 GPU 上查看实验检查点

```bash
uv run minifrontier demo --root outputs --gpu 6 --include-experiments \
  --experiment-root strategy-single-gpu-v2 \
  --experiment-root strategy-recipe-pilots-v2 --port 7860
```

`--gpu` 使用 `nvidia-smi` 的物理卡号，并通过 GPU UUID 固定到单卡。它与 `--device` 互斥；也可以用 `--device cpu` 或设置 `CUDA_VISIBLE_DEVICES` 后指定可见的 `--device cuda:0`。更换显卡时重新启动服务，例如 `--gpu 7 --port 7861`。

`--experiment-root` 指定本轮试验目录，相对于 `--root`，可重复传入。上面的命令默认打开“本轮实验（未验收）”，只展示这两个目录下的运行；其他 acceptance 检查点放在“历史实验 / 工程验证”入口。它只控制展示范围，不移动、删除权重，也不修改训练记录。更换实验批次时调整启动参数即可；不传此参数时，“实验检查点”仍展示全部 acceptance 运行。

页面支持按模型筛选，分别保留 Muon、AdamW、reference、lower-lr 等试验，每个试验只显示最新保存的一份 `model.pt` / `checkpoint.pt`。每次刷新或生成都会重新发现新保存的权重，无需复制；未完成首次保存的运行还不能选择。详情显示完整目录、训练阶段、保存时间，以及记录中的 step 和 CE token 进度；这些是保存时的进度，不是实时训练计数。“本次实验已结束”只表示该次运行结束，不表示完整预训练或能力验收完成。生成后显示实际载入的 step 和 GPU。

预训练检查点默认采用文本续写，SFT 等后训练阶段采用对话模板；页面可手动选择。每次请求串行加载一个模型，结束后释放模型和 CUDA 缓存。默认不常驻三套权重。

实验视图允许直接观察当前能力，不会把 acceptance 运行改成正式训练或通过能力验收的模型。未加该选项时仍默认展示通过能力验收的检查点。

先通过 `nvidia-smi` 检查所选卡的空闲显存。利用率为零可能只是已有进程暂时空闲；启动 demo 不会终止这些进程。单卡即可运行，无需两卡分布式推理。
