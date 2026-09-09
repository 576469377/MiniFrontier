# 在指定 GPU 上查看实验检查点

下面以物理 GPU 0 为例。`--root` 指向你保存训练结果的目录；需要先有本地检查点。

```bash
uv run minifrontier demo --root outputs --gpu 0 --include-experiments --port 7860
```

`--gpu` 使用 `nvidia-smi` 的物理卡号。它与 `--device` 互斥；没有 GPU 时可以改用 `--device cpu`。已经设置 `CUDA_VISIBLE_DEVICES` 时，也可以使用可见设备编号 `--device cuda:0`。

如需只观察某几组试验，可追加 `--experiment-root <相对目录>`，此参数可重复传入。例如检查点位于 `outputs/my-experiment/` 时使用 `--experiment-root my-experiment`。页面会将选中的试验与其他历史试验分开；不传此参数则展示全部实验检查点。

页面支持按模型筛选，分别保留 Muon、AdamW、reference、lower-lr 等试验，每个试验只显示最新保存的一份 `model.pt` / `checkpoint.pt`。每次刷新或生成都会重新发现新保存的权重，无需复制；未完成首次保存的运行还不能选择。详情显示完整目录、训练阶段、保存时间，以及记录中的 step 和 CE token 进度；这些是保存时的进度，不是实时训练计数。“本次实验已结束”只表示该次运行结束，不表示完整预训练或能力验收完成。生成后显示实际载入的 step 和 GPU。

预训练检查点默认采用文本续写，SFT 等后训练阶段采用对话模板；页面可手动选择。每次请求串行加载一个模型，结束后释放模型和 CUDA 缓存。默认不常驻三套权重。

实验视图允许直接观察当前能力，不会把 acceptance 运行改成正式训练或通过能力验收的模型。未加该选项时仍默认展示通过能力验收的检查点。

先通过 `nvidia-smi` 检查所选卡的空闲显存。利用率为零可能只是已有进程暂时空闲；启动 demo 不会终止这些进程。单卡即可运行，无需两卡分布式推理。
