# astrbot_plugin_kuang

AstrBot 插件“框”：识别图片、头像和 GIF 中的主体，绘制 FPS 风格白色线框。

- 发送单字 `框` 并附图，或回复带图消息发送 `框`。
- 单独发送 `框` 使用发送者头像；`@某人` 并发送 `框` 使用被提及用户头像。
- 支持多图，逐张处理后分别回图；触发前会过滤常见 Unicode 隐形控制字符。
- 默认最多保留 5 个主体框，可用 `box_count` 调整。没有可靠主体时不绘框。

## 主体检测

YOLO26n 负责常见目标，U²-Net-p 显著性检测负责未知角色、玩偶和其他前景。
显著性默认开启；安装后首次使用会按配置下载 ONNX 模型到插件数据目录。
升级时已有的显式配置会保留，原来关闭显著性的实例需要在面板开启 `enable_saliency`。

动漫检测完整保留，通过 `enable_anime_fallback` 开启。模型类别为 `Head / Torso / Legs`，
可以合成人物框，也支持只出现头部的大头特写。高置信度、占画面较大的头部本身就是有效主体，
不会要求补出画面外的躯干。对小头部或局部肢体，可靠的完整前景可以补全主体；多个头部或躯干
不会被合并成一个前景框。

`fallback_policy=adaptive` 为默认策略：已有可靠人物、动物、动漫合成主体或大头特写时，
跳过不需要的补强模型。若要逐阶段评估，设置 `all`，运行全部已启用的阶段。
`enable_motion` 保留为可选运动候选，默认关闭；GIF 光流跟踪不依赖此开关。
显著性分数表示前景面积占比，与 YOLO 的分类置信度含义不同，不应混合平均作为准确率。

## GIF 与性能

默认每 4 帧或累计 240 毫秒执行一次完整检测，任一上限达到就重检。
中间帧通过 OpenCV 金字塔光流跟踪主体；跟踪点不足、外观变化较大、出现明显新区域时立即重检。
跟踪始终参考上次实际检测的帧，不累计逐帧偏移。每张 GIF 使用独立状态。

| 配置 | 默认值 | 用途 |
| --- | --- | --- |
| `gif_detection_interval` | `4` | 完整检测间隔，设为 `1` 可逐帧检测 |
| `gif_max_reuse_ms` | `240` | 按 GIF 原始帧时长累计的复用上限，单位毫秒 |
| `max_concurrent_images` | `1` | 插件同时处理图片的任务数量 |
| `inference_num_threads` | `0` | YOLO 的 ORT 算子线程数，0 使用运行时默认 |
| `saliency_intra_op_num_threads` | `0` | 显著性 ORT 算子线程数 |
| `saliency_max_foreground_ratio` | `0.995` | 允许接近满屏的特写，过滤几乎全白的 mask |

光流主要估计平移，明显形变、快速缩放、遮挡时通过重检恢复。多人画面分别跟踪，任一主体失效
都会重新检测。更高的检测间隔会增加短时间遗漏新主体的可能性，需要按素材测试。
原有 `saliency_reuse_static_frames` 已由 GIF 任务级跟踪设置替代。

当前使用 CPUExecutionProvider。线程数建议在实际部署机器上对比 1、2、4 和默认值。
固定尺寸 ONNX 会优先使用模型输入尺寸，修改配置不会改变模型本身。

## 安装与验证

```bash
pip install -r requirements.txt
python -m unittest discover -s tests -v
python examples/render_image.py path/to/image.gif --anime
python examples/render_stages.py path/to/image.gif
```

使用已有本地模型对比全阶段、按需补强和 GIF 跟踪三种策略：

```bash
python examples/benchmark_pipeline.py path/to/image.gif --anime --render
```

模型默认从 `sample_output/models/` 读取，可通过 `--model-dir` 指定。评估工具不自动下载模型。
输出 `report.json`，记录模型 SHA-256、运行环境、预热后耗时、调用次数、空帧和逐帧框；绘图编码
不计入检测耗时。`--repeat` 默认 3。单张图或一个 GIF 的结果不能替代完整样本集评估。

可用 JSON 标注清单测主体召回率与精确率：

```json
{
  "samples": [
    {
      "path": "input.gif",
      "boxes": {
        "0": [[10, 20, 100, 180]],
        "1": []
      }
    }
  ]
}
```

```bash
python examples/benchmark_pipeline.py --manifest samples.json --anime --render
```

路径相对清单文件解析，帧号从 0 开始，坐标为原图像素 `[x1,y1,x2,y2]`。
只有明确标注的帧参与质量统计，空数组表示该帧无目标，匹配阈值为 IoU ≥ 0.5。
无标注时质量指标为空，空帧数不能当作准确率。

## 开发基线

本轮优化前的代码与阶段实验保存在 `archive/baseline-20261028`，提交 `afdf11c`。
当前实现分支为 `feat/subject-detection-performance`。
