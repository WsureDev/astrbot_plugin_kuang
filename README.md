# astrbot_plugin_kuang

AstrBot 插件“框”：识别图片、头像和 GIF 中的主体，绘制 FPS 风格白色线框。

- 发送单字 `框` 并附图，或回复带图消息发送 `框`。
- 单独发送 `框` 使用发送者头像；`@某人` 并发送 `框` 使用被提及用户头像。
- 支持多图，逐张处理后分别回图；触发前会过滤常见 Unicode 隐形控制字符。
- 默认最多保留 5 个主体框，可用 `box_count` 调整。没有可靠主体时不绘框。

## 插件图标

AstrBot 自动读取插件根目录的 `logo.png`，无需在 `metadata.yaml` 中添加图标字段。
按[官方约定](https://docs.astrbot.app/dev/star/plugin-new.html)，图标保持 1:1，推荐 256×256 像素。
仓库中的 `logo.png` 为正式插件图标；更换时直接替换同名 PNG，并随插件代码提交即可。
更新后在 AstrBot 插件页重载“框”，再刷新页面。

## 安装与模块管理

0.9.0 起，首次从 GitHub 安装只获取插件本体，不安装 NumPy、Pillow、ONNX Runtime 或 OpenCV，
也不下载模型。安装任一识别模块后才能识别画框。

在 AstrBot 的插件视图中打开“模块管理”，点击需要的能力的“安装并启用”。
页面显示依赖安装、模型下载和加载验证三个阶段，验证通过才启用；已安装模块可以调整设置和关闭。
完整页面按 [AstrBot 4.29.3 的插件视图接口](https://github.com/AstrBotDevs/AstrBot/blob/v4.29.3/docs/zh/dev/star/guides/plugin-pages.md)适配；没有插件视图入口的旧版使用下方管理员指令。

| 模块 | 用途 | 新增资源 |
| --- | --- | --- |
| 通用识别 | 人物、动物、常见物体 | YOLO26n，约 9.5 MiB |
| 前景识别 | 主体不明确、未知角色、卡通和玩偶 | U²-Net-p，约 4.4 MiB |
| 动漫识别 | 动漫人物和大头特写 | Head / Torso / Legs 模型，约 76.7 MiB |
| GIF 加速 | 光流跟踪，减少重复推理 | OpenCV，展开体积依系统而异 |

三个识别模块可单独使用或组合使用。首次安装识别模块时还需要 ONNX Runtime、NumPy 和 Pillow，
之后按需复用共享依赖。前景识别不再强制安装 OpenCV。GIF 加速是可选项；关闭时仍支持逐帧识别 GIF。
只有 GIF 加速模块、没有识别模块时，仍需要先安装识别能力。

模型下载显示真实的已下载大小、总大小和速度；服务端未提供总大小时仅显示已下载大小。
依赖安装和验证使用活动进度条及日志，不估算百分比。依赖安装复用 AstrBot 的安装器与 PyPI 镜像设置，
模型下载地址可以在插件配置中指定镜像。

安装在后台排队执行，刷新或关闭页面不影响任务。失败可重试，已安装的共享依赖和有效模型会复用。
新模型先写入临时文件并试运行，成功后再替换正式模型。关闭模块保留资源；此版本不提供共享依赖卸载。
重载插件会等待当前依赖安装步骤结束并停止后续步骤，下载会中断；重启后的未完成任务会标为“安装中断”，可手动重试。
如果原生依赖的旧版本已经被进程加载，页面会提示重启 AstrBot 后重试启用。

管理员指令（模块名支持 `通用 / 前景 / 动漫 / GIF`，也支持 `general / saliency / anime / gif`）：

```text
框模块 状态
框模块 安装 前景
框模块 启用 动漫
框模块 关闭 GIF
```

命令安装后会立即返回，发送“框模块 状态”查看阶段与下载百分比。
“启用”只验证已有资源；缺少资源时使用“安装”。不支持插件视图的版本仍可在原配置面板调整参数。

升级会复用插件数据目录和自定义路径中的已有模型，并迁移原来的识别组合和 GIF 跟踪状态。
显式关闭的动漫和前景识别保持关闭。新安装默认全部关闭；处理图片时不会触发依赖安装或模型下载。
模块进度保存在插件数据目录 `modules.json`，模型保存在 `models/`，不随插件代码更新删除。

## 主体检测

YOLO26n 负责常见目标，U²-Net-p 显著性检测负责未知角色、玩偶和其他前景。
在模块管理中按需启用；模型和依赖准备好之前不会参与处理。

动漫检测完整保留，通过模块管理安装并启用，配置开关为 `enable_anime_fallback`。模型类别为 `Head / Torso / Legs`，
可以合成人物框，也支持只出现头部的大头特写。高置信度、占画面较大的头部本身就是有效主体，
不会要求补出画面外的躯干。对小头部或局部肢体，可靠的完整前景可以补全主体；多个头部或躯干
不会被合并成一个前景框。

`fallback_policy=adaptive` 为默认策略：已有可靠人物、动物、动漫合成主体或大头特写时，
跳过不需要的补强模型。若要逐阶段评估，设置 `all`，运行全部已启用的阶段。
`enable_motion` 保留为 GIF 加速模块内的可选运动候选，默认关闭；GIF 光流跟踪不依赖此开关。
显著性分数表示前景面积占比，与 YOLO 的分类置信度含义不同，不应混合平均作为准确率。

## GIF 与性能

启用 GIF 加速后，默认每 4 帧或累计 240 毫秒执行一次完整检测，任一上限达到就重检。
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

## 本地开发与验证

以下依赖清单仅用于本地开发和独立评估脚本，不属于插件首次安装流程。
`requirements/runtime.txt` 包含识别共享运行库，`requirements/gif.txt` 包含可选 GIF 加速依赖。

```bash
pip install -r requirements/dev.txt
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
