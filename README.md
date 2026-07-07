# astrbot_plugin_kuang

AstrBot 插件“框”。

触发方式：

- 发送单字 `框` 并附带图片
- 回复带图消息发送单字 `框`
- 单独发送单字 `框`，此时自动使用发送者头像
- `@某人` 并发送单字 `框`，此时自动使用被 `@` 用户头像

兼容性：

- 会先过滤常见 Unicode 双向/隐形控制字符，再按原有 `框` 触发规则判断

效果说明：

- 每张图固定绘制 5 个白色线框
- 优先使用 YOLO26n ONNX 识别框
- 可选启用“二次元检测兜底”，在主检测的人形结果较少时用动漫部位 YOLO 模型补强
- 默认在识别不足 5 个时自动补随机框，可在配置面板关闭
- 随机框按 FPS 外挂透视框风格生成，默认使用偏细长的矩形比例
- 支持 GIF，并对每一帧重新识别

当前实现策略：

- 人体优先级最高，常见动物次之，其他目标按置信度补充
- 二次元补强使用第二个可配置 ONNX 模型，当前默认模型可识别 `Head / Torso / Legs`
- 当 `enable_anime_fallback=true` 且本地缺少 `anime_yolo.onnx` 时，会按默认 Hugging Face 地址自动下载
- 首次运行若缺少 `yolo26n.onnx`，会自动下载到插件数据目录
- GIF 仅使用常规 256 色保存，不再做激进降色压缩
- 多张图会逐张处理并分别回图

依赖安装：

```bash
pip install -r requirements.txt
```

随机框分布审阅示例：

```bash
python examples/generate_random_box_samples.py
```

默认会在 `sample_output/random_box_samples/` 下生成 20 张 `300x300` 透视分布预览图。
