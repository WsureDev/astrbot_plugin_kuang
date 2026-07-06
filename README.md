# astrbot_plugin_kuang

AstrBot 插件“框”。

触发方式：

- 发送单字 `框` 并附带图片
- 回复带图消息发送单字 `框`
- 单独发送单字 `框`，此时自动使用发送者头像

效果说明：

- 每张图固定绘制 5 个白色线框
- 优先使用 YOLO26n ONNX 识别框
- 识别不足 5 个时自动补随机框
- 随机框按 FPS 外挂透视框风格生成，默认使用偏细长的矩形比例
- 支持 GIF，按 0.5 秒检测一次，其余帧沿用上次识别结果

当前实现策略：

- 人体优先级最高，常见动物次之，其他目标按置信度补充
- 首次运行若缺少 `yolo26n.onnx`，会自动下载到插件数据目录
- GIF 输出会尝试逐级压缩调色板，尽量不让体积继续膨胀
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
