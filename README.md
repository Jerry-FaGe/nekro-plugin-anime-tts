# Nekro Plugin - Anime TTS (二游语音生成插件)

<div align="center">

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![Nekro Agent](https://img.shields.io/badge/Nekro_Agent-%3E%3D2.0.0-orange)](https://github.com/KroMiose/nekro-agent)
[![License](https://img.shields.io/github/license/Jerry-FaGe/nekro-plugin-anime-tts)](https://github.com/Jerry-FaGe/nekro-plugin-anime-tts/blob/main/LICENSE)
[![GitHub Repo stars](https://img.shields.io/github/stars/Jerry-FaGe/nekro-plugin-anime-tts?style=social)](https://github.com/Jerry-FaGe/nekro-plugin-anime-tts)

**一个为 [NekroAgent](https://github.com/KroMiose/nekro-agent) 设计的 TTS (文本转语音) 插件，允许 AI 代理自主选择二次元游戏角色的语音模型，生成并发送语音消息。**

</div>

---

## ✨ 功能特性

- **🤖 AI 自主调用**: AI 代理可以自主查询可用模型、生成语音并发送，实现完整的自动化流程。
- **🎤 丰富的角色语音**: 对接 [AI Hobbyist TTS](https://gsv.acgnai.top/) 服务，支持大量二次元游戏角色语音，如《崩坏3》、《原神》、《星穹铁道》、《鸣潮》、《明日方舟》等。
- **🔊 直接发送语音条**: 生成的语音可以直接通过 OneBot v11 适配器以语音条的形式发送到私聊或群聊。
- **⚙️ 高度可配置**: 核心服务地址和 API Token 均可通过插件配置页面轻松设置。

## 🚀 快速开始

### 1. 前置要求

- 你需要一个正在运行的 **NekroAgent** 实例。
- 插件通过 **OneBot v11** 协议发送消息，请确保相关适配器已正确安装和配置。

### 2. 安装插件

在 NekroAgent 的插件市场中搜索 `anime_tts` 或 `二游语音生成插件` 进行安装。

### 3. 获取并配置 Token

1.  访问 [AI Hobbyist TTS](https://gsv.acgnai.top/) 网站并注册一个账户。
2.  登录后，在右上角用户名处获取你的访问令牌。
3.  在 NekroAgent 的插件配置页面中，找到本插件的配置项：
    - `TTS_API_URL`: 保持默认的 `https://gsv2p.acgnai.top` 即可，除非服务地址有变动。
    - `TTS_API_TOKEN`: 填入你在上一步获取到的访问令牌。

## 🔧 使用说明

配置完成后，AI 代理即可调用本插件提供的工具。一个典型的使用流程如下：

1.  **(可选) 查询可用模型**: AI 可以调用 `获取语音模型` 来了解当前所有可用的角色、语言和语气。
2.  **生成语音**: AI 根据需求（如用户指令或自身逻辑）构思文本，并调用 `生成语音` 工具，选择合适的模型、语言和语气来生成语音文件的 URL。
3.  **发送语音**: AI 调用 `发送语音消息` 工具，将上一步获取到的语音 URL 发送到指定的聊天会话中。

**给 AI 的指令示例:**

> "请用爱莉希雅的声音祝我生日快乐。"

## 🛠️ 函数列表

本插件向 AI 代理暴露了以下三个工具函数：

| 函数名 | 类型 | 描述 |
| :--- | :--- | :--- |
| `获取语音模型` | Agent | 获取所有可用的语音模型、支持的语言和语气。 |
| `生成语音` | Tool | 根据指定的文本、模型、语言和语气生成一段 `.wav` 格式的语音，并返回其 URL。 |
| `发送语音消息` | Tool | 将一个语音文件（URL 或本地路径）作为语音条发送到指定的会话。 |

## 💖 鸣谢

- GPT-SoVITS开发者：[@花儿不哭](https://space.bilibili.com/5760446)

- 模型训练者：[@红血球AE3803](https://space.bilibili.com/6589795) [@白菜工厂1145号员工](https://space.bilibili.com/518098961)

- 推理特化包适配 & 在线推理：[@AI-Hobbyist](https://space.bilibili.com/1918820)

## 📄 许可证

本项目使用 [LGPL-3.0 License](https://www.gnu.org/licenses/lgpl-3.0) 授权。
