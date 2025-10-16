import random
from typing import Optional

from pydantic import Field
from httpx import AsyncClient, Timeout, URL

from nekro_agent.api import core
from nekro_agent.api.schemas import AgentCtx
from nekro_agent.api.plugin import NekroPlugin, ConfigBase, SandboxMethodType

from nonebot import get_bot
from nonebot.adapters.onebot.v11 import MessageSegment, ActionFailed


# 创建插件实例
plugin = NekroPlugin(
    name="二游语音生成插件",
    module_name="anime_tts",
    description="AI 自主选择语音模型使用 TTS 生成语音并直接发送语音条",
    version="0.1.1",
    author="Jerry_FaGe",
    url="https://github.com/Jerry-FaGe/nekro-plugin-anime-tts",
)

# 添加配置类
@plugin.mount_config()
class TtsMsgConfig(ConfigBase):
    """插件配置"""
    TTS_API_URL: str = Field(
        default="https://gsv2p.acgnai.top",
        title="TTS API URL",
        description="TTS 服务的基础 URL。请前往 <a href='https://gsv.acgnai.top' target='_blank'>AI Hobbyist TTS</a> 注册。",
    )
    TTS_API_TOKEN: str = Field(
        default="None",
        title="TTS API TOKEN",
        description="注册登录后在上述网站右上角用户名处获取访问令牌",
    )

# 获取配置实例
config = plugin.get_config(TtsMsgConfig)

# 常量定义
TTS_API_URL = URL(config.TTS_API_URL)
TTS_API_TOKEN = config.TTS_API_TOKEN

HEADERS = {"Content-Type": "application/json", "Accept": "application/json", "Authorization": f"Bearer {TTS_API_TOKEN}",}
TIMEOUT = Timeout(read=60, write=60, connect=10, pool=10)
CLIENT = AsyncClient(timeout=TIMEOUT)


async def _make_request(method: str, url: str, json: Optional[dict] = None) -> dict:
    """通用请求函数"""
    response = await CLIENT.request(
        method=method,
        url=TTS_API_URL.join(url),
        headers=HEADERS,
        json=json,
        timeout=TIMEOUT,
    )
    response.raise_for_status()
    return response.json()


@plugin.mount_sandbox_method(
    SandboxMethodType.AGENT,
    name="获取语音模型",
    description="获取所有生成语音可用的模型",
)
async def get_tts_model(_ctx: AgentCtx) -> str:
    """获取所有生成语音可用的模型"""
    data = await _make_request("POST", "models/v4")
    data = data.get("models")  # dict[str, dict[str, list[str]]] | None
    return f"[get_tts_model Results]\n{data}\n这是语音生成接口可用的全部模型，键为模型名，值为该模型可用的语言字典，语言字典内为语气列表，请根据用户要求从中选择一个最合适的。"


@plugin.mount_sandbox_method(
    SandboxMethodType.TOOL,
    name="生成语音",
    description="根据传入文本，模型，语言，语气生成语音",
)
async def generate_voice(_ctx: AgentCtx, text: str, model_name: str, language: str, emotion: str) -> str:
    """根据传入文本，语音模型，语言，语气生成一段音频的 URL

    可选角色大多是《崩坏3》《原神》《星穹铁道》《鸣潮》《明日方舟》《蔚蓝档案》《妮姬》中的人物。
    
    **重要提示：** 请务必使用**语音模型角色**的语气和人设来构思文本。例如，如果选择爱莉希雅的语音模型，请使用符合爱莉希雅性格和背景的措辞，避免使用大模型自己的人设或其他角色的口头禅或表达方式。

    Args:
        text: 要生成语音的文本，大部分模型支持中文，少部分支持日语、英语等
        model_name: 模型，务必先通过 get_tts_model 方法获取可用模型后填入
        language: 语言，通过 get_tts_model 方法获取的模型字典的值为模型支持的语言字典，请填入语言字典的键名
        emotion: 语气，语言字典的值为支持的语气列表，请填入合适的语气

    Returns:
        str: 生成音频的 URL，后缀名为 `.wav`

    Example:
        generate_voice("愿你前行的道路有群星闪耀，愿你留下的足迹有百花绽放。你即是上帝的馈赠，世界因你而瑰丽。", "崩环三-中文-爱莉希雅", "中文", "默认")
    """
    payload = {
        "version": "v4",
        "model_name": model_name,
        "prompt_text_lang": language,
        "emotion": emotion,
        "text": text,
        "text_lang": language,
        "top_k": 10,
        "top_p": 1,
        "temperature": 1,
        "text_split_method": "按标点符号切",
        "batch_size": 1,
        "batch_threshold": 0.75,
        "split_bucket": True,
        "speed_facter": 1,
        "fragment_interval": 0.3,
        "media_type": "wav",
        "parallel_infer": True,
        "repetition_penalty": 1.35,
        "seed": random.randint(0, 999999999),
        "sample_steps": 16,
        "if_sr": False,
    }
    data = await _make_request("POST", "infer_single", json=payload)

    if data.get("msg") == "参数错误":
        core.logger.error(f"TTS API 参数错误: 模型: {model_name}, 语言: {language}, 语气: {emotion}")
        raise Exception(f"参数错误: 模型: {model_name}, 语言: {language}, 语气: {emotion}")

    if data.get("msg") == "合成成功":
        core.logger.info(f"TTS API 参数: 模型: {model_name}, 语言: {language}, 语气: {emotion}")
        core.logger.info(f"TTS API 文本: {text}")
        return data["audio_url"]
    
    raise Exception(f"出现未知错误: {str(data)}，请检查参数是否正确: 模型: {model_name}, 语言: {language}, 语气: {emotion}")


@plugin.mount_sandbox_method(
    SandboxMethodType.TOOL,
    name="发送语音消息",
    description="发送语音消息",
)
async def send_record_msg(_ctx: AgentCtx, chat_key: str, voice_path: str):
    """发送语音消息

    Args:
        chat_key (str): 会话标识
        voice_path (str): 语音文件路径或 URL
    """
    try:
        bot = get_bot()
        voice_message = MessageSegment.record(file=voice_path)
        
        if "_" not in chat_key:
            raise ValueError(f"无效的 chat_key 格式: {chat_key}")
        
        adapter_id, old_chat_key = chat_key.split("-", 1)
        
        chat_type, target_id = old_chat_key.split("_", 1)
        
        if not target_id.isdigit():
            raise ValueError(f"目标ID必须为数字: {target_id}")
        
        if chat_type == "private":
            await bot.call_api(
                "send_private_msg",
                user_id=int(target_id),
                message=voice_message
            )
            core.logger.success(f"私聊语音发送成功: QQ={target_id}, voice={voice_path}")
            
        elif chat_type == "group":
            await bot.call_api(
                "send_group_msg",
                group_id=int(target_id),
                message=voice_message
            )
            core.logger.success(f"群聊语音发送成功: 群={target_id}, voice={voice_path}")
            
        else:
            raise ValueError(f"不支持的聊天类型: {chat_type}")
        
    except ActionFailed as e:
        core.logger.error(f"API调用失败: {e.info.get('msg', '未知错误')}")
    except ValueError as e:
        core.logger.error(f"参数错误: {str(e)}")
    except Exception as e:
        core.logger.exception(f"发送语音消息异常: {str(e)}")


@plugin.mount_cleanup_method()
async def clean_up():
    """清理插件资源"""
    # 如有必要，在此实现清理资源的逻辑
    pass
