import ast
import copy
import json
import zlib
import html
import inspect
import logging
import os
import os.path
import time
import threading
import traceback
import re
import base64
from dataclasses import dataclass
from enum import Enum
from html.parser import HTMLParser
from struct import unpack
from typing import List, Callable, Optional, Any, Union, Dict, Tuple, get_origin, get_args
from urllib.parse import urlencode, parse_qs, urlparse
from ui.bulletin import BulletinHelper as _BulletinHelper # type: ignore
from ui.settings import Header, Switch, Input, Text, Divider # type: ignore
from ui.alert import AlertDialogBuilder # type: ignore
from base_plugin import BasePlugin, HookResult, MethodHook, HookStrategy # type: ignore
from android_utils import log as _log, run_on_ui_thread # type: ignore
from client_utils import get_messages_controller, get_last_fragment, send_request, send_message, get_file_loader, run_on_queue # type: ignore
from java import dynamic_proxy, jclass # type: ignore
from java.util import Locale, ArrayList # type: ignore
from java.lang import Long, Integer, Boolean # type: ignore
from org.telegram.tgnet import TLRPC, TLObject # type: ignore
from org.telegram.ui import ChatActivity # type: ignore
from org.telegram.messenger import R, Utilities, AndroidUtilities, ApplicationLoader, MessageObject, AccountInstance, FileLoader # type: ignore
from com.exteragram.messenger.plugins import PluginsController # type: ignore
from android.view import View # type: ignore
from hook_utils import get_private_field, set_private_field # type: ignore
from android.view import MotionEvent # type: ignore


__name__ = "MSLib"
__id__ = "MSLib"
__description__ = "MSLib is a powerful plugin development library"
__icon__ = "MSMainPack/3"
__author__ = "@MiracleStudios"
__version__ = "1.1"
__min_version__ = "12.0.0"


# ==================== Constants ====================
CACHE_DIRECTORY = None
PLUGINS_DIRECTORY = None
COMPANION_PATH = None
LOCALE = "en"
ALLOWED_ARG_TYPES = (str, int, float, bool, Any)
ALLOWED_ORIGIN = (Union, Optional)
NOT_PREMIUM = 0
TELEGRAM_PREMIUM = 1
MSLIB_GLOBAL_PREMIUM = 2
DEFAULT_AUTOUPDATE_TIMEOUT = "600"
DEFAULT_DISABLE_TIMESTAMP_CHECK = False
DEFAULT_DEBUG_MODE = False
MSLIB_AUTOUPDATE_CHANNEL_ID = -1003314084396
MSLIB_AUTOUPDATE_MSG_ID = 3
autoupdater = None

def _init_constants():
    global CACHE_DIRECTORY, PLUGINS_DIRECTORY, COMPANION_PATH, LOCALE
    if CACHE_DIRECTORY is None:
        CACHE_DIRECTORY = os.path.join(AndroidUtilities.getCacheDir().getAbsolutePath(), "mslib_cache")
    if PLUGINS_DIRECTORY is None:
        PLUGINS_DIRECTORY = os.path.dirname(os.path.dirname(__file__))
    if COMPANION_PATH is None:
        COMPANION_PATH = os.path.join(PLUGINS_DIRECTORY, "mslib_companion.py")
    try:
        LOCALE = Locale.getDefault().getLanguage()
    except Exception:
        LOCALE = "en"

# ==================== Utilities ====================
def pluralization_string(count: int, forms: List[str]) -> str:
    if len(forms) == 2:
        return f"{count} {forms[1] if count != 1 else forms[0]}"
    elif len(forms) == 3:
        if count % 10 == 1 and count % 100 != 11:
            return f"{count} {forms[0]}"
        elif 2 <= count % 10 <= 4 and (count % 100 < 10 or count % 100 >= 20):
            return f"{count} {forms[1]}"
        else:
            return f"{count} {forms[2]}"
    else:
        return f"{count} {forms[0]}"

def get_locale() -> str:
    return LOCALE

# ==================== Logging utilities ====================
class CustomLogger(logging.Logger):
    def _log(self, level: int, msg: Any, args: Tuple[Any, ...], exc_info=None, extra=None, stack_info=False, stacklevel=1):
        caller_frame = inspect.stack()[2]
        func_name = caller_frame.function
        
        level_name = logging.getLevelName(level).upper()
        
        prefix_items = [level_name, self.name, func_name]
        prefix_items = filter(lambda i: i, prefix_items)
        prefix_items = [f"[{i}]" for i in prefix_items]
        prefix = " ".join(prefix_items)
        
        try:
            formatted_msg = str(msg) % args if args else str(msg)
        except (TypeError, ValueError):
            formatted_msg = f"{msg} {args}"
        
        _log(f"{prefix} {formatted_msg}")

logging.setLoggerClass(CustomLogger)

def build_log(tag: str, level=logging.INFO) -> logging.Logger:
    logger = logging.getLogger(tag)
    logger.setLevel(level)
    return logger

logger = build_log(__name__)

def format_exc() -> str:
    return traceback.format_exc().strip()

def format_exc_from(e: Exception) -> str:
    return "".join(traceback.format_exception(type(e), e, e.__traceback__)).strip()

def format_exc_only(e: Exception) -> str:
    return ''.join(traceback.format_exception_only(type(e), e)).strip()

# ==================== Markdown & HTML parsers ====================
def add_surrogates(text: str) -> str:
    return re.compile(r"[\U00010000-\U0010FFFF]").sub(
        lambda match: "".join(chr(i) for i in unpack("<HH", match.group().encode("utf-16le"))),
        text
    )

def remove_surrogates(text: str) -> str:
    return text.encode("utf-16", "surrogatepass").decode("utf-16")

class TLEntityType(Enum):
    CODE = 'code'
    PRE = 'pre'
    STRIKETHROUGH = 'strikethrough'
    TEXT_LINK = 'text_link'
    BOLD = 'bold'
    ITALIC = 'italic'
    UNDERLINE = 'underline'
    SPOILER = 'spoiler'
    CUSTOM_EMOJI = 'custom_emoji'
    BLOCKQUOTE = 'blockquote'

@dataclass
class RawEntity:
    type: TLEntityType
    offset: int
    length: int
    extra: Optional[str] = None
    
    def to_tlrpc_object(self) -> 'TLRPC.MessageEntity':
        if self.type == TLEntityType.BOLD:
            entity = TLRPC.TL_messageEntityBold()
        elif self.type == TLEntityType.ITALIC:
            entity = TLRPC.TL_messageEntityItalic()
        elif self.type == TLEntityType.UNDERLINE:
            entity = TLRPC.TL_messageEntityUnderline()
        elif self.type == TLEntityType.STRIKETHROUGH:
            entity = TLRPC.TL_messageEntityStrike()
        elif self.type == TLEntityType.CODE:
            entity = TLRPC.TL_messageEntityCode()
        elif self.type == TLEntityType.PRE:
            entity = TLRPC.TL_messageEntityPre()
            if self.extra:
                entity.language = self.extra
        elif self.type == TLEntityType.TEXT_LINK:
            entity = TLRPC.TL_messageEntityTextUrl()
            entity.url = self.extra or ""
        elif self.type == TLEntityType.CUSTOM_EMOJI:
            entity = TLRPC.TL_messageEntityCustomEmoji()
            try:
                entity.document_id = int(self.extra) if self.extra else 0
            except (ValueError, TypeError):
                entity.document_id = 0
        elif self.type == TLEntityType.SPOILER:
            entity = TLRPC.TL_messageEntitySpoiler()
        elif self.type == TLEntityType.BLOCKQUOTE:
            entity = TLRPC.TL_messageEntityBlockquote()
        else:
            entity = TLRPC.TL_messageEntityUnknown()
        
        entity.offset = self.offset
        entity.length = self.length
        return entity

@dataclass
class ParsedMessage:
    text: str
    entities: List[RawEntity]

class HTMLParser_(HTMLParser):
    def __init__(self):
        super().__init__()
        self.text = ""
        self.entities = []
        self.tag_stack = []
    
    def handle_starttag(self, tag, attrs):
        self.tag_stack.append((tag, dict(attrs), len(self.text)))
    
    def handle_data(self, data):
        self.text += data
    
    def handle_endtag(self, tag):
        if not self.tag_stack or self.tag_stack[-1][0] != tag:
            return
        
        tag_name, attrs, start_pos = self.tag_stack.pop()
        length = len(self.text) - start_pos
        
        if length <= 0:
            return
        
        entity_type = None
        extra = None
        
        if tag_name == 'b' or tag_name == 'strong':
            entity_type = TLEntityType.BOLD
        elif tag_name == 'i' or tag_name == 'em':
            entity_type = TLEntityType.ITALIC
        elif tag_name == 'u':
            entity_type = TLEntityType.UNDERLINE
        elif tag_name == 's' or tag_name == 'del' or tag_name == 'strike':
            entity_type = TLEntityType.STRIKETHROUGH
        elif tag_name == 'code':
            entity_type = TLEntityType.CODE
        elif tag_name == 'pre':
            entity_type = TLEntityType.PRE
        elif tag_name == 'a':
            entity_type = TLEntityType.TEXT_LINK
            extra = attrs.get('href', '')
        elif tag_name == 'emoji':
            entity_type = TLEntityType.CUSTOM_EMOJI
            extra = attrs.get('id', '')
        elif tag_name == 'blockquote':
            entity_type = TLEntityType.BLOCKQUOTE
        elif tag_name == 'spoiler':
            entity_type = TLEntityType.SPOILER
        
        if entity_type:
            self.entities.append(RawEntity(entity_type, start_pos, length, extra))

class HTML:
    @staticmethod
    def parse(text: str) -> ParsedMessage:
        """Parses HTML text and returns ParsedMessage with plain text and entities"""
        parser = HTMLParser_()
        parser.feed(text)
        return ParsedMessage(text=add_surrogates(parser.text), entities=parser.entities)
    
    @staticmethod
    def unparse(text: str, entities: List[RawEntity]) -> str:
        if not entities:
            return text
        
        result = []
        last_offset = 0
        
        for entity in sorted(entities, key=lambda e: e.offset):
            result.append(text[last_offset:entity.offset])
            
            content = text[entity.offset:entity.offset + entity.length]
            
            if entity.type == TLEntityType.BOLD:
                result.append(f"<b>{content}</b>")
            elif entity.type == TLEntityType.ITALIC:
                result.append(f"<i>{content}</i>")
            elif entity.type == TLEntityType.UNDERLINE:
                result.append(f"<u>{content}</u>")
            elif entity.type == TLEntityType.STRIKETHROUGH:
                result.append(f"<s>{content}</s>")
            elif entity.type == TLEntityType.CODE:
                result.append(f"<code>{content}</code>")
            elif entity.type == TLEntityType.PRE:
                result.append(f"<pre>{content}</pre>")
            elif entity.type == TLEntityType.TEXT_LINK:
                result.append(f'<a href="{entity.extra}">{content}</a>')
            elif entity.type == TLEntityType.CUSTOM_EMOJI:
                result.append(f'<emoji id="{entity.extra}">{content}</emoji>')
            elif entity.type == TLEntityType.BLOCKQUOTE:
                result.append(f"<blockquote>{content}</blockquote>")
            elif entity.type == TLEntityType.SPOILER:
                result.append(f"<spoiler>{content}</spoiler>")
            
            last_offset = entity.offset + entity.length
        
        result.append(text[last_offset:])
        return ''.join(result)

class Markdown:
    BOLD_DELIM = "**"
    ITALIC_DELIM = "_"
    UNDERLINE_DELIM = "__"
    STRIKE_DELIM = "~~"
    SPOILER_DELIM = "||"
    CODE_DELIM = "`"
    PRE_DELIM = "```"
    BLOCKQUOTE_DELIM = ">"
    
    @staticmethod
    def parse(text: str, strict: bool = False) -> ParsedMessage:
        """Parse Markdown text and return ParsedMessage with plain text and entities"""
        entities = []
        markers_to_remove = []
        
        # Bold (**text**)
        for match in re.finditer(r'\*\*(.+?)\*\*', text):
            content = match.group(1)
            start = match.start()
            markers_to_remove.append((start, match.end(), content))
            
        # Strikethrough (~~text~~)
        for match in re.finditer(r'~~(.+?)~~', text):
            content = match.group(1)
            start = match.start()
            # Check if not part of bold
            if not any(m[0] <= start < m[1] for m in markers_to_remove):
                markers_to_remove.append((start, match.end(), content))
        
        # Code (`text`)
        for match in re.finditer(r'`([^`]+)`', text):
            content = match.group(1)
            start = match.start()
            markers_to_remove.append((start, match.end(), content))
        
        # Spoiler (||text||)
        for match in re.finditer(r'\|\|(.+?)\|\|', text):
            content = match.group(1)
            start = match.start()
            markers_to_remove.append((start, match.end(), content))
        
        # Italic (*text* but not **) - process after bold
        for match in re.finditer(r'(?<!\*)\*([^*]+)\*(?!\*)', text):
            content = match.group(1)
            start = match.start()
            # Check if not part of other formatting
            if not any(m[0] <= start < m[1] for m in markers_to_remove):
                markers_to_remove.append((start, match.end(), content))
        
        markers_to_remove.sort(key=lambda x: x[0], reverse=True)
        
        clean_text = text
        offset_map = {}
        
        for start, end, content in markers_to_remove:
            marker_len = end - start
            content_len = len(content)
            shift = marker_len - content_len
            offset_map[start] = (start, shift)

            clean_text = clean_text[:start] + content + clean_text[end:]

        for match in re.finditer(r'\*\*(.+?)\*\*', text):
            original_start = match.start()
            content_len = len(match.group(1))

            new_offset = original_start
            for pos, (orig_pos, shift) in offset_map.items():
                if orig_pos < original_start:
                    new_offset -= shift
            
            entities.append(RawEntity(
                TLEntityType.BOLD,
                new_offset,
                content_len
            ))

        for match in re.finditer(r'~~(.+?)~~', text):
            original_start = match.start()
            content_len = len(match.group(1))
            
            new_offset = original_start
            for pos, (orig_pos, shift) in offset_map.items():
                if orig_pos < original_start:
                    new_offset -= shift
            
            entities.append(RawEntity(
                TLEntityType.STRIKETHROUGH,
                new_offset,
                content_len
            ))

        for match in re.finditer(r'`([^`]+)`', text):
            original_start = match.start()
            content_len = len(match.group(1))
            
            new_offset = original_start
            for pos, (orig_pos, shift) in offset_map.items():
                if orig_pos < original_start:
                    new_offset -= shift
            
            entities.append(RawEntity(
                TLEntityType.CODE,
                new_offset,
                content_len
            ))

        for match in re.finditer(r'\|\|(.+?)\|\|', text):
            original_start = match.start()
            content_len = len(match.group(1))
            
            new_offset = original_start
            for pos, (orig_pos, shift) in offset_map.items():
                if orig_pos < original_start:
                    new_offset -= shift
            
            entities.append(RawEntity(
                TLEntityType.SPOILER,
                new_offset,
                content_len
            ))

        for match in re.finditer(r'(?<!\*)\*([^*]+)\*(?!\*)', text):
            original_start = match.start()
            content_len = len(match.group(1))
            
            new_offset = original_start
            for pos, (orig_pos, shift) in offset_map.items():
                if orig_pos < original_start:
                    new_offset -= shift
            
            entities.append(RawEntity(
                TLEntityType.ITALIC,
                new_offset,
                content_len
            ))

        entities.sort(key=lambda e: e.offset)
        
        return ParsedMessage(text=add_surrogates(clean_text), entities=entities)
    
    @staticmethod
    def unparse(text: str, entities: List[RawEntity]) -> str:
        if not entities:
            return text
        
        result = []
        last_offset = 0
        
        for entity in sorted(entities, key=lambda e: e.offset):
            result.append(text[last_offset:entity.offset])
            
            content = text[entity.offset:entity.offset + entity.length]
            
            if entity.type == TLEntityType.BOLD:
                result.append(f"**{content}**")
            elif entity.type == TLEntityType.ITALIC:
                result.append(f"*{content}*")
            elif entity.type == TLEntityType.UNDERLINE:
                result.append(f"__{content}__")
            elif entity.type == TLEntityType.STRIKETHROUGH:
                result.append(f"~~{content}~~")
            elif entity.type == TLEntityType.CODE:
                result.append(f"`{content}`")
            elif entity.type == TLEntityType.PRE:
                result.append(f"```{content}```")
            elif entity.type == TLEntityType.TEXT_LINK:
                result.append(f"[{content}]({entity.extra})")
            elif entity.type == TLEntityType.SPOILER:
                result.append(f"||{content}||")
            else:
                result.append(content)
            
            last_offset = entity.offset + entity.length
        
        result.append(text[last_offset:])
        return ''.join(result)

def link(text: str, url: str) -> str:
    return f'<a href="{url}">{text}</a>'

# ==================== Working with Java collections ====================
def arraylist_to_list(jarray: Optional[ArrayList]) -> Optional[List[Any]]:
    return [jarray.get(i) for i in range(jarray.size())] if jarray else None

def list_to_arraylist(python_list: Optional[List[Any]], int_auto_convert: bool = True) -> Optional[ArrayList]:
    if not python_list:
        return None
    
    arraylist = ArrayList()
    for item in python_list:
        if int_auto_convert and isinstance(item, int):
            arraylist.add(Integer(item))
        else:
            arraylist.add(item)
    return arraylist

# ==================== Compression & Encoding ====================
def compress_and_encode(data: Union[bytes, str], level: int = 9) -> str:
    """Сжимает и кодирует данные в base64"""
    try:
        if isinstance(data, str):
            data = data.encode('utf-8')
        compressed = zlib.compress(data, level=level)
        return base64.b64encode(compressed).decode('utf-8')
    except Exception as e:
        logger.error(f"Failed to compress and encode: {format_exc_only(e)}")
        return ""

def decode_and_decompress(encoded_data: Union[bytes, str]) -> bytes:
    """Декодирует из base64 и разжимает данные"""
    try:
        if isinstance(encoded_data, str):
            encoded_data = encoded_data.encode('utf-8')
        compressed = base64.b64decode(encoded_data)
        return zlib.decompress(compressed)
    except Exception as e:
        logger.error(f"Failed to decode and decompress: {format_exc_only(e)}")
        return b""

# ==================== Decorators for plugin development ====================
def command(
        cmd: Optional[str] = None, *,
        aliases: Optional[List[str]] = None,
        doc: Optional[str] = None,
        enabled: Optional[Union[str, bool]] = None
):
    """
    Decorator for commands
    
    Args:
        cmd (str): The command name (uses function name if not specified)
        aliases (List[str]): A list of aliases for the command
        doc (str): String-key in `strings` for command description
        enabled (str/bool): Setting-key or boolean for enabling the command
    """
    def decorator(func):
        func.__is_command__ = True
        func.__aliases__ = aliases or []
        func.__cdoc__ = doc
        func.__enabled__ = enabled
        func.__cmd__ = cmd or func.__name__
        return func
    return decorator

def uri(uri: str):
    """
    Decorator for URIs
    
    Args:
        uri (str): The URI
    """
    def decorator(func):
        func.__is_uri_handler__ = True
        func.__uri__ = uri
        return func
    return decorator

def message_uri(uri: str, support_long_click: bool = False):
    """
    Decorator for URIs in messages
    
    Args:
        uri (str): The URI
        support_long_click (bool): if true, func will be called on long click too
    """
    def decorator(func):
        func.__is_uri_message_handler__ = True
        func.__uri__ = uri
        func.__support_long__ = support_long_click
        return func
    return decorator

# ==================== PluginsData helper (как в CactusLib) ====================

class PluginsData:
    """Класс для парсинга метаданных плагинов"""
    _current_instance = None
    plugins = {}
    
    @classmethod
    def parse(cls, plugin_path: str, plugin_id=None):
        """Парсит плагин и извлекает strings, commands, description"""
        strings, commands, description = cls.get_plugin_strings_and_commands(plugin_path)
        if not strings:
            return
        
        cls.plugins[plugin_id or strings.get("__id__", os.path.basename(plugin_path))] = {
            "strings": strings,
            "commands": commands,
            "description": description
        }
    
    @classmethod
    def description(cls, plugin_id: str) -> str:
        """Возвращает описание плагина"""
        if plugin_id not in cls.plugins:
            return "<unknown-plugin>"
        
        return cls.locale(plugin_id).get("__doc__", cls.plugins[plugin_id].get("description", ""))
    
    @classmethod
    def locale(cls, plugin_id: str) -> Dict[str, str]:
        """Возвращает локализованные строки для плагина"""
        locale_dict: Dict[str, Union[str, Dict[str, str]]] = cls.plugins[plugin_id]["strings"].get(
            LOCALE,
            cls.plugins[plugin_id]["strings"]
        )
        if "en" in locale_dict:
            locale_dict = locale_dict["en"]
        
        return locale_dict
    
    @classmethod
    def commands(cls, plugin_id: str) -> Dict[str, str]:
        """Возвращает команды плагина"""
        if plugin_id not in cls.plugins:
            return {}
        return cls.plugins[plugin_id].get("commands", {})
    
    @staticmethod
    def get_plugin_strings_and_commands(
            filepath: Optional[str] = None,
            file_content: Optional[str] = None
    ) -> Tuple[Dict[str, Dict[str, str]], Dict[str, str], Optional[str]]:
        """Извлекает strings, commands и description из файла плагина"""
        if file_content:
            tree = ast.parse(file_content, filename=filepath or "<unknown>")
        else:
            if not os.path.exists(filepath):
                return {}, {}, None
            
            with open(filepath, "r", encoding="utf-8") as f:
                tree = ast.parse(f.read(), filename=filepath)
        
        description, strings, commands, _id = "", {}, {}, None
        
        # Извлекаем __description__ и __id__
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "__description__":
                        if isinstance(node.value, ast.Constant):
                            description = node.value.value
                            if _id:
                                break
                    if isinstance(target, ast.Name) and target.id == "__id__":
                        if isinstance(node.value, ast.Constant):
                            _id = node.value.value
                            if description:
                                break
        
        # Ищем класс наследующийся от MSLib.Plugin/CactusModule и т.д.
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                inherits_from_plugin = False
                for base in node.bases:
                    if (isinstance(base, ast.Attribute) and
                            isinstance(base.value, ast.Name) and
                            base.value.id in ["MSLib", "CactusUtils"] and
                            base.attr in ["Plugin", "CactusPlugin", "CactusModule", "MSLib"]):
                        inherits_from_plugin = True
                        break
                    elif isinstance(base, ast.Name) and base.id in ["MSLib", "BasePlugin"]:
                        inherits_from_plugin = True
                        break
                
                if inherits_from_plugin:
                    # Извлекаем strings
                    for item in node.body:
                        if isinstance(item, ast.Assign):
                            for target in item.targets:
                                if isinstance(target, ast.Name) and target.id == "strings":
                                    try:
                                        strings = ast.literal_eval(item.value)  # type: ignore
                                    except Exception:
                                        pass
                                    break
                        
                        # Извлекаем commands из декораторов
                        elif isinstance(item, ast.FunctionDef):
                            for decorator in item.decorator_list:
                                is_command_decorator = False
                                decorator_args = {}
                                
                                if isinstance(decorator, ast.Call):
                                    if (isinstance(decorator.func, ast.Name) and decorator.func.id == "command") or \
                                            (isinstance(decorator.func, ast.Attribute) and decorator.func.attr == "command"):
                                        is_command_decorator = True
                                        for keyword in decorator.keywords:
                                            if keyword.arg == "command":
                                                try:
                                                    decorator_args['cmd'] = ast.literal_eval(keyword.value)  # type: ignore
                                                except Exception:
                                                    pass
                                            elif keyword.arg == "doc":
                                                try:
                                                    decorator_args['doc'] = ast.literal_eval(keyword.value)  # type: ignore
                                                except Exception:
                                                    pass
                                        if decorator.args and len(decorator.args) > 0 and 'cmd' not in decorator_args:
                                            try:
                                                decorator_args['cmd'] = ast.literal_eval(decorator.args[0])  # type: ignore
                                            except Exception:
                                                pass
                                
                                elif isinstance(decorator, ast.Name) and decorator.id == "command":
                                    is_command_decorator = True
                                    decorator_args['cmd'] = item.name
                                    decorator_args['doc'] = None
                                
                                if is_command_decorator:
                                    cmd_value = decorator_args.get('cmd', item.name)
                                    doc_value = decorator_args.get('doc')
                                    commands[cmd_value] = doc_value
                                    break
                    
                    if strings is not None:
                        strings["__id__"] = _id
                        return strings, commands, description
        
        strings["__id__"] = _id
        return strings, commands, description
    
    @staticmethod
    def is_mslib_plugin(filepath: str) -> bool:
        """Проверяет является ли файл MSLib плагином"""
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                tree = ast.parse(f.read(), filename=filepath)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    for base in node.bases:
                        if (isinstance(base, ast.Attribute) and
                                isinstance(base.value, ast.Name) and
                                base.value.id in ["MSLib", "CactusUtils"] and
                                base.attr in ["Plugin", "CactusPlugin", "CactusModule"]):
                            return True
                        elif isinstance(base, ast.Name) and base.id in ["MSLib", "BasePlugin"]:
                            return True
            
            return False
        except Exception:
            return False

# ==================== PluginInfo helper ====================
class PluginInfo:
    """Класс для работы с информацией о плагине"""
    def __init__(self, lib_instance, plugin_instance, is_compatible: bool = True):
        self.lib = lib_instance
        self.plugin = plugin_instance
        self.is_compatible = is_compatible
    
    def export(self, with_data: bool = True) -> Dict[str, Any]:
        """Экспортирует плагин в словарь"""
        try:
            plugin_id = getattr(self.plugin, 'id', 'unknown')
            plugin_name = getattr(self.plugin, 'name', plugin_id)
            plugin_version = getattr(self.plugin, 'version', '1.0')
            plugin_enabled = getattr(self.plugin, 'enabled', False)
            
            data = {
                "plugin_meta": {
                    "id": plugin_id,
                    "name": plugin_name,
                    "version": plugin_version,
                    "enabled": plugin_enabled
                },
                "file_content": "",  # Должно заполняться при чтении файла
            }
            
            if with_data and hasattr(self.plugin, '_export_data'):
                try:
                    exported = self.plugin._export_data()
                    if exported:
                        data["data"] = exported
                except Exception as e:
                    logger.error(f"Failed to export data from {plugin_id}: {format_exc_only(e)}")
            
            # Экспорт настроек
            if with_data:
                try:
                    settings = {}
                    # Здесь можно добавить логику экспорта настроек
                    if settings:
                        data["settings"] = settings
                except Exception as e:
                    logger.error(f"Failed to export settings from {plugin_id}: {format_exc_only(e)}")
            
            return data
        except Exception as e:
            logger.error(f"Failed to export plugin: {format_exc()}")
            return {}

# ==================== Command system ====================
class CannotCastError(Exception):
    pass

class WrongArgumentAmountError(Exception):
    pass

class MissingRequiredArguments(Exception):
    pass

class InvalidTypeError(Exception):
    pass

class ArgSpec:
    def __init__(self, name, annotation, kind, default=None, is_optional=False):
        self.name = name
        self.annotation = annotation
        self.kind = kind
        self.default = default if default is not None else inspect.Parameter.empty
        self.is_optional = is_optional
    
    @classmethod
    def from_parameter(cls, param):
        is_optional = False
        annotation = param.annotation
        
        if hasattr(annotation, '__origin__'):
            if annotation.__origin__ is Union:
                if type(None) in annotation.__args__:
                    is_optional = True
                    non_none_args = [arg for arg in annotation.__args__ if arg is not type(None)]
                    if len(non_none_args) == 1:
                        annotation = non_none_args[0]
        
        return cls(
            name=param.name,
            annotation=annotation if annotation != inspect.Parameter.empty else Any,
            kind=param.kind,
            default=param.default,
            is_optional=is_optional
        )

@dataclass
class UriCallback:
    """Callback для URI обработчиков (как в CactusLib)"""
    cell: Any  # ChatMessageCell
    message: MessageObject
    method: str
    raw_url: str
    long_press: bool = False
    
    def edit_message(self, text: str, **kwargs):
        """Редактирует сообщение"""
        fragment = kwargs.pop("fragment", get_last_fragment())
        # Будем использовать edit_message когда он определён ниже
        from MSLib import edit_message as _edit_msg
        _edit_msg(self.message, text, fragment=fragment, **kwargs)
        if kwargs.get("markup", None) is None and self.message.messageOwner.reply_markup:
            self.edit_markup()
    
    edit = edit_message
    
    def edit_markup(self, markup=None):
        """Редактирует Inline-клавиатуру"""
        # Будем использовать edit_message_markup когда он определён ниже
        from MSLib import edit_message_markup as _edit_markup
        _edit_markup(self.cell, markup)
    
    def delete_message(self):
        """Удаляет сообщение"""
        dialog_id = self.message.getDialogId()
        chat = get_messages_controller().getChat(-dialog_id)
        if self.message.canDeleteMessage(
                self.message.getChatMode() == 1,
                chat
        ):
            topic_id = self.message.getTopicId()
            # Используем Requests когда он определён
            Requests.delete_messages(
                [self.message.getRealId()],
                dialog_id,
                lambda r, e: None
            )
    
    delete = delete_message

@dataclass  
class Uri:
    """Класс для создания URI ссылок (tg://cactus/...)"""
    plugin_id: str
    command: str
    kwargs: Dict[str, str]
    
    @classmethod
    def create(cls, plugin, cmd: str, **kwargs):
        """Создаёт Uri из plugin объекта"""
        return cls(
            plugin_id=plugin.id if hasattr(plugin, 'id') else str(plugin),
            command=cmd,
            kwargs=kwargs
        )
    
    def string(self) -> str:
        """Возвращает строку URI"""
        base = f"tg://mslib/{self.plugin_id}/{self.command}"
        if self.kwargs:
            params = urlencode(self.kwargs)
            return f"{base}?{params}"
        return base
    
    def __str__(self):
        return self.string()

@dataclass
class MessageUri(Uri):
    """Класс для URI внутри сообщений (tg://mslibX/...)"""
    
    def string(self) -> str:
        """Возвращает строку URI для сообщений"""
        base = f"tg://mslibX/{self.plugin_id}/{self.command}"
        if self.kwargs:
            params = urlencode(self.kwargs)
            return f"{base}?{params}"
        return base

class Command:
    def __init__(self, func, name, args=None, subcommands=None, error_handler=None, aliases=None, doc=None, enabled=None):
        self.func = func
        self.name = name
        self.args = args if args is not None else []
        self.subcommands = subcommands if subcommands is not None else {}
        self.error_handler = error_handler
        # Новые атрибуты из CactusLib
        self.aliases = aliases if aliases is not None else []
        self.doc = doc  # Ключ в strings для описания команды
        self.enabled = enabled  # Ключ настройки или булево значение
    
    def subcommand(self, name: str):
        def decorator(func: Callable):
            cmd = create_command(func, name)
            self.subcommands[name] = cmd
            return func
        return decorator
    
    def register_error_handler(self, func: Callable[[Any, int, Exception], HookResult]):
        self.error_handler = func
        return func
    
    def add_alias(self, alias: str):
        """Добавляет алиас к команде"""
        if alias not in self.aliases:
            self.aliases.append(alias)
    
    def remove_alias(self, alias: str):
        """Удаляет алиас из команды"""
        if alias in self.aliases:
            self.aliases.remove(alias)
    
    def is_enabled(self, plugin_instance=None) -> bool:
        """Проверяет включена ли команда"""
        if self.enabled is None:
            return True
        if isinstance(self.enabled, bool):
            return self.enabled
        if isinstance(self.enabled, str) and plugin_instance:
            # Пытаемся получить настройку из плагина
            try:
                return plugin_instance.get_setting(self.enabled, True)
            except Exception:
                return True
        return True
    
    def get_subcommand(self, name: str) -> Optional['Command']:
        """Получает подкоманду по имени"""
        return self.subcommands.get(name)
    
    def has_subcommands(self) -> bool:
        """Проверяет есть ли подкоманды"""
        return len(self.subcommands) > 0
    
    def list_subcommands(self) -> List[str]:
        """Возвращает список имён подкоманд"""
        return list(self.subcommands.keys())

def is_allowed_type(arg_type) -> bool:
    if arg_type in ALLOWED_ARG_TYPES:
        return True
    
    if arg_type is type(None):
        return True
    
    origin = get_origin(arg_type)
    if origin in ALLOWED_ORIGIN:
        return all(is_allowed_type(t) for t in get_args(arg_type))
    return False

def create_command(func: Callable, name: str) -> Command:
    signature = inspect.signature(func)
    parameters = list(signature.parameters.values())
    return_type = signature.return_annotation
    
    if len(parameters) < 2:
        raise MissingRequiredArguments("Command must have 'param' variable as first argument and 'account' variable as second argument")
    
    args = [ArgSpec.from_parameter(param) for param in parameters]
    
    for index, arg in enumerate(args):
        if arg.kind == inspect.Parameter.VAR_POSITIONAL:
            if index != len(args) - 1:
                raise InvalidTypeError(f"VAR_POSITIONAL argument must be the last argument")
            if arg.annotation is not str and arg.annotation is not Any:
                raise InvalidTypeError(f"VAR_POSITIONAL argument must be str or Any, got {arg.annotation}")
        elif not is_allowed_type(arg.annotation):
            raise InvalidTypeError(f"Unsupported argument type: {arg.annotation}")
    
    if return_type != HookResult:
        return_type_name = "NoneType" if return_type == inspect.Parameter.empty else return_type
        raise InvalidTypeError(f"Command function must return {HookResult} object, got {return_type_name}")
    
    # Извлекаем атрибуты из декоратора @command
    aliases = getattr(func, '__aliases__', None) or []
    doc = getattr(func, '__cdoc__', None)
    enabled = getattr(func, '__enabled__', None)
    
    return Command(
        func=func,
        name=name,
        args=args,
        aliases=aliases,
        doc=doc,
        enabled=enabled
    )

def cast_arg(arg: str, target_type: type):
    if target_type is str or target_type is Any:
        return arg
    elif target_type is int:
        return int(arg)
    elif target_type is float:
        return float(arg)
    elif target_type is bool:
        lower = arg.lower()
        if lower in ('true', '1', 'yes', 'on'):
            return True
        elif lower in ('false', '0', 'no', 'off'):
            return False
        raise CannotCastError(f"Cannot cast '{arg}' to bool")
    else:
        raise CannotCastError(f"Unsupported type: {target_type}")

def smart_cast(arg, annotation):
    if annotation in ALLOWED_ARG_TYPES:
        try:
            return cast_arg(arg, annotation)
        except Exception:
            raise CannotCastError("Cannot cast '{}' to {}".format(arg, annotation))
    
    if hasattr(annotation, '__origin__') and annotation.__origin__ is Union:
        for arg_type in annotation.__args__:
            if arg_type is type(None):
                continue
            try:
                return cast_arg(arg, arg_type)
            except Exception:
                continue
        raise CannotCastError("Cannot cast '{}' to any of Union types".format(arg))
    
    raise CannotCastError("Unsupported annotation: {}".format(annotation))

def parse_quoted_args(text: str) -> List[str]:
    """
    Парсит аргументы с поддержкой кавычек.
    Поддерживает одинарные и двойные кавычки, экранирование.
    
    Примеры:
        'arg1 arg2' -> ['arg1', 'arg2']
        '"arg with spaces" arg2' -> ['arg with spaces', 'arg2']
        "'single quoted' arg" -> ['single quoted', 'arg']
    """
    args = []
    current_arg = []
    in_quotes = False
    quote_char = None
    escaped = False
    
    for char in text:
        if escaped:
            current_arg.append(char)
            escaped = False
            continue
        
        if char == '\\':
            escaped = True
            continue
        
        if char in ('"', "'"):
            if not in_quotes:
                in_quotes = True
                quote_char = char
            elif char == quote_char:
                in_quotes = False
                quote_char = None
            else:
                current_arg.append(char)
            continue
        
        if char == ' ' and not in_quotes:
            if current_arg:
                args.append(''.join(current_arg))
                current_arg = []
            continue
        
        current_arg.append(char)
    
    if current_arg:
        args.append(''.join(current_arg))
    
    return args

def parse_args(raw_args: List[str], command_args: List[ArgSpec]) -> Tuple[Any, ...]:
    out: List[Any] = []
    required_arg_count = sum(
        1 for arg in command_args
        if not arg.is_optional and arg.default == inspect.Parameter.empty and arg.kind != inspect.Parameter.VAR_POSITIONAL
    )
    is_variadic = any(arg.kind == inspect.Parameter.VAR_POSITIONAL for arg in command_args)
    
    if not is_variadic and len(raw_args) > len(command_args):
        raise WrongArgumentAmountError(f"Too many arguments: expected {len(command_args)}, got {len(raw_args)}")
    if len(raw_args) < required_arg_count:
        raise WrongArgumentAmountError(f"Not enough arguments: expected at least {required_arg_count}, got {len(raw_args)}")
    
    for i, cmd_arg in enumerate(command_args):
        if cmd_arg.kind == inspect.Parameter.VAR_POSITIONAL:
            out.extend(raw_args[i:])
            break
        elif i < len(raw_args):
            out.append(smart_cast(raw_args[i], cmd_arg.annotation))
        elif cmd_arg.default != inspect.Parameter.empty:
            out.append(cmd_arg.default)
        elif cmd_arg.is_optional:
            out.append(None)
        else:
            raise WrongArgumentAmountError(f"Missing required argument: {cmd_arg.name}")
    
    return tuple(out)

@dataclass
class CommandContext:
    """Контекст выполнения команды"""
    command_name: str
    raw_text: str
    args: Tuple[Any, ...]
    param: Any
    account: Any
    plugin_instance: Optional[Any] = None
    dispatcher: Optional['Dispatcher'] = None

class Dispatcher:
    def __init__(self, plugin_id: str, prefix: str = ".", commands_priority: int = -1):
        self.plugin_id = plugin_id
        self.prefix = prefix
        self.commands_priority = commands_priority
        self.listeners: Dict[str, Command] = {}
        self.aliases: Dict[str, str] = {}  # alias -> command_name
        self.before_hooks: List[Callable[[CommandContext], Optional[Any]]] = []  # Middleware до выполнения
        self.after_hooks: List[Callable[[CommandContext, Any], None]] = []  # Middleware после выполнения
    
    @staticmethod
    def validate_prefix(prefix: str) -> bool:
        return len(prefix) == 1 and not prefix.isalnum()
    
    def set_prefix(self, prefix: str):
        if not self.validate_prefix(prefix):
            logger.error(f"Invalid prefix: {prefix}")
            return
        
        logger.info(f"{self.plugin_id} dp: Set '{prefix}' prefix.")
        self.prefix = prefix
    
    def register_command(self, name: str):
        def decorator(func: Callable):
            cmd = create_command(func, name)
            self.listeners[name] = cmd
            
            # Регистрируем алиасы
            for alias in cmd.aliases:
                self.aliases[alias] = name
                logger.info(f"{self.plugin_id} dp: Registered alias '{alias}' for command {name}.")
            
            logger.info(f"{self.plugin_id} dp: Registered command {name}.")
            return func
        return decorator
    
    def add_alias(self, command_name: str, alias: str):
        """Добавляет алиас к команде"""
        if command_name in self.listeners:
            self.listeners[command_name].add_alias(alias)
            self.aliases[alias] = command_name
            logger.info(f"{self.plugin_id} dp: Added alias '{alias}' for command {command_name}.")
    
    def remove_alias(self, alias: str):
        """Удаляет алиас"""
        if alias in self.aliases:
            command_name = self.aliases[alias]
            if command_name in self.listeners:
                self.listeners[command_name].remove_alias(alias)
            del self.aliases[alias]
            logger.info(f"{self.plugin_id} dp: Removed alias '{alias}'.")
    
    def get_command(self, name: str) -> Optional[Command]:
        """Получает команду по имени или алиасу"""
        # Сначала ищем по имени
        if name in self.listeners:
            return self.listeners[name]
        # Затем по алиасу
        if name in self.aliases:
            command_name = self.aliases[name]
            return self.listeners.get(command_name)
        return None
    
    def unregister_command(self, name: str):
        if name in self.listeners:
            # Удаляем также все алиасы команды
            cmd = self.listeners[name]
            for alias in cmd.aliases:
                if alias in self.aliases:
                    del self.aliases[alias]
            
            logger.info(f"{self.plugin_id} dp: Unregistered command {name}.")
            del self.listeners[name]
    
    def get_all_commands(self) -> Dict[str, Command]:
        """Возвращает словарь всех команд"""
        return self.listeners.copy()
    
    def get_command_with_aliases(self, name: str) -> Tuple[Optional[Command], List[str]]:
        """Возвращает команду и список её алиасов"""
        cmd = self.get_command(name)
        if cmd:
            return cmd, cmd.aliases.copy()
        return None, []
    
    def execute_command(self, command: Command, parsed_args: Tuple[Any, ...], plugin_instance=None, context: Optional[CommandContext] = None) -> Optional[Any]:
        """Выполняет команду с распарсенными аргументами"""
        try:
            # Выполняем before hooks
            if context:
                for hook in self.before_hooks:
                    try:
                        hook_result = hook(context)
                        if hook_result is not None:
                            # Хук прервал выполнение
                            logger.debug(f"Before hook interrupted command execution")
                            return hook_result
                    except Exception as hook_error:
                        logger.error(f"Before hook failed: {format_exc_only(hook_error)}")
            
            # Выполняем команду
            result = command.func(*parsed_args)
            
            # Выполняем after hooks
            if context:
                for hook in self.after_hooks:
                    try:
                        hook(context, result)
                    except Exception as hook_error:
                        logger.error(f"After hook failed: {format_exc_only(hook_error)}")
            
            return result
        except Exception as e:
            # Если есть error_handler, вызываем его
            if command.error_handler:
                try:
                    # error_handler принимает (param, account, exception)
                    # Но у нас parsed_args уже содержит param и account
                    if len(parsed_args) >= 2:
                        return command.error_handler(parsed_args[0], parsed_args[1], e)
                except Exception as handler_error:
                    logger.error(f"Error handler failed: {format_exc_only(handler_error)}")
            logger.error(f"Command execution failed: {format_exc_only(e)}")
            raise
    
    def dispatch(self, message_text: str, param: Any, account: Any, plugin_instance=None) -> Optional[Any]:
        """
        Главная функция диспетчеризации команд.
        Парсит текст сообщения, находит команду и выполняет её.
        Поддерживает подкоманды (например: .cmd subcommand args).
        
        Args:
            message_text: Текст сообщения
            param: Первый аргумент команды (обычно CommandParams или MessageObject)
            account: Аккаунт пользователя
            plugin_instance: Инстанс плагина для проверки enabled
        
        Returns:
            Результат выполнения команды или None
        """
        # Проверяем что сообщение начинается с префикса
        if not message_text.startswith(self.prefix):
            return None
        
        # Убираем префикс
        text_without_prefix = message_text[len(self.prefix):]
        
        # Парсим команду и аргументы
        parts = text_without_prefix.split(maxsplit=1)
        if not parts:
            return None
        
        command_name = parts[0]
        args_text = parts[1] if len(parts) > 1 else ""
        
        # Находим команду
        cmd = self.get_command(command_name)
        if not cmd:
            return None
        
        # Проверяем включена ли команда
        if not cmd.is_enabled(plugin_instance):
            logger.info(f"Command {command_name} is disabled")
            return None
        
        # Проверяем есть ли подкоманда
        if args_text and cmd.has_subcommands():
            subparts = args_text.split(maxsplit=1)
            potential_subcommand = subparts[0]
            
            subcmd = cmd.get_subcommand(potential_subcommand)
            if subcmd:
                # Нашли подкоманду, используем её
                cmd = subcmd
                args_text = subparts[1] if len(subparts) > 1 else ""
                logger.debug(f"Using subcommand: {potential_subcommand}")
        
        # Парсим аргументы с поддержкой кавычек
        raw_args = parse_quoted_args(args_text) if args_text else []
        
        try:
            # Первые два аргумента - param и account (добавляем их автоматически)
            parsed_args = parse_args(raw_args, cmd.args[2:])  # Пропускаем param и account
            full_args = (param, account) + parsed_args
            
            # Создаём контекст команды
            context = CommandContext(
                command_name=command_name,
                raw_text=message_text,
                args=full_args,
                param=param,
                account=account,
                plugin_instance=plugin_instance,
                dispatcher=self
            )
            
            # Выполняем команду
            return self.execute_command(cmd, full_args, plugin_instance, context)
            
        except WrongArgumentAmountError as e:
            logger.warning(f"Wrong argument amount for {command_name}: {e}")
            if cmd.error_handler:
                return cmd.error_handler(param, account, e)
            return None
        except CannotCastError as e:
            logger.warning(f"Cannot cast argument for {command_name}: {e}")
            if cmd.error_handler:
                return cmd.error_handler(param, account, e)
            return None
        except Exception as e:
            logger.error(f"Command dispatch failed: {format_exc_only(e)}")
            if cmd.error_handler:
                return cmd.error_handler(param, account, e)
            raise
    
    def set_command_enabled(self, name: str, enabled: bool, plugin_instance=None):
        """Включает или выключает команду динамически"""
        cmd = self.get_command(name)
        if cmd:
            cmd.enabled = enabled
            logger.info(f"{self.plugin_id} dp: Command '{name}' {'enabled' if enabled else 'disabled'}.")
    
    def is_command_enabled(self, name: str, plugin_instance=None) -> bool:
        """Проверяет включена ли команда"""
        cmd = self.get_command(name)
        if cmd:
            return cmd.is_enabled(plugin_instance)
        return False
    
    def get_enabled_commands(self, plugin_instance=None) -> List[str]:
        """Возвращает список имён включённых команд"""
        return [name for name, cmd in self.listeners.items() if cmd.is_enabled(plugin_instance)]
    
    def get_disabled_commands(self, plugin_instance=None) -> List[str]:
        """Возвращает список имён выключенных команд"""
        return [name for name, cmd in self.listeners.items() if not cmd.is_enabled(plugin_instance)]
    
    def reset_command(self, name: str):
        """Сбрасывает команду к начальному состоянию (из декоратора)"""
        cmd = self.get_command(name)
        if cmd:
            # Восстанавливаем enabled из __enabled__ атрибута функции
            original_enabled = getattr(cmd.func, '__enabled__', None)
            cmd.enabled = original_enabled
            logger.info(f"{self.plugin_id} dp: Reset command '{name}' to original state.")
    
    def get_command_info(self, name: str) -> Optional[Dict[str, Any]]:
        """Возвращает полную информацию о команде"""
        cmd = self.get_command(name)
        if not cmd:
            return None
        
        return {
            'name': cmd.name,
            'aliases': cmd.aliases,
            'doc': cmd.doc,
            'enabled': cmd.enabled,
            'args': [{'name': arg.name, 'type': str(arg.annotation), 'optional': arg.is_optional, 'default': arg.default} 
                     for arg in cmd.args[2:]],  # Пропускаем param и account
            'has_error_handler': cmd.error_handler is not None,
            'subcommands': list(cmd.subcommands.keys())
        }
    
    def get_all_commands_info(self) -> Dict[str, Dict[str, Any]]:
        """Возвращает информацию обо всех командах"""
        return {name: self.get_command_info(name) for name in self.listeners.keys()}
    
    def format_command_list(self, plugin_instance=None) -> str:
        """Форматирует список команд для отображения пользователю"""
        lines = [f"Commands (prefix: {self.prefix}):"]
        
        for name, cmd in sorted(self.listeners.items()):
            status = "✓" if cmd.is_enabled(plugin_instance) else "✗"
            aliases_str = f" (aliases: {', '.join(cmd.aliases)})" if cmd.aliases else ""
            doc_str = f" - {cmd.doc}" if cmd.doc else ""
            lines.append(f"  {status} {self.prefix}{name}{aliases_str}{doc_str}")
        
        return "\n".join(lines)
    
    def generate_help_text(self, command_name: str, strings: Optional[Dict[str, str]] = None) -> str:
        """
        Генерирует текст справки для команды.
        
        Args:
            command_name: Имя команды
            strings: Словарь локализации (опционально)
        
        Returns:
            Форматированный текст справки
        """
        cmd = self.get_command(command_name)
        if not cmd:
            return f"Command '{command_name}' not found"
        
        lines = []
        
        # Заголовок
        lines.append(f"<b>{self.prefix}{cmd.name}</b>")
        
        # Описание из doc или strings
        if cmd.doc and strings and cmd.doc in strings:
            lines.append(strings[cmd.doc])
        elif cmd.doc:
            lines.append(cmd.doc)
        
        # Алиасы
        if cmd.aliases:
            aliases_list = ", ".join([f"{self.prefix}{alias}" for alias in cmd.aliases])
            lines.append(f"\n<b>Aliases:</b> {aliases_list}")
        
        # Аргументы
        if len(cmd.args) > 2:  # Пропускаем param и account
            lines.append("\n<b>Arguments:</b>")
            for arg in cmd.args[2:]:
                arg_name = arg.name
                arg_type = str(arg.annotation).replace('typing.', '').replace('<class \'', '').replace('\'>', '')
                optional_marker = " (optional)" if arg.is_optional or arg.default != inspect.Parameter.empty else ""
                default_str = f" = {arg.default}" if arg.default != inspect.Parameter.empty else ""
                lines.append(f"  • <code>{arg_name}</code>: {arg_type}{optional_marker}{default_str}")
        
        # Подкоманды
        if cmd.has_subcommands():
            lines.append("\n<b>Subcommands:</b>")
            for subname in sorted(cmd.subcommands.keys()):
                lines.append(f"  • {self.prefix}{cmd.name} <code>{subname}</code>")
        
        # Usage
        usage = f"{self.prefix}{cmd.name}"
        if len(cmd.args) > 2:
            arg_names = []
            for arg in cmd.args[2:]:
                if arg.is_optional or arg.default != inspect.Parameter.empty:
                    arg_names.append(f"[{arg.name}]")
                else:
                    arg_names.append(f"<{arg.name}>")
            usage += " " + " ".join(arg_names)
        
        lines.insert(1, f"\n<b>Usage:</b> <code>{usage}</code>")
        
        return "\n".join(lines)
    
    def generate_all_commands_help(self, strings: Optional[Dict[str, str]] = None, plugin_instance=None) -> str:
        """Генерирует справку по всем командам"""
        lines = [f"<b>Available Commands (prefix: {self.prefix}):</b>\n"]
        
        for name in sorted(self.listeners.keys()):
            cmd = self.listeners[name]
            if not cmd.is_enabled(plugin_instance):
                continue
            
            # Короткое описание
            doc_text = ""
            if cmd.doc and strings and cmd.doc in strings:
                doc_text = strings[cmd.doc]
            elif cmd.doc:
                doc_text = cmd.doc
            
            # Первая строка описания
            if doc_text:
                doc_text = doc_text.split('\n')[0]
                if len(doc_text) > 60:
                    doc_text = doc_text[:57] + "..."
            
            lines.append(f"• {self.prefix}<code>{name}</code> - {doc_text}")
        
        return "\n".join(lines)
    
    def validate_command_name(self, name: str) -> Tuple[bool, str]:
        """
        Проверяет корректность имени команды.
        
        Returns:
            Tuple[bool, str]: (is_valid, error_message)
        """
        if not name:
            return False, "Command name cannot be empty"
        
        if ' ' in name:
            return False, "Command name cannot contain spaces"
        
        if not name.replace('_', '').isalnum():
            return False, "Command name must be alphanumeric (underscore allowed)"
        
        if name in self.listeners:
            return False, f"Command '{name}' already exists"
        
        if name in self.aliases:
            return False, f"Name '{name}' is already used as alias for '{self.aliases[name]}'"
        
        return True, ""
    
    def check_alias_conflicts(self) -> List[str]:
        """Проверяет конфликты алиасов с именами команд"""
        conflicts = []
        
        for alias, command_name in self.aliases.items():
            if alias in self.listeners and alias != command_name:
                conflicts.append(f"Alias '{alias}' conflicts with command '{alias}'")
        
        return conflicts
    
    def bulk_enable_commands(self, command_names: List[str], plugin_instance=None):
        """Включает несколько команд"""
        for name in command_names:
            self.set_command_enabled(name, True, plugin_instance)
    
    def bulk_disable_commands(self, command_names: List[str], plugin_instance=None):
        """Выключает несколько команд"""
        for name in command_names:
            self.set_command_enabled(name, False, plugin_instance)
    
    def enable_all_commands(self, plugin_instance=None):
        """Включает все команды"""
        for name in self.listeners.keys():
            self.set_command_enabled(name, True, plugin_instance)
    
    def disable_all_commands(self, plugin_instance=None):
        """Выключает все команды"""
        for name in self.listeners.keys():
            self.set_command_enabled(name, False, plugin_instance)
    
    def clear_all_commands(self):
        """Удаляет все команды и алиасы"""
        self.listeners.clear()
        self.aliases.clear()
        logger.info(f"{self.plugin_id} dp: Cleared all commands.")
    
    def add_before_hook(self, hook: Callable[[CommandContext], Optional[Any]]):
        """
        Добавляет middleware хук, который выполняется перед командой.
        Если хук возвращает не None, выполнение команды прерывается и возвращается результат хука.
        """
        self.before_hooks.append(hook)
    
    def add_after_hook(self, hook: Callable[[CommandContext, Any], None]):
        """
        Добавляет middleware хук, который выполняется после команды.
        Получает контекст и результат выполнения команды.
        """
        self.after_hooks.append(hook)
    
    def remove_before_hook(self, hook: Callable):
        """Удаляет before хук"""
        if hook in self.before_hooks:
            self.before_hooks.remove(hook)
    
    def remove_after_hook(self, hook: Callable):
        """Удаляет after хук"""
        if hook in self.after_hooks:
            self.after_hooks.remove(hook)
    
    def clear_hooks(self):
        """Очищает все хуки"""
        self.before_hooks.clear()
        self.after_hooks.clear()
    
    def register_help_command(self, strings: Optional[Dict[str, str]] = None):
        """
        Регистрирует встроенную help команду.
        
        Args:
            strings: Словарь локализации (опционально)
        """
        @self.register_command("help")
        def help_cmd(param: Any, account: Any, command_name: Optional[str] = None) -> HookResult:
            """Показывает справку по командам"""
            if command_name:
                # Справка по конкретной команде
                help_text = self.generate_help_text(command_name, strings)
            else:
                # Общая справка
                help_text = self.generate_all_commands_help(strings, None)
            
            # Возвращаем HookResult с текстом справки
            return HookResult.from_string(help_text)
        
        logger.info(f"{self.plugin_id} dp: Registered built-in help command.")

# ==================== Decorators ====================
def command(cmd: Optional[str] = None, *, aliases: Optional[List[str]] = None, doc: Optional[str] = None, enabled: Optional[Union[str, bool]] = None):
    def decorator(func):
        func.__is_command__ = True
        func.__aliases__ = aliases or []
        func.__cdoc__ = doc
        func.__enabled__ = enabled
        func.__cmd__ = cmd or func.__name__
        return func
    return decorator

# Декораторы uri и message_uri определены выше (строки 570-597)
# Удалены дубликаты

def watcher():
    def decorator(func):
        func.__is_watcher__ = True
        return func
    return decorator

def inline_handler(method: str, support_long_click: bool = False):
    def decorator(func):
        func.__is_inline_handler__ = True
        func.__method__ = method
        func.__support_long__ = support_long_click
        return func
    return decorator

# ==================== JsonDB - JSON-based database ====================
class JsonDB(dict):
    def __init__(self, filepath: str):
        super().__init__()
        self.filepath = filepath
        self._load()
    
    def _load(self):
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.update(data)
            except Exception as e:
                logger.error(f"Failed to load JsonDB from {self.filepath}: {format_exc_only(e)}")
    
    def save(self):
        try:
            dir_path = os.path.dirname(self.filepath)
            if dir_path:  # Only create if not empty string
                os.makedirs(dir_path, exist_ok=True)
            with open(self.filepath, 'w', encoding='utf-8') as f:
                json.dump(dict(self), f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Failed to save JsonDB to {self.filepath}: {format_exc_only(e)}")
    
    def set(self, key: str, value: Any):
        self[key] = value
        self.save()
    
    def reset(self):
        self.clear()
        self.save()
    
    def pop(self, key: str, default: Any = None) -> Any:
        value = self.get(key, default)
        if key in self:
            del self[key]
            self.save()
        return value
    
    def update_from(self, **kwargs):
        self.update(kwargs)
        self.save()

# ==================== CommandParams helper (как в CactusLib) ====================
@dataclass
class CommandParams:
    """Обёртка для параметров команды с методами html()/markdown()"""
    text: str
    entities: List[Any]  # TLRPC entities
    peer: int
    replyToMsg: Optional[Any] = None
    replyToTopMsg: Optional[Any] = None
    
    def html(self) -> str:
        """Возвращает HTML-представление сообщения"""
        # Конвертируем TLRPC entities в RawEntity
        raw_entities = []
        for entity in self.entities:
            entity_type = None
            extra = None
            
            if isinstance(entity, TLRPC.TL_messageEntityBold):
                entity_type = TLEntityType.BOLD
            elif isinstance(entity, TLRPC.TL_messageEntityItalic):
                entity_type = TLEntityType.ITALIC
            elif isinstance(entity, TLRPC.TL_messageEntityUnderline):
                entity_type = TLEntityType.UNDERLINE
            elif isinstance(entity, TLRPC.TL_messageEntityStrike):
                entity_type = TLEntityType.STRIKETHROUGH
            elif isinstance(entity, TLRPC.TL_messageEntityCode):
                entity_type = TLEntityType.CODE
            elif isinstance(entity, TLRPC.TL_messageEntityPre):
                entity_type = TLEntityType.PRE
                extra = getattr(entity, 'language', None)
            elif isinstance(entity, TLRPC.TL_messageEntityTextUrl):
                entity_type = TLEntityType.TEXT_LINK
                extra = entity.url
            elif isinstance(entity, TLRPC.TL_messageEntitySpoiler):
                entity_type = TLEntityType.SPOILER
            elif isinstance(entity, TLRPC.TL_messageEntityCustomEmoji):
                entity_type = TLEntityType.CUSTOM_EMOJI
                extra = str(entity.document_id)
            elif isinstance(entity, TLRPC.TL_messageEntityBlockquote):
                entity_type = TLEntityType.BLOCKQUOTE
                extra = getattr(entity, 'collapsed', None)
            
            if entity_type:
                raw_entities.append(RawEntity(
                    type=entity_type,
                    offset=entity.offset,
                    length=entity.length,
                    extra=extra
                ))
        
        return HTML.unparse(self.text, raw_entities)
    
    def markdown(self) -> str:
        """Возвращает Markdown-представление сообщения"""
        # Конвертируем TLRPC entities в RawEntity (аналогично html())
        raw_entities = []
        for entity in self.entities:
            entity_type = None
            extra = None
            
            if isinstance(entity, TLRPC.TL_messageEntityBold):
                entity_type = TLEntityType.BOLD
            elif isinstance(entity, TLRPC.TL_messageEntityItalic):
                entity_type = TLEntityType.ITALIC
            elif isinstance(entity, TLRPC.TL_messageEntityUnderline):
                entity_type = TLEntityType.UNDERLINE
            elif isinstance(entity, TLRPC.TL_messageEntityStrike):
                entity_type = TLEntityType.STRIKETHROUGH
            elif isinstance(entity, TLRPC.TL_messageEntityCode):
                entity_type = TLEntityType.CODE
            elif isinstance(entity, TLRPC.TL_messageEntityPre):
                entity_type = TLEntityType.PRE
            elif isinstance(entity, TLRPC.TL_messageEntityTextUrl):
                entity_type = TLEntityType.TEXT_LINK
                extra = entity.url
            elif isinstance(entity, TLRPC.TL_messageEntitySpoiler):
                entity_type = TLEntityType.SPOILER
            elif isinstance(entity, TLRPC.TL_messageEntityCustomEmoji):
                entity_type = TLEntityType.CUSTOM_EMOJI
                extra = str(entity.document_id)
            
            if entity_type:
                raw_entities.append(RawEntity(
                    type=entity_type,
                    offset=entity.offset,
                    length=entity.length,
                    extra=extra
                ))
        
        return Markdown.unparse(self.text, raw_entities)
    
    def answer(self, text: str, **kwargs):
        """Отправляет ответное сообщение"""
        # Используем глобальную функцию send_message_with_entities
        return send_message_with_entities(
            self.peer, 
            text,
            replyToTopMsg=self.replyToTopMsg,
            replyToMsg=self.replyToMsg,
            **kwargs
        )

# ==================== Message sending utilities (как в CactusLib) ====================
def send_message_with_entities(
    peer: int,
    text: str,
    *,
    parse_message: bool = True,
    parse_mode: str = "HTML",
    markup: Optional[Any] = None,
    on_sent: Optional[Callable] = None,
    **kwargs
):
    """
    Отправляет сообщение с поддержкой HTML/Markdown и on_sent callback
    
    Args:
        peer: ID чата
        text: Текст сообщения
        parse_message: Парсить HTML/Markdown
        parse_mode: "HTML" или "MARKDOWN"
        markup: Inline.Markup объект
        on_sent: Callback после отправки
        **kwargs: Дополнительные параметры
    """
    try:
        entities = None
        
        if parse_message:
            if parse_mode.upper() == "HTML":
                parsed = HTML.parse(text)
                text = parsed.text
                entities = [e.to_tlrpc_object() for e in parsed.entities]
            elif parse_mode.upper() == "MARKDOWN":
                parsed = Markdown.parse(text)
                text = parsed.text
                entities = [e.to_tlrpc_object() for e in parsed.entities]
        
        # Создаём params dict для send_message
        params = {
            "peer": peer,
            "message": text,
            **kwargs
        }
        
        if entities:
            params["entities"] = entities
        
        if markup:
            params["reply_markup"] = markup.to_tlrpc() if hasattr(markup, 'to_tlrpc') else markup
        
        # Если есть on_sent callback, нужно обработать после отправки
        if on_sent:
            # Создаём обёртку которая вызовет callback после отправки
            def send_with_callback():
                # Отправляем сообщение
                msg_obj = send_message(params)
                
                # Вызываем on_sent с объектом похожим на CallbackParams
                if msg_obj:
                    try:
                        # Создаём псевдо CallbackParams для совместимости
                        callback_params = type('CallbackParams', (), {
                            'message': msg_obj,
                            'cell': None,
                            'edit': lambda self, txt, **kw: None,  # stub
                            'delete': lambda self: None  # stub
                        })()
                        on_sent(callback_params)
                    except Exception as e:
                        logger.error(f"Error in on_sent callback: {format_exc_only(e)}")
            
            # Запускаем в UI потоке
            run_on_ui_thread(send_with_callback)
        else:
            # Просто отправляем
            send_message(params)
            
    except Exception as e:
        logger.error(f"Failed to send message: {format_exc()}")

def edit_message(message_object: MessageObject, text: str, **kwargs):
    """Редактирует сообщение"""
    try:
        parse_message = kwargs.pop('parse_message', True)
        parse_mode = kwargs.pop('parse_mode', 'HTML')
        markup = kwargs.pop('markup', None)
        
        entities = None
        if parse_message:
            if parse_mode.upper() == "HTML":
                parsed = HTML.parse(text)
                text = parsed.text
                entities = [e.to_tlrpc_object() for e in parsed.entities]
            elif parse_mode.upper() == "MARKDOWN":
                parsed = Markdown.parse(text)
                text = parsed.text
                entities = [e.to_tlrpc_object() for e in parsed.entities]
        
        Requests.edit_message(
            message_object,
            text,
            entities=list_to_arraylist(entities) if entities else None,
            markup=markup.to_tlrpc() if markup and hasattr(markup, 'to_tlrpc') else markup,
            **kwargs
        )
    except Exception as e:
        logger.error(f"Failed to edit message: {format_exc_only(e)}")

def edit_message_markup(cell: Any, markup: Optional[Any]):
    """Редактирует inline-клавиатуру сообщения"""
    try:
        from org.telegram.ui.Cells import ChatMessageCell  # type: ignore
        
        if not isinstance(cell, ChatMessageCell):
            logger.warning("cell is not ChatMessageCell, cannot edit markup")
            return
        
        message = cell.getMessageObject()
        if not message or not message.messageOwner:
            logger.warning("Message or messageOwner is None")
            return
        
        if markup is None:
            # Удаляем клавиатуру
            message.messageOwner.reply_markup = None
            # Обновляем UI
            try:
                cell.setMessageObject(message, message.getGroupId(), False, False)
            except Exception as e:
                logger.error(f"Failed to update cell after markup removal: {format_exc_only(e)}")
        else:
            # Обновляем клавиатуру
            if hasattr(markup, '_markup'):
                # Inline.Markup объект
                tlrpc_markup = markup._markup
            elif hasattr(markup, 'to_tlrpc'):
                # Старый Inline.Markup с методом to_tlrpc
                tlrpc_markup = markup.to_tlrpc()
            else:
                # Уже TLRPC объект
                tlrpc_markup = markup
            
            # Устанавливаем новую разметку
            message.messageOwner.reply_markup = tlrpc_markup
            
            # Обновляем UI
            try:
                cell.setMessageObject(message, message.getGroupId(), False, False)
            except Exception as e:
                logger.error(f"Failed to update cell after markup change: {format_exc_only(e)}")
    except Exception as e:
        logger.error(f"Failed to edit markup: {format_exc_only(e)}")

def answer_file(
    params: CommandParams,
    path: str,
    caption: Optional[str] = None,
    *,
    parse_markdown: bool = True,
    **kwargs
):
    """
    Отправляет файл как документ (как в CactusLib)
    
    Args:
        params: CommandParams объект
        path: Путь к файлу
        caption: Подпись к файлу
        parse_markdown: Парсить Markdown в подписи
        **kwargs: Дополнительные параметры
    """
    try:
        from java.io import File as JFile # type: ignore
        from org.telegram.messenger import SendMessagesHelper # type: ignore
        
        file = JFile(path)
        if not file.exists():
            logger.error(f"File not found: {path}")
            return
        
        # Подготавливаем подпись
        caption_entities = None
        if caption and parse_markdown:
            try:
                parsed = Markdown.parse(caption)
                caption = parsed.text
                caption_entities = [e.to_tlrpc_object() for e in parsed.entities]
            except Exception as e:
                logger.error(f"Failed to parse caption: {format_exc_only(e)}")
        
        # Отправляем файл через SendMessagesHelper
        try:
            helper = get_messages_controller().getSendMessagesHelper()
            
            # Создаём ArrayList для файлов
            files = ArrayList()
            files.add(file.getAbsolutePath())
            
            # Отправляем
            helper.sendMessage(
                SendMessagesHelper.SendMessageParams.of(
                    files,  # paths
                    caption,  # caption
                    params.peer,  # peer
                    params.replyToMsg,  # replyToMsg  
                    params.replyToTopMsg,  # replyToTopMsg
                    None,  # webPage
                    True,  # searchLinks
                    caption_entities,  # entities
                    None,  # replyMarkup
                    None,  # params
                    True,  # notify
                    0,  # scheduleDate
                    None,  # quickReplyShortcut
                    None  # effectId
                )
            )
            
            logger.info(f"File sent: {path}")
        except Exception as e:
            logger.error(f"Failed to send file via helper: {format_exc_only(e)}")
            # Fallback - попробуем другой способ
            logger.warning("Trying fallback method...")
            
    except Exception as e:
        logger.error(f"Failed to send file: {format_exc()}")

# ==================== Inline buttons ====================
class Inline:

    callbacks: Dict[str, Callable] = {}
    
    class Markup:
        def __init__(self):
            self.rows = []
        
        def add_row(self, *buttons):
            if buttons and buttons[0] is not None:
                self.rows.append(list(buttons))
            return self
        
        def to_tlrpc(self) -> TLRPC.TL_replyInlineMarkup:
            markup = TLRPC.TL_replyInlineMarkup()
            markup.rows = ArrayList()
            
            for row in self.rows:
                tlrpc_row = TLRPC.TL_keyboardButtonRow()
                tlrpc_row.buttons = ArrayList()
                
                for btn in row:
                    if isinstance(btn, dict):
                        btn_type = btn.get('type', 'url')
                        
                        if btn_type == 'url':
                            button = TLRPC.TL_keyboardButtonUrl()
                            button.text = btn.get('text', '')
                            button.url = btn.get('url', '')
                        elif btn_type == 'callback':
                            button = TLRPC.TL_keyboardButtonCallback()
                            button.text = btn.get('text', '')
                            button.data = btn.get('callback_data', '').encode('utf-8')
                        else:
                            continue
                        
                        tlrpc_row.buttons.add(button)
                
                markup.rows.add(tlrpc_row)
            
            return markup
    
    @staticmethod
    def CallbackData(plugin_id: str, method: str, **kwargs) -> str:
        params = urlencode(kwargs)
        return f"mslib://{plugin_id}/{method}?{params}" if params else f"mslib://{plugin_id}/{method}"
    
    @staticmethod
    def button(text: str, *, url: Optional[str] = None, callback_data: Optional[str] = None, **kwargs):
        if url:
            return {'text': text, 'url': url, 'type': 'url'}
        elif callback_data:
            return {'text': text, 'callback_data': callback_data, 'type': 'callback'}
        return {'text': text}
    
    @classmethod
    def on_click(cls, method: str, support_long_click: bool = False):
        def decorator(func):
            cls.callbacks[method] = func
            func.__is_inline_handler__ = True
            func.__method__ = method
            func.__support_long__ = support_long_click
            return func
        return decorator

# ==================== Localization ====================
class Locales:
    en = {
        "copy_button": "Copy",
        "loaded": "MSLib loaded successfully!",
        "unloaded": "MSLib unloaded.",
        "error": "Error",
        "success": "Success",
        "info": "Info",
        "commands_header": "Commands",
        "command_prefix_label": "Command prefix",
        "command_prefix_hint": "Symbol used to trigger commands (e.g., . ! /)",
        "autoupdater_header": "AutoUpdater",
        "enable_autoupdater": "Enable AutoUpdater",
        "force_update_check": "Force update check",
        "autoupdate_timeout": "Update check interval (seconds)",
        "autoupdate_timeout_title": "Update check interval",
        "autoupdate_timeout_hint": "Time between update checks",
        "autoupdater_started": "AutoUpdater started!",
        "autoupdater_already_running": "AutoUpdater already running",
        "autoupdater_stopped": "AutoUpdater stopped",
        "autoupdater_already_stopped": "AutoUpdater already stopped",
        "command_prefix_updated": "Command prefix updated to: {prefix}",
        "invalid_prefix": "Invalid prefix! Must be a single non-alphanumeric character",
        "autoupdater_not_running": "AutoUpdater is not running. Enable it first!",
        "disable_timestamp_check_title": "Disable message edit check",
        "disable_timestamp_check_hint": "Plugin will be updated even if the file has not been modified",
        "addons_header": "Additional Features",
        "addon_article_viewer_fix": "Disable swipe-to-close gesture in browser",
        "addon_no_call_confirmation": "No Call Confirmation",
        "addon_old_bottom_forward": "Old Bottom Forward",
        "addon_hide_profile_edit": "Hide Profile Edit Button",
        "dev_header": "Developer",
        "debug_mode_title": "Debug mode",
        "debug_mode_hint": "Enables detailed logging for troubleshooting",
        "addon_article_viewer_fix_enabled": "Article Viewer Fix enabled",
        "addon_article_viewer_fix_disabled": "Article Viewer Fix disabled",
        "addon_no_call_confirmation_enabled": "No Call Confirmation enabled",
        "addon_no_call_confirmation_disabled": "No Call Confirmation disabled",
        "addon_old_bottom_forward_enabled": "Old Bottom Forward enabled",
        "addon_old_bottom_forward_disabled": "Old Bottom Forward disabled",
        "addon_hide_profile_edit_enabled": "Profile edit button hidden",
        "addon_hide_profile_edit_disabled": "Profile edit button visible",
        "update_check_started": "Update check started!",
        "autoupdater_not_initialized": "AutoUpdater is not initialized",
    }
    ru = {
        "copy_button": "Копировать",
        "loaded": "MSLib успешно загружена!",
        "unloaded": "MSLib выгружена.",
        "error": "Ошибка",
        "success": "Успешно",
        "info": "Информация",
        "commands_header": "Команды",
        "command_prefix_label": "Префикс команд",
        "command_prefix_hint": "Символ для вызова команд (например, . ! /)",
        "autoupdater_header": "Автообновления",
        "enable_autoupdater": "Автообновление",
        "force_update_check": "Принудительная проверка",
        "autoupdate_timeout": "Интервал проверки обновлений (секунды)",
        "autoupdate_timeout_title": "Интервал проверки обновлений",
        "autoupdate_timeout_hint": "Время между проверками обновлений",
        "autoupdater_started": "Автообновление запущено!",
        "autoupdater_already_running": "Автообновление уже запущено",
        "autoupdater_stopped": "Автообновление остановлено",
        "autoupdater_already_stopped": "Автообновление уже остановлено",
        "command_prefix_updated": "Префикс команд изменён на: {prefix}",
        "invalid_prefix": "Неверный префикс! Должен быть один не-буквенно-цифровой символ",
        "autoupdater_not_running": "Автообновление не запущено. Сначала включите его!",
        "disable_timestamp_check_title": "Отключить проверку изменений",
        "disable_timestamp_check_hint": "Плагин будет обновлён, даже если файл не был изменён",
        "addons_header": "Дополнительные функции",
        "addon_article_viewer_fix": "Отключить свайп в браузере",
        "addon_no_call_confirmation": "Без подтверждения звонка",
        "addon_old_bottom_forward": "Старое меню пересылки",
        "addon_hide_profile_edit": "Скрыть кнопку редактирования профиля",
        "dev_header": "Разработчик",
        "debug_mode_title": "Режим отладки",
        "debug_mode_hint": "Включает подробное логирование для диагностики",
        "addon_article_viewer_fix_enabled": "Свайп в браузере отключён",
        "addon_article_viewer_fix_disabled": "Свайп в браузере включён",
        "addon_no_call_confirmation_enabled": "Подтверждение звонка отключено",
        "addon_no_call_confirmation_disabled": "Подтверждение звонка включено",
        "addon_old_bottom_forward_enabled": "Старое меню пересылки включено",
        "addon_old_bottom_forward_disabled": "Старое меню пересылки выключено",
        "addon_hide_profile_edit_enabled": "Кнопка редактирования профиля скрыта",
        "addon_hide_profile_edit_disabled": "Кнопка редактирования профиля показана",
        "update_check_started": "Проверка обновлений запущена!",
        "autoupdater_not_initialized": "AutoUpdater не инициализирован",
    }
    default = en

def localise(key: str) -> str:
    try:
        locale = LOCALE if LOCALE else "en"
        locale_dict = getattr(Locales, locale, Locales.default)
        return locale_dict.get(key, key)
    except Exception:
        return key

class CacheFile:
    def __init__(self, filename: str, read_on_init: bool = True, compress: bool = False):
        self.filename = filename
        self.path = None
        self._content: Optional[bytes] = None
        self.compress = compress
        self.logger = build_log(f"{__name__}.{self.filename}")
        self.read_on_init = read_on_init
    
    def _ensure_path(self):
        if self.path is None and CACHE_DIRECTORY:
            self.path = os.path.join(CACHE_DIRECTORY, self.filename)
            try:
                os.makedirs(CACHE_DIRECTORY, exist_ok=True)
            except Exception as e:
                self.logger.error(f"Failed to create cache directory: {format_exc_only(e)}")
            if self.read_on_init:
                self.read()
    
    def read(self):
        self._ensure_path()
        if not self.path or not os.path.exists(self.path):
            if self.path:
                self.logger.warning(f"{self.path} does not exist, setting value to None.")
            self._content = None
            return
        
        try:
            with open(self.path, "rb") as file:
                file_content = file.read()
            
            if self.compress and file_content.startswith(b"\x78\x9c"):
                file_content = zlib.decompress(file_content)
            
            self._content = file_content
        except Exception as e:
            self.logger.error(f"Failed to load data from {self.path}: {format_exc_only(e)}")
            self._content = None
    
    def write(self):
        self._ensure_path()
        if not self.path:
            return
        try:
            save_data = self._content
            if self.compress and save_data:
                save_data = zlib.compress(save_data, level=6)
            
            with open(self.path, "wb") as file:
                file.write(save_data)
        except PermissionError as e:
            self.logger.error(f"No permission to edit {self.path}: {format_exc_only(e)}")
        except Exception as e:
            self.logger.error(f"Error writing to {self.path}: {format_exc_only(e)}")
    
    def delete(self):
        self._ensure_path()
        if not self.path or not os.path.exists(self.path):
            if self.path:
                self.logger.warning(f"File {self.path} does not exist.")
            return
        
        try:
            os.remove(self.path)
            self.logger.info(f"File {self.path} deleted.")
        except Exception as e:
            self.logger.error(f"Failed to delete {self.path}: {format_exc_only(e)}")
    
    @property
    def content(self) -> Optional[bytes]:
        return self._content
    
    @content.setter
    def content(self, value: Optional[bytes]):
        self._content = value

class JsonCacheFile(CacheFile):
    def __init__(self, filename: str, default: Any, read_on_init: bool = True, compress: bool = False):
        self._default = copy.deepcopy(default)
        self.json_content = self._get_copy_of_default()
        super().__init__(filename, read_on_init, compress)
    
    def _get_copy_of_default(self) -> Any:
        return copy.deepcopy(self._default)
    
    def read(self):
        super().read()
        
        if not self._content:
            self.json_content = self._get_copy_of_default()
            self._content = json.dumps(self.json_content).encode()
            return
        
        try:
            self.json_content = json.loads(self._content.decode("utf-8", errors="replace"))
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            self.logger.error(f"Failed to load JSON from {self.path}: {format_exc_only(e)}")
            self.json_content = self._get_copy_of_default()
    
    def write(self):
        self._content = json.dumps(self.json_content, ensure_ascii=False, indent=2).encode("utf-8")
        super().write()
    
    def wipe(self):
        self.json_content = self._get_copy_of_default()
        self._content = json.dumps(self.json_content).encode()
        self.write()
    
    @property
    def content(self) -> Any:
        if self._content is None:
            return self._get_copy_of_default()
        return self.json_content
    
    @content.setter
    def content(self, value: Any):
        self.json_content = value

# ==================== Dynamic Proxy Generator (like CactusUtils.gen) ====================
def gen(java_class, method_name: str, return_value: bool = False):
    """
    Генерирует динамический прокси-класс, который переопределяет указанный метод.
    
    Args:
        java_class: Java-класс для создания прокси
        method_name: Имя метода для переопределения
        return_value: Если True, возвращает значение из оригинального вызова
    
    Returns:
        Класс прокси, который можно инстанцировать с функцией
    """
    def __init__(self, fn: callable, *args, **kwargs):
        super(NewClass, self).__init__()
        self._fn = fn
        self._args = args
        self._kwargs = kwargs

    def _run(self, *method_args):
        try:
            result = self._fn(*method_args, *self._args, **self._kwargs)
            return result if return_value else None
        except Exception:
            logger.error(f"Error in gen proxy: {format_exc()}")
            return None
    
    NewClass = type(
        f'GeneratedProxy_{java_class.__name__}',
        (dynamic_proxy(java_class),),
        {
            '__init__': __init__,
            method_name: _run,
        }
    )
    
    return NewClass

def gen2(java_class, return_value: bool = False, **methods):
    """
    Генерирует динамический прокси-класс с несколькими переопределёнными методами.
    
    Args:
        java_class: Java-класс для создания прокси
        return_value: Если True, методы возвращают свои значения
        **methods: Именованные методы {имя_метода: функция}
    
    Returns:
        Класс прокси, который можно инстанцировать с аргументами
    """
    def __init__(self, *args, **kwargs):
        super(NewClass, self).__init__()
        self._args = args
        self._kwargs = kwargs

    method_dict = {'__init__': __init__}
    
    for method_name, method_fn in methods.items():
        def _run(self, *method_args, fn=method_fn):
            try:
                result = fn(*method_args, *self._args, **self._kwargs)
                return result if return_value else None
            except Exception:
                logger.error(f"Error in gen2 proxy method {method_name}: {format_exc()}")
                return None
        
        method_dict[method_name] = _run
    
    NewClass = type(
        f'GeneratedMultiProxy_{java_class.__name__}',
        (dynamic_proxy(java_class),),
        method_dict
    )
    
    return NewClass

# ==================== Callback wrappers ====================
class Callback1(dynamic_proxy(Utilities.Callback)):
    def __init__(self, fn: Callable[[Any], None], *args, **kwargs):
        super().__init__()
        self._fn = fn
        self._args = args
        self._kwargs = kwargs
    
    def run(self, arg):
        try:
            self._fn(arg, *self._args, **self._kwargs)
        except Exception as e:
            logger.error(f"Error in Callback1: {format_exc()}")

class Callback2(dynamic_proxy(Utilities.Callback2)):
    def __init__(self, fn: Callable[[Any, Any], None], *args, **kwargs):
        super().__init__()
        self._fn = fn
        self._args = args
        self._kwargs = kwargs
    
    def run(self, arg1, arg2):
        try:
            self._fn(arg1, arg2, *self._args, **self._kwargs)
        except Exception as e:
            logger.error(f"Error in Callback2: {format_exc()}")

class Callback3(dynamic_proxy(Utilities.Callback)):
    def __init__(self, fn: Callable[[Any, Any, Any], None], *args, **kwargs):
        super().__init__()
        self._fn = fn
        self._args = args
        self._kwargs = kwargs
    
    def run(self, arg1, arg2, arg3):
        try:
            self._fn(arg1, arg2, arg3, *self._args, **self._kwargs)
        except Exception as e:
            logger.error(f"Error in Callback3: {format_exc()}")

class Callback5(dynamic_proxy(Utilities.Callback5)):
    def __init__(self, fn: Callable[[Any, Any, Any, Any, Any], None], *args, **kwargs):
        super().__init__()
        self._fn = fn
        self._args = args
        self._kwargs = kwargs
    
    def run(self, arg1, arg2, arg3, arg4, arg5):
        try:
            self._fn(arg1, arg2, arg3, arg4, arg5, *self._args, **self._kwargs)
        except Exception as e:
            logger.error(f"Error in Callback5: {format_exc()}")

# ==================== Text Utilities ====================
def escape_html(text: str) -> str:
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

def unescape_html(text: str) -> str:
    return html.unescape(text)

def format_size(size_bytes: int) -> str:
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"

def format_duration(seconds: int) -> str:
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"

# compress_and_encode и decode_and_decompress определены выше (строка 531)
# Удалён дубликат

# ==================== Clipboard Utilities ====================
def copy_to_clipboard(text: str, show_bulletin: bool = True) -> bool:
    success = AndroidUtilities.addToClipboard(text)
    if success and show_bulletin:
        BulletinHelper.show_copied_to_clipboard()
    return success

# ==================== Bulletin Helper ====================
class InnerBulletinHelper(_BulletinHelper):
    def __init__(self, prefix: Optional[str] = None):
        self.prefix = "" if not prefix or not prefix.strip() else f"{prefix}:"
    
    def show_info(self, message: str, fragment: Optional[Any] = None):
        _BulletinHelper.show_info(f"{self.prefix} {message}", fragment)
    
    def show_error(self, message: str, fragment: Optional[Any] = None):
        _BulletinHelper.show_error(f"{self.prefix} {message}", fragment)
    
    def show_success(self, message: str, fragment: Optional[Any] = None):
        _BulletinHelper.show_success(f"{self.prefix} {message}", fragment)
    
    def show_with_copy(self, message: str, text_to_copy: str, icon_res_id: int):
        _BulletinHelper.show_with_button(
            f"{self.prefix} {message}" if not message.startswith(f"{self.prefix} ") else message,
            icon_res_id,
            localise("copy_button"),
            on_click=lambda: copy_to_clipboard(text_to_copy, show_bulletin=False),
        )
    
    def show_info_with_copy(self, message: str, copy_text: str):
        self.show_with_copy(f"{self.prefix} {message}", str(copy_text), R.raw.info)
    
    def show_error_with_copy(self, message: str, copy_text: str):
        self.show_with_copy(f"{self.prefix} {message}", str(copy_text), R.raw.error)
    
    def show_success_with_copy(self, message: str, copy_text: str):
        self.show_with_copy(f"{self.prefix} {message}", str(copy_text), R.raw.contact_check)
    
    def show_with_post_redirect(self, message: str, button_text: str, peer_id: int, message_id: int, icon_res_id: int = 0):
        _BulletinHelper.show_with_button(
            f"{self.prefix} {message}",
            icon_res_id,
            button_text,
            on_click=lambda: get_last_fragment().presentFragment(ChatActivity.of(peer_id, message_id)),
        )
    
    def show_info_with_post_redirect(self, message: str, button_text: str, peer_id: int, message_id: int):
        self.show_with_post_redirect(message, button_text, peer_id, message_id, R.raw.info)
    
    def show_error_with_post_redirect(self, message: str, button_text: str, peer_id: int, message_id: int):
        self.show_with_post_redirect(message, button_text, peer_id, message_id, R.raw.error)
    
    def show_success_with_post_redirect(self, message: str, button_text: str, peer_id: int, message_id: int):
        self.show_with_post_redirect(message, button_text, peer_id, message_id, R.raw.contact_check)

def build_bulletin_helper(prefix: Optional[str] = None) -> InnerBulletinHelper:
    return InnerBulletinHelper(prefix)

BulletinHelper = build_bulletin_helper(__name__)

def _bulletin(level: str, message: str):
    getattr(BulletinHelper, f"show_{level}")(message, None)

# ==================== AutoUpdater ====================
class UpdaterTask:
    def __init__(self, plugin_id, channel_id, message_id):
        self.plugin_id = plugin_id
        self.channel_id = channel_id
        self.message_id = message_id

class AutoUpdater:
    def __init__(self, plugin_instance: Optional['MSLib'] = None):
        self.plugin_instance = plugin_instance
        self.thread: Optional[threading.Thread] = None
        self.forced_stop = False
        self.forced_update_check = False
        self.tasks: List[UpdaterTask] = []
        self.msg_edited_ts_cache = JsonCacheFile("mslib_au__msg_edited_ts", {})
        self._lock = threading.RLock()
        self.hash = str(zlib.adler32(id(self).to_bytes(8, "little")))
        self.logger = build_log(f"{__name__} AU {self.hash}")
    
    def run(self):
        self.forced_stop = False
        
        if self.thread is None:
            self.thread = threading.Thread(target=self.cycle, daemon=True)
        
        if self.thread.is_alive():
            self.logger.warning("Thread is already running.")
            return
        
        self.thread.start()
        self.logger.info("Thread was started.")
    
    def force_stop(self):
        if self.thread is None:
            self.logger.warning("Thread is not running.")
            return
        self.forced_stop = True
    
    def cycle(self):
        event = threading.Event()
        event.wait(5)
        
        while not self.forced_stop:
            try:
                self.check_for_updates(show_notifications=False)
                start_time = time.time()
                timeout = self.get_timeout_time()
                
                while (time.time() - start_time) < timeout:
                    event.wait(1)
                    
                    if self.forced_update_check:
                        self.logger.info("Forced update check requested, checking immediately...")
                        self.check_for_updates(show_notifications=True)
                        self.forced_update_check = False
                    
                    if self.forced_stop:
                        break
                    
                    if (time.time() - start_time) >= timeout:
                        break
                        
            except KeyboardInterrupt:
                self.logger.info("Received keyboard interrupt, stopping...")
                break
            except Exception as e:
                self.logger.error(f"Exception in cycle: {format_exc_only(e)}")
                if not self.forced_stop:
                    event.wait(60)
        
        self.thread = None
        self.logger.info("Force stopped.")
    
    def check_for_updates(self, show_notifications: bool = False):
        self.logger.info(f"Checking for updates... (notifications: {show_notifications})")
        
        with self._lock:
            tasks_snapshot = list(self.tasks)

        for task in tasks_snapshot:
            try:
                plugin = get_plugin(task.plugin_id)
                
                if plugin is None:
                    self.logger.info(f"Plugin {task.plugin_id} not found. Removing task...")
                    self.remove_task(task)
                    continue
                
                if not plugin.isEnabled():
                    self.logger.info(f"Plugin {task.plugin_id} is disabled. Skipping update check...")
                    continue
                
                self.logger.info(f"Checking update for task {task.plugin_id}...")
                self._check_task_for_update(task, show_notifications)
                
            except Exception as e:
                self.logger.error(f"Error checking update for {task.plugin_id}: {format_exc_only(e)}")
    
    def _check_task_for_update(self, task: UpdaterTask, show_notifications: bool = False):
        def get_message_callback(msg):
            if not msg or isinstance(msg, TLRPC.TL_messageEmpty):
                self.logger.warning(f"Message not found for {task.plugin_id}. Removing task...")
                if show_notifications:
                    InnerBulletinHelper("MSLib").show_error(f"Update check failed: message not found for {task.plugin_id}")
                self.remove_task(task)
                return
            
            if not msg.media:
                self.logger.warning(f"Message has no media for {task.plugin_id}. Removing task...")
                if show_notifications:
                    InnerBulletinHelper("MSLib").show_error(f"Update check failed: no document for {task.plugin_id}")
                self.remove_task(task)
                return
            
            # Try to get from plugin settings first
            disable_ts_check = DEFAULT_DISABLE_TIMESTAMP_CHECK
            if self.plugin_instance:
                try:
                    disable_ts_check = self.plugin_instance.get_setting("disable_timestamp_check", DEFAULT_DISABLE_TIMESTAMP_CHECK)
                except Exception:
                    pass  # Fall back to default
            
            if not disable_ts_check:
                cache_key = f"{task.channel_id}_{task.message_id}"
                cached_edit_date = self.msg_edited_ts_cache.content.get(cache_key, 0)
                current_edit_date = msg.edit_date if msg.edit_date != 0 else msg.date
                
                if current_edit_date <= cached_edit_date:
                    self.logger.info(f"No updates for {task.plugin_id}")
                    if show_notifications:
                        InnerBulletinHelper("MSLib").show_info(f"{task.plugin_id}: Already up to date")
                    return
                
                self.logger.info(f"Update available for {task.plugin_id}: {cached_edit_date} -> {current_edit_date}")
                if show_notifications:
                    InnerBulletinHelper("MSLib").show_success(f"Update found for {task.plugin_id}")
                
                # Update cache under lock to avoid concurrent writes
                with self._lock:
                    self.msg_edited_ts_cache.content[cache_key] = current_edit_date
                    try:
                        self.msg_edited_ts_cache.write()
                    except Exception as e:
                        self.logger.error(f"Failed to write AU cache: {format_exc_only(e)}")
            else:
                self.logger.info(f"Timestamp check disabled, forcing update for {task.plugin_id}")
                if show_notifications:
                    InnerBulletinHelper("MSLib").show_info(f"Forcing update for {task.plugin_id}")
            
            run_on_queue(lambda: download_and_install_plugin(msg, task.plugin_id))
        
        Requests.get_message(
            task.channel_id,
            task.message_id,
            callback=lambda msg, process_task=task: get_message_callback(msg)
        )
    
    def is_task_already_present(self, task: UpdaterTask) -> bool:
        with self._lock:
            for t in self.tasks:
                if t.plugin_id == task.plugin_id:
                    return True
            return False
    
    def add_task(self, task: UpdaterTask):
        with self._lock:
            if self.is_task_already_present(task):
                self.logger.warning(f"Task {task.plugin_id} already exists.")
                return

            self.tasks.append(task)
            self.logger.info(f"Added task {task.plugin_id}.")
    
    def remove_task(self, task: UpdaterTask):
        with self._lock:
            if task not in self.tasks:
                self.logger.warning(f"Task {task.plugin_id} not found.")
                return

            try:
                self.tasks.remove(task)
                self.logger.info(f"Removed task {task.plugin_id}")
            except ValueError:
                self.logger.warning(f"Failed to remove task {task.plugin_id} - not found")
    
    def remove_task_by_id(self, plugin_id: str):
        with self._lock:
            filtered_tasks = [t for t in self.tasks if t.plugin_id != plugin_id]
            if len(filtered_tasks) < len(self.tasks):
                self.tasks = filtered_tasks
                self.logger.info(f"Removed task {plugin_id}")
            else:
                self.logger.warning(f"Task {plugin_id} not found.")
    
    def get_timeout_time(self) -> int:
        try:
            # Try to get from plugin settings first
            timeout_str = DEFAULT_AUTOUPDATE_TIMEOUT
            if self.plugin_instance:
                try:
                    timeout_str = self.plugin_instance.get_setting("autoupdate_timeout", DEFAULT_AUTOUPDATE_TIMEOUT)
                except Exception:
                    pass  # Fall back to default
            return int(timeout_str)
        except (ValueError, TypeError) as e:
            self.logger.error(f"Failed to get timeout: {format_exc_only(e)}")
            return int(DEFAULT_AUTOUPDATE_TIMEOUT)
    
    def force_update_check(self):
        self.logger.info("Forced update check was requested.")
        self.forced_update_check = True

# ==================== AutoUpdater Helper Features ====================
def download_and_install_plugin(msg, plugin_id: str, max_tries: int = 10, is_queued: bool = False, current_try: int = 0):
    def plugin_install_error(arg):
        if arg is None:
            return
        logger.error(f"Error installing {plugin_id}: {arg}")
        InnerBulletinHelper("MSLib").show_error(f"Error installing {plugin_id}. Check logs.")
    
    try:
        file_loader = get_file_loader()
        plugins_controller = PluginsController.getInstance()
        document = msg.media.getDocument()
        path = file_loader.getPathToAttach(document, True)
        
        if not path.exists():
            if is_queued:
                logger.info(f"Waiting 1s for {plugin_id} file to download ({current_try}/{max_tries})...")
            else:
                logger.info(f"Starting download of {plugin_id} plugin file...")
            
            file_loader.loadFile(document, msg, FileLoader.PRIORITY_NORMAL, 1)
            
            if current_try >= max_tries:
                logger.error(f"Max tries reached for {plugin_id}, installation aborted.")
                InnerBulletinHelper("MSLib").show_error(f"Failed to download {plugin_id}")
                return
            
            run_on_queue(
                lambda: download_and_install_plugin(msg, plugin_id, max_tries, True, current_try + 1),
                delay=1
            )
            return
        
        logger.info(f"Installing {plugin_id}...")
        
        try:
            plugins_controller.loadPluginFromFile(str(path), None, Callback1(plugin_install_error))
        except TypeError:
            plugins_controller.loadPluginFromFile(str(path), Callback1(plugin_install_error))
        
        logger.info(f"Plugin {plugin_id} installed successfully")
        InnerBulletinHelper("MSLib").show_success(f"{plugin_id} updated successfully!")
        
    except AttributeError as e:
        logger.error(f"AttributeError in download_and_install_plugin for {plugin_id}: {format_exc_only(e)}")
        InnerBulletinHelper("MSLib").show_error(f"Invalid message format for {plugin_id}")
    except Exception as e:
        logger.error(f"Error in download_and_install_plugin for {plugin_id}: {format_exc()}")
        InnerBulletinHelper("MSLib").show_error(f"Failed to install {plugin_id}: {format_exc_only(e)}")

def get_plugin(plugin_id: str):
    return PluginsController.getInstance().plugins.get(plugin_id)

def add_autoupdater_task(plugin_id: str, channel_id: int, message_id: int, updater: 'AutoUpdater'):
    if not updater:
        logger.warning("AutoUpdater is not initialized")
        return
    
    task = UpdaterTask(plugin_id, channel_id, message_id)
    updater.add_task(task)
    logger.info(f"Added autoupdate task for {plugin_id}: channel={channel_id}, message={message_id}")

def remove_autoupdater_task(plugin_id: str, updater: 'AutoUpdater'):
    if not updater:
        logger.warning("AutoUpdater is not initialized")
        return
    
    updater.remove_task_by_id(plugin_id)
    logger.info(f"Removed autoupdate task for {plugin_id}")

# ==================== Requests utilities ====================
def request_callback_factory(custom_callback: Optional[Callable]):
    def default_callback(response, error):
        if custom_callback:
            custom_callback(response, error)
        else:
            if error:
                logger.error(f"Request error: {error}")
    return default_callback

class Requests:
    @staticmethod
    def send(request: TLObject, callback: Optional[Callable] = None, account: int = 0):
        send_request(request, request_callback_factory(callback), account)
    
    @staticmethod
    def get_user(user_id: int, callback: Callable, account: int = 0):
        request = TLRPC.TL_users_getUsers()
        input_user = TLRPC.TL_inputUser()
        input_user.user_id = user_id
        request.id = ArrayList()
        request.id.add(input_user)
        
        def user_callback(response, error):
            if error or not response:
                callback(None, error)
            else:
                users = arraylist_to_list(response)
                callback(users[0] if users else None, error)
        
        Requests.send(request, user_callback, account)
    
    @staticmethod
    def get_chat(chat_id: int, callback: Callable, account: int = 0):
        request = TLRPC.TL_messages_getChats()
        request.id = ArrayList()
        request.id.add(Long(abs(chat_id)))
        
        def chat_callback(response, error):
            if error or not response:
                callback(None, error)
            else:
                chats = arraylist_to_list(response.chats)
                callback(chats[0] if chats else None, error)
        
        Requests.send(request, chat_callback, account)
    
    @staticmethod
    def get_message(channel_id: int, message_id: int, callback: Callable, account: int = 0):
        request = TLRPC.TL_channels_getMessages()
        # Normalize channel id: accept UI-style -100<id>, negative raw id, or positive raw id
        raw = channel_id
        try:
            if isinstance(channel_id, int) and channel_id < 0:
                av = abs(channel_id)
                if channel_id < -1000000000000:
                    raw = av - 1000000000000
                elif av > 1000000000:
                    raw = av
                else:
                    raw = av
        except Exception:
            raw = abs(channel_id)

        input_channel = TLRPC.TL_inputChannel()
        input_channel.channel_id = abs(int(raw))
        input_channel.access_hash = 0
        request.channel = input_channel
        request.id = ArrayList()
        input_message = TLRPC.TL_inputMessageID()
        input_message.id = message_id
        request.id.add(input_message)
        
        def message_callback(response, error):
            if error or not response:
                callback(None, error)
            else:
                messages = arraylist_to_list(response.messages) if hasattr(response, 'messages') else []
                callback(messages[0] if messages else None, error)
        
        Requests.send(request, message_callback, account)
    
    @staticmethod
    def search_messages(
        peer_id: int,
        query: str,
        callback: Callable,
        limit: int = 100,
        offset_id: int = 0,
        filter_type: Optional[Any] = None,
        account: int = 0
    ):
        request = TLRPC.TL_messages_search()
        request.peer = Requests._get_input_peer(peer_id)
        request.q = query
        request.filter = filter_type or TLRPC.TL_inputMessagesFilterEmpty()
        request.limit = limit
        request.offset_id = offset_id
        request.min_date = 0
        request.max_date = 0
        request.add_offset = 0
        request.max_id = 0
        request.min_id = 0
        request.hash = 0
        
        def search_callback(response, error):
            if error or not response:
                callback([], error)
            else:
                messages = arraylist_to_list(response.messages) if hasattr(response, 'messages') else []
                callback(messages, error)
        
        Requests.send(request, search_callback, account)
    
    @staticmethod
    def _get_input_peer(peer_id: int):
        # Handles multiple peer id formats used across the codebase:
        # - positive user ids
        # - chat ids as negative small ints (-12345)
        # - UI-style channels: -100<channel_id>
        # - raw channel ids (positive) or negative raw channel ids (-<channel_id>)
        if peer_id is None or peer_id == 0:
            return None

        try:
            if peer_id > 0:
                input_peer = TLRPC.TL_inputPeerUser()
                input_peer.user_id = peer_id
                input_peer.access_hash = 0
                return input_peer

            av = abs(int(peer_id))

            # UI-style channel id (-100<channel_id>)
            if peer_id < -1000000000000 or av > 1000000000:
                # treat as channel
                # if UI-style, strip the -100 prefix
                if peer_id < -1000000000000:
                    channel_id = av - 1000000000000
                else:
                    channel_id = av

                input_peer = TLRPC.TL_inputPeerChannel()
                input_peer.channel_id = int(channel_id)
                input_peer.access_hash = 0
                return input_peer

            # Fallback: small negative integers are chats
            input_peer = TLRPC.TL_inputPeerChat()
            input_peer.chat_id = av
            return input_peer
        except Exception:
            # As a last resort, return a chat peer with absolute id
            peer = TLRPC.TL_inputPeerChat()
            peer.chat_id = abs(int(peer_id))
            return peer
    
    @staticmethod
    def delete_messages(message_ids: List[int], peer_id: int, callback: Optional[Callable] = None, revoke: bool = True, account: int = 0):
        if peer_id < -1000000000000:
            request = TLRPC.TL_channels_deleteMessages()
            request.channel = Requests._get_input_peer(peer_id)
            request.id = list_to_arraylist(message_ids)
        else:
            request = TLRPC.TL_messages_deleteMessages()
            request.id = list_to_arraylist(message_ids)
            request.revoke = revoke
        
        Requests.send(request, callback, account)
    
    @staticmethod
    def ban(chat_id: int, peer_id: int, until_date: Optional[int] = None, callback: Optional[Callable] = None, account: int = 0):
        msg_controller = get_messages_controller()
        
        banned_rights = TLRPC.TL_chatBannedRights()
        banned_rights.view_messages = True
        banned_rights.send_messages = True
        banned_rights.send_media = True
        banned_rights.send_stickers = True
        banned_rights.send_gifs = True
        banned_rights.send_games = True
        banned_rights.send_inline = True
        banned_rights.embed_links = True
        banned_rights.send_polls = True
        banned_rights.change_info = True
        banned_rights.invite_users = True
        banned_rights.pin_messages = True
        banned_rights.until_date = until_date or 0
        
        request = TLRPC.TL_channels_editBanned()
        request.channel = msg_controller.getInputChannel(chat_id)
        request.participant = msg_controller.getInputPeer(peer_id)
        request.banned_rights = banned_rights
        
        Requests.send(request, callback, account)
    
    @staticmethod
    def unban(chat_id: int, target_peer_id: int, callback: Optional[Callable] = None, account: int = 0):
        msg_controller = get_messages_controller()
        
        banned_rights = TLRPC.TL_chatBannedRights()
        banned_rights.until_date = 0
        
        request = TLRPC.TL_channels_editBanned()
        request.channel = msg_controller.getInputChannel(chat_id)
        request.participant = msg_controller.getInputPeer(target_peer_id)
        request.banned_rights = banned_rights
        
        Requests.send(request, callback, account)
    
    @staticmethod
    def change_slowmode(chat_id: int, seconds: int = 0, callback: Optional[Callable] = None, account: int = 0):
        msg_controller = get_messages_controller()
        
        request = TLRPC.TL_channels_toggleSlowMode()
        request.channel = msg_controller.getInputChannel(chat_id)
        request.seconds = seconds
        
        Requests.send(request, callback, account)
    
    @staticmethod
    def reload_admins(chat_id: int, account: int = 0):
        get_messages_controller().loadChannelAdmins(chat_id, False)
    
    @staticmethod
    def get_chat_participant(chat_id: int, target_peer_id: int, callback: Callable, account: int = 0):
        msg_controller = get_messages_controller()
        
        request = TLRPC.TL_channels_getParticipant()
        request.channel = msg_controller.getInputChannel(chat_id)
        request.participant = msg_controller.getInputPeer(target_peer_id)
        
        Requests.send(request, callback, account)
    
    @staticmethod
    def edit_message(message_object: MessageObject, text: str, entities: Optional[ArrayList] = None, 
                     markup: Optional[Any] = None, callback: Optional[Callable] = None, account: int = 0):
        msg_controller = get_messages_controller()
        
        request = TLRPC.TL_messages_editMessage()
        request.peer = msg_controller.getInputPeer(message_object.getDialogId())
        request.id = message_object.getId()
        request.message = text
        request.no_webpage = True
        
        if entities:
            request.entities = entities
            request.flags |= 8
        
        if markup:
            request.reply_markup = markup
            request.flags |= 4
        
        Requests.send(request, callback, account)
    
    @staticmethod
    def forward_messages(from_peer: int, to_peer: int, message_ids: List[int], 
                        callback: Optional[Callable] = None, account: int = 0):
        msg_controller = get_messages_controller()
        
        request = TLRPC.TL_messages_forwardMessages()
        request.from_peer = msg_controller.getInputPeer(from_peer)
        request.to_peer = msg_controller.getInputPeer(to_peer)
        request.id = list_to_arraylist(message_ids)
        request.random_id = ArrayList()
        
        for _ in message_ids:
            request.random_id.add(Long(Utilities.random.nextLong()))
        
        Requests.send(request, callback, account)
    
    @staticmethod
    def get_full_user(user_id: int, callback: Callable, account: int = 0):
        msg_controller = get_messages_controller()
        
        request = TLRPC.TL_users_getFullUser()
        request.id = msg_controller.getInputUser(user_id)
        
        Requests.send(request, callback, account)
    
    @staticmethod
    def get_full_chat(chat_id: int, callback: Callable, account: int = 0):
        if chat_id < -1000000000000:
            # Channel
            request = TLRPC.TL_channels_getFullChannel()
            request.channel = get_messages_controller().getInputChannel(chat_id)
        else:
            # Chat
            request = TLRPC.TL_messages_getFullChat()
            request.chat_id = abs(chat_id)
        
        Requests.send(request, callback, account)

# ==================== System utilities ====================
def runtime_exec(cmd: List[str], return_list_lines: bool = False, raise_errors: bool = True) -> Union[List[str], str]:
    from java.lang import Runtime # type: ignore
    from java.io import BufferedReader, InputStreamReader, IOException # type: ignore
    
    result = []
    process = None
    reader = None
    try:
        process = Runtime.getRuntime().exec(cmd)
        reader = BufferedReader(InputStreamReader(process.getInputStream()))
        line = reader.readLine()
        while line is not None:
            result.append(str(line))
            line = reader.readLine()
    except IOException as e:
        if raise_errors:
            raise e
        logger.error(f"IOException in runtime_exec: {format_exc_only(e)}")
    except Exception as e:
        if raise_errors:
            raise e
        logger.error(f"Error in runtime_exec: {format_exc()}")
    finally:
        if reader:
            try:
                reader.close()
            except Exception:
                pass
        if process:
            try:
                process.destroy()
            except Exception:
                pass
    
    return result if return_list_lines else "\n".join(result)

def get_logs(__id__: Optional[str] = None, times: Optional[int] = None, lvl: Optional[str] = None, as_list: bool = False) -> Union[List[str], str]:
    cmd = ["logcat", "-d", "-v", "time"]
    
    if times:
        from java.lang import System as JavaSystem # type: ignore
        time_str = f"{times}s ago"
        cmd.extend(["-t", time_str])
    
    if lvl:
        cmd.extend(["*:{}".format(lvl)])
    
    result = runtime_exec(cmd, return_list_lines=True, raise_errors=False)
    
    if __id__:
        result = [line for line in result if f"[{__id__}]" in line]
    
    logger.debug(f"Got logs with {__id__=}, {times=}s, {lvl=}")
    
    return result if as_list else "\n".join(result)

# Удалён дублирующийся pluralization_string (основная версия в строке 79)

# ==================== UI utilities ====================
class UI:
    @staticmethod
    @run_on_ui_thread
    def show_alert(title: str, message: str, positive_button: str = "OK", on_click: Optional[Callable] = None):
        # Prefer current fragment's activity when available (safer in plugin context)
        frag = get_last_fragment()
        activity = None
        try:
            if frag:
                activity = frag.getParentActivity()
        except Exception:
            activity = None

        ctx = activity if activity is not None else ApplicationLoader.applicationContext

        builder = AlertDialogBuilder(ctx)
        builder.set_title(title)
        builder.set_message(message)
        builder.set_positive_button(positive_button, on_click)
        builder.show()
    
    @staticmethod
    @run_on_ui_thread
    def show_confirm(title: str, message: str, on_confirm: Callable, on_cancel: Optional[Callable] = None):
        frag = get_last_fragment()
        activity = None
        try:
            if frag:
                activity = frag.getParentActivity()
        except Exception:
            activity = None

        ctx = activity if activity is not None else ApplicationLoader.applicationContext

        builder = AlertDialogBuilder(ctx)
        builder.set_title(title)
        builder.set_message(message)
        builder.set_positive_button("OK", on_confirm)
        builder.set_negative_button("Cancel", on_cancel)
        builder.show()

class Spinner:
    def __init__(self, text: str = "Loading..."):
        self.text = text
        self.alert_dialog = None
        self._shown = False
    
    def show(self):
        if self._shown:
            return
        @run_on_ui_thread
        def _show():
            # Use AlertDialogBuilder wrapper to create a spinner dialog
            frag = get_last_fragment()
            activity = None
            try:
                if frag:
                    activity = frag.getParentActivity()
            except Exception:
                activity = None

            ctx = activity if activity is not None else ApplicationLoader.applicationContext

            builder = AlertDialogBuilder(ctx, AlertDialogBuilder.ALERT_TYPE_SPINNER)
            builder.set_title(self.text)
            builder.set_cancelable(False)
            shown = builder.show()
            # keep reference to builder for later dismissal
            self.alert_dialog = shown

        _show()
        self._shown = True
    
    def hide(self):
        if not self._shown:
            return
        @run_on_ui_thread
        def _hide():
            if self.alert_dialog:
                try:
                    # AlertDialogBuilder.show() returns builder-like object
                    # that supports dismiss()
                    self.alert_dialog.dismiss()
                except Exception:
                    pass

        _hide()
        self._shown = False
    
    def __enter__(self):
        self.show()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.hide()
        return False

# ==================== File System utilities (расширенный как в CactusLib) ====================
class FileSystem:
    """Файловая система с методами в стиле CactusLib"""
    
    @staticmethod
    def basedir(*path: str):
        """Возвращает базовый каталог приложения с опциональными подкаталогами"""
        from java.io import File # type: ignore
        _dir = ApplicationLoader.getFilesDirFixed()
        if path:
            for p in path:
                _dir = File(_dir, p)
                if not _dir.exists():
                    _dir.mkdirs()
        return _dir
    
    @staticmethod
    def cachedir(*path: str):
        """Возвращает внешний кэш-каталог приложения"""
        from java.io import File # type: ignore
        _dir = ApplicationLoader.applicationContext.getExternalCacheDir()
        if path:
            for p in path:
                _dir = File(_dir, p)
                if not _dir.exists():
                    _dir.mkdirs()
        return _dir
    
    @staticmethod
    def tempdir():
        """Возвращает временный каталог внутри кэша"""
        from java.io import File # type: ignore
        _dir = File(FileSystem.cachedir(), "mslib_temp_files")
        if not _dir.exists():
            _dir.mkdirs()
        return _dir
    
    @staticmethod
    def get_file_content(file_path, mode: str = "rb"):
        """Читает содержимое файла"""
        with open(file_path, mode) as f:
            return f.read()
    
    @staticmethod
    def get_temp_file_content(filename: str, mode: str = "rb", delete_after: int = 0):
        """Читает содержимое временного файла с опциональным удалением"""
        from java.io import File # type: ignore
        file_path = File(FileSystem.tempdir(), filename).getAbsolutePath()
        content = FileSystem.get_file_content(file_path, mode)
        if delete_after > 0:
            FileSystem.delete_file_after(file_path, delete_after)
        return content
    
    @staticmethod
    def write_file(file_path, content, mode: str = "wb"):
        """Записывает содержимое в файл"""
        with open(file_path, mode) as file:
            file.write(content)
        return file_path
    
    @staticmethod
    def write_temp_file(filename: str, content, mode="wb", delete_after: int = 0):
        """Записывает содержимое во временный файл"""
        from java.io import File # type: ignore
        path = FileSystem.write_file(File(FileSystem.tempdir(), filename).getAbsolutePath(), content, mode)
        if delete_after > 0:
            FileSystem.delete_file_after(path, delete_after)
        return path
    
    @staticmethod
    def delete_file_after(file_path, seconds: int = 0):
        """Удаляет файл после указанной задержки"""
        if os.path.exists(file_path):
            if seconds > 0:
                threading.Timer(seconds, lambda: os.remove(file_path) if os.path.exists(file_path) else None).start()
                return
            os.remove(file_path)
    
    # =========== Старые методы для обратной совместимости ===========
    @staticmethod
    def get_cache_dir(*path: str) -> str:
        if not CACHE_DIRECTORY:
            _init_constants()
        return os.path.join(CACHE_DIRECTORY, *path)
    
    @staticmethod
    def get_plugins_dir(*path: str) -> str:
        if not PLUGINS_DIRECTORY:
            _init_constants()
        return os.path.join(PLUGINS_DIRECTORY, *path)
    
    @staticmethod
    def read_file(filepath: str, mode: str = 'r', encoding: str = 'utf-8') -> Union[str, bytes]:
        try:
            with open(filepath, mode, encoding=encoding if 'b' not in mode else None) as f:
                return f.read()
        except Exception as e:
            logger.error(f"Failed to read file {filepath}: {format_exc_only(e)}")
            return '' if 'b' not in mode else b''
    
    @staticmethod
    def file_exists(filepath: str) -> bool:
        return os.path.exists(filepath)
    
    @staticmethod
    def get_file_size(filepath: str) -> int:
        try:
            return os.path.getsize(filepath)
        except Exception:
            return 0

# ==================== Singleton metaclass ====================
class SingletonMeta(type):
    _instances = {}
    
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

# ==================== Companion File System ====================
class Companion(metaclass=SingletonMeta):
    """Companion file mechanism for preserving state between plugin reloads."""
    defaults = {
        "autoupdates_tasks": [],
        "pending_commands": {},
        "dispatcher_prefixes": {},
    }
    
    def __init__(self):
        self.module = None
    
    @staticmethod
    def create():
        """Creates the companion Python module file."""
        if not COMPANION_PATH:
            _init_constants()
        
        lines = ["# Auto-generated MSLib companion file\n"]
        for key, default in Companion.defaults.items():
            lines.append(f"\n{key} = {repr(default)}")
        
        try:
            dir_path = os.path.dirname(COMPANION_PATH)
            if dir_path:  # Only create if not empty string
                os.makedirs(dir_path, exist_ok=True)
            with open(COMPANION_PATH, "w", encoding="utf-8") as f:
                f.writelines(lines)
            logger.info(f"Companion file created at: {COMPANION_PATH}")
        except Exception as e:
            logger.error(f"Failed to create companion file: {format_exc_only(e)}")
    
    def import_it(self):
        """Imports the companion module."""
        try:
            import sys
            # Remove from sys.modules if already imported to force reload
            if 'mslib_companion' in sys.modules:
                del sys.modules['mslib_companion']
            
            # Add plugins directory to path if not present
            if PLUGINS_DIRECTORY not in sys.path:
                sys.path.insert(0, PLUGINS_DIRECTORY)
            
            import mslib_companion  # type: ignore
            self.module = mslib_companion
            logger.info("Companion module imported successfully")
        except ImportError:
            logger.warning("Companion file not found, creating new one")
            self.create()
            try:
                import mslib_companion  # type: ignore
                self.module = mslib_companion
                logger.info("Companion module imported after creation")
            except Exception as e:
                logger.error(f"Failed to import companion after creation: {format_exc_only(e)}")
        except Exception as e:
            logger.error(f"Failed to import companion module: {format_exc_only(e)}")

# Global companion instance
companion = Companion()

def cache_all_autoupdater_tasks(updater: AutoUpdater):
    """Saves AutoUpdater tasks to companion module."""
    if not companion.module:
        logger.warning("Companion module not initialized, cannot cache tasks")
        return
    
    try:
        companion.module.autoupdates_tasks.clear()
        companion.module.autoupdates_tasks = updater.tasks.copy()
        logger.info(f"Cached {len(updater.tasks)} autoupdater tasks")
    except Exception as e:
        logger.error(f"Failed to cache autoupdater tasks: {format_exc_only(e)}")

def load_cached_autoupdater_tasks(updater: AutoUpdater):
    """Loads AutoUpdater tasks from companion module."""
    if not companion.module:
        logger.warning("Companion module not initialized, cannot load tasks")
        return
    
    try:
        if hasattr(companion.module, 'autoupdates_tasks') and companion.module.autoupdates_tasks:
            updater.tasks = companion.module.autoupdates_tasks.copy()
            companion.module.autoupdates_tasks.clear()
            logger.info(f"Loaded {len(updater.tasks)} autoupdater tasks from cache")
        else:
            logger.info("No cached autoupdater tasks found")
    except Exception as e:
        logger.error(f"Failed to load cached autoupdater tasks: {format_exc_only(e)}")

def cache_dispatcher_commands(dispatcher: Dispatcher):
    """Saves dispatcher commands to companion module."""
    if not companion.module:
        logger.warning("Companion module not initialized, cannot cache commands")
        return
    
    try:
        plugin_id = dispatcher.plugin_id
        companion.module.pending_commands[plugin_id] = list(dispatcher.listeners.values())
        companion.module.dispatcher_prefixes[plugin_id] = dispatcher.prefix
        logger.info(f"Cached {len(dispatcher.listeners)} commands for {plugin_id}")
    except Exception as e:
        logger.error(f"Failed to cache dispatcher commands: {format_exc_only(e)}")

def load_cached_dispatcher_commands(dispatcher: Dispatcher):
    """Loads dispatcher commands from companion module."""
    if not companion.module:
        logger.warning("Companion module not initialized, cannot load commands")
        return
    
    try:
        plugin_id = dispatcher.plugin_id
        
        # Load prefix
        if hasattr(companion.module, 'dispatcher_prefixes') and plugin_id in companion.module.dispatcher_prefixes:
            cached_prefix = companion.module.dispatcher_prefixes.get(plugin_id)
            if cached_prefix:
                dispatcher.prefix = cached_prefix
                logger.info(f"Loaded prefix '{cached_prefix}' for {plugin_id}")
        
        # Load commands
        if hasattr(companion.module, 'pending_commands') and plugin_id in companion.module.pending_commands:
            commands = companion.module.pending_commands.get(plugin_id, [])
            for command in commands:
                dispatcher.listeners[command.name] = command
            companion.module.pending_commands.pop(plugin_id, None)
            logger.info(f"Loaded {len(commands)} commands for {plugin_id}")
        else:
            logger.info(f"No cached commands found for {plugin_id}")
    except Exception as e:
        logger.error(f"Failed to load cached dispatcher commands: {format_exc_only(e)}")

# ==================== Telegram API Helper (полная реализация как в CactusLib) ====================
class TelegramAPI:
    """Полноценный Telegram API helper как в CactusLib"""
    
    class Result:
        def __init__(self):
            self.req_id: int = None  # type: ignore
            self.response: TLObject = None
            self.error: Optional[Any] = None  # TLRPC.TL_error
            self._event = threading.Event()
    
    class TLRPCException(Exception):
        def __init__(self, req_id: int, error: Any):
            super().__init__(f"[{req_id}] TLRPCError {error.code}: {error.text}")
            self.error = error
            self.code: int = error.code
            self.text: str = error.text
            self.req_id: int = req_id
    
    class SearchFilter(Enum):
        GIF = 'gif'
        MUSIC = 'music'
        CHAT_PHOTOS = 'chat_photos'
        PHOTOS = 'photos'
        URL = 'url'
        DOCUMENT = 'document'
        PHOTO_VIDEO = 'photo_video'
        PHOTO_VIDEO_DOCUMENT = 'photo_video_document'
        GEO = 'geo'
        PINNED = 'pinned'
        MY_MENTIONS = 'my_mentions'
        ROUND_VOICE = 'round_voice'
        CONTACTS = 'contacts'
        VOICE = 'voice'
        VIDEO = 'video'
        PHONE_CALLS = 'phone_calls'
        ROUND_VIDEO = 'round_video'
        EMPTY = 'empty'
        
        def to_tlrpc_object(self):
            camel_case = re.sub(r'_([a-z])', lambda match: match.group(1).upper(), self.value)
            camel_case = camel_case[0].upper() + camel_case[1:]
            return getattr(TLRPC, f"TL_inputMessagesFilter" + camel_case)()
    
    _res: Dict[str, Result] = {}
    
    @classmethod
    def tlrpc_object(cls, request_class, **kwargs):
        """Устанавливает атрибуты TLRPC объекта"""
        for k, v in kwargs.items():
            setattr(request_class, k, v)
        return request_class
    
    @classmethod
    def send(cls, req, callback: Optional[Callable] = None, *, wait_response: bool = True, timeout: int = 10, raise_errors: bool = True, account: int = 0):
        """Отправляет TLRPC запрос с поддержкой синхронного/асинхронного режима"""
        from uuid import uuid4
        
        uid = uuid4().hex
        if wait_response:
            cls._res[uid] = cls.Result()
        
        def internal_callback(response, error):
            if uid in cls._res:
                cls._res[uid].response = response
                cls._res[uid].error = error
                cls._res[uid]._event.set()
            if callback and not wait_response:
                try:
                    callback(response, error)
                except Exception as e:
                    logger.error(f"Error in callback: {format_exc_only(e)}")
        
        req_id = send_request(req, request_callback_factory(internal_callback), account)
        
        if not wait_response:
            return req_id
        
        cls._res[uid].req_id = req_id
        
        if not cls._res[uid]._event.wait(timeout):
            cls._res.pop(uid, None)
            raise TimeoutError(f"Request {req_id} | {uid} timed out")
        
        result = cls._res.pop(uid)
        if result.error and raise_errors:
            raise cls.TLRPCException(result.req_id, result.error)
        
        return result
    
    send_request = send
    
    @classmethod
    def search_messages(cls, dialog_id: int, query: Optional[str] = None, from_id: Optional[int] = None, 
                       offset_id: int = 0, limit: int = 20, reply_message_id: Optional[int] = None,
                       top_message_id: Optional[int] = None, filter = None, **kwargs):
        """Поиск сообщений в диалоге"""
        if filter is None:
            filter = cls.SearchFilter.EMPTY
        
        req = cls.tlrpc_object(
            TLRPC.TL_messages_search(),
            peer=get_messages_controller().getInputPeer(dialog_id),
            q=query or "",
            offset_id=offset_id,
            limit=limit,
            filter=filter.to_tlrpc_object() if hasattr(filter, 'to_tlrpc_object') else filter,
        )
        
        if from_id:
            req.from_id = get_messages_controller().getInputPeer(from_id)
            req.flags |= 1
        if reply_message_id or top_message_id:
            req.top_msg_id = reply_message_id or top_message_id
            req.flags |= 2
        
        if not kwargs.get("callback"):
            result = cls.send(req, **kwargs)
            messages = arraylist_to_list(result.response.messages) if hasattr(result.response, 'messages') else []
            return [MessageObject(0, msg, False, False) for msg in messages]
        else:
            fn = kwargs.pop("callback")
            def _cb(res, err):
                if not err and res:
                    messages = arraylist_to_list(res.messages) if hasattr(res, 'messages') else []
                    fn([MessageObject(0, msg, False, False) for msg in messages])
            return cls.send(req, callback=_cb, wait_response=False, **kwargs)
    
    @staticmethod
    def get_user(user_id: int):
        """Получает пользователя из кэша"""
        return get_messages_controller().getUser(Long(user_id))
    
    @staticmethod
    def input_user(user_id: int):
        """Создаёт InputUser"""
        return get_messages_controller().getInputUser(Long(user_id))
    
    @staticmethod
    def peer(peer_id: int):
        """Создаёт Peer"""
        return get_messages_controller().getPeer(Long(peer_id))
    
    @staticmethod
    def input_peer(peer_id: int):
        """Создаёт InputPeer"""
        return get_messages_controller().getInputPeer(Long(peer_id))
    
    @classmethod
    def get_sticker_set_by_short_name(cls, short_name: str, **kwargs):
        """Получает стикерпак по короткому имени"""
        return cls.send(
            cls.tlrpc_object(
                TLRPC.TL_messages_getStickerSet(),
                stickerset=cls.tlrpc_object(
                    TLRPC.TL_inputStickerSetShortName(),
                    short_name=short_name
                )
            ), **kwargs
        )
    
    @classmethod
    def get_user_photos(cls, user_id: int, limit: int = 1, offset: int = 0, max_id: int = 0, **kwargs):
        """Получает фотографии пользователя"""
        return cls.send(
            cls.tlrpc_object(
                TLRPC.TL_photos_getUserPhotos(),
                user_id=cls.input_user(user_id),
                offset=offset,
                max_id=max_id,
                limit=limit
            ),
            **kwargs
        )
    
    @classmethod
    def resolve_channel(cls, username: str, callback: Optional[Callable] = None, timeout: int = 20):
        """Резолвит канал по username"""
        from com.exteragram.messenger.utils import ChatUtils  # type: ignore
        from uuid import uuid4
        
        if callback:
            ChatUtils.getInstance().resolveChannel(username, Callback1(callback))
            return
        
        uid = uuid4().hex
        cls._res[uid] = cls.Result()
        
        def internal_callback(result):
            cls._res[uid].response = result
            cls._res[uid]._event.set()
        
        ChatUtils.getInstance().resolveChannel(username, Callback1(internal_callback))
        
        if not cls._res[uid]._event.wait(timeout):
            cls._res.pop(uid, None)
            raise TimeoutError(f"Channel resolve for @{username} timed out")
        
        result = cls._res.pop(uid)
        return result.response
    
    @staticmethod
    def get_chat(chat_id: int):
        """Получает чат из кэша"""
        return get_messages_controller().getChat(Long(-chat_id if chat_id < 0 else chat_id))
    
    @staticmethod
    def get_channel(channel_id: int):
        """Получает канал из кэша (алиас get_chat)"""
        return get_messages_controller().getChat(Long(-channel_id if channel_id < 0 else channel_id))
    
    @classmethod
    def delete_messages(cls, message_id: int, dialog_id: int, topic_id: int = 0, revoke: bool = True):
        """Удаляет сообщения"""
        from org.telegram.messenger import MessagesController  # type: ignore
        
        messages_list = ArrayList()
        messages_list.add(Integer(message_id))
        
        get_messages_controller().deleteMessages(
            messages_list,
            None,  # random_ids
            None,  # encryptedChat
            dialog_id,
            topic_id,
            revoke,
            False  # scheduled
        )

# Создаём глобальный экземпляр для удобства
Telegram = TelegramAPI

# ==================== Inline Buttons Helper ====================
class Inline:
    """Класс для работы с inline кнопками"""
    
    need_markups: Dict[Any, List[Dict[str, Any]]] = {}
    msg_markups: Dict[Any, Dict[Any, Any]] = {}
    callbacks: Dict[Any, Callable] = {}
    
    @staticmethod
    def CallbackData(plugin_id: str, method: str, **kwargs) -> str:
        """Создаёт callback_data для кнопки"""
        from urllib.parse import urlencode
        return f"mslib://{plugin_id}/{method}?{urlencode(kwargs)}"
    
    @staticmethod
    def Button(
            text: str, *,
            url: Optional[str] = None,
            callback_data: Optional[str] = None,
            query: Optional[str] = None,
            requires_password: Optional[bool] = None,
            copy: Optional[str] = None,
            **kwargs
    ):
        """
        Создаёт inline кнопку
        
        Args:
            text: Текст кнопки
            url: URL для открытия
            callback_data: Data для callback
            query: Query для switch_inline
            requires_password: Требуется ли пароль
            copy: Текст для копирования
        """
        # Обработка emoji в тексте
        if m := re.findall(r'(<emoji\s+(?:document_id=|id=)(\d+)>([^<]+)</emoji>)', text):
            for tag, emoji_id, emoji in m:
                text = text.replace(tag, f'<emoji id={emoji_id}/>', 1)
        
        if url:
            btn = TLRPC.TL_keyboardButtonUrl()
            btn.text = text
            btn.url = url
            return btn
        elif callback_data or kwargs.get("data"):
            btn = TLRPC.TL_keyboardButtonCallback()
            btn.text = text
            data = callback_data or kwargs.get("data", "")
            if isinstance(data, str):
                data = data.encode("utf-8")
            btn.data = data
            btn.requires_password = requires_password if requires_password is not None else False
            return btn
        elif query:
            btn = TLRPC.TL_keyboardButtonCallback()
            btn.text = text
            btn.data = Inline.CallbackData("mslib", "setQuery", query=query).encode("utf-8")
            btn.requires_password = False
            return btn
        elif copy or kwargs.get("copy_text"):
            btn = TLRPC.TL_keyboardButtonCopy()
            btn.text = text
            btn.copy_text = copy or kwargs.get("copy_text", "")
            return btn
        
        raise ValueError("Invalid button configuration")
    
    @classmethod
    def to_json(cls, btn):
        """Конвертирует кнопку в JSON"""
        return {
            "text": btn.text,
            **{
                key: getattr(btn, key) if key != "data" else bytes(getattr(btn, key)).decode("utf-8")
                for key in (
                    ["data", "requires_password"]
                    if isinstance(btn, TLRPC.TL_keyboardButtonCallback)
                    else ["url"]
                    if isinstance(btn, TLRPC.TL_keyboardButtonUrl)
                    else ["copy_text"]
                    if isinstance(btn, TLRPC.TL_keyboardButtonCopy)
                    else []
                )
            }
        }
    
    class Markup:
        """Класс для создания inline markup"""
        def __init__(self, is_global: bool = False, on_sent: Optional[Callable] = None, *args, **kwargs):
            self.is_global = is_global
            self.on_sent = (on_sent, args, kwargs)
            self._markup = TLRPC.TL_replyInlineMarkup()
            self._json = []
        
        def add_row(self, *btns):
            """Добавляет ряд кнопок"""
            row = []
            for btn in btns:
                if btn is None:
                    continue
                
                if isinstance(btn, ArrayList):
                    btn = list(btn.toArray())
                
                if isinstance(btn, list):
                    for b in btn:
                        if isinstance(b, dict):
                            b = Inline.Button(**b)
                        row.append(b)
                else:
                    if isinstance(btn, dict):
                        btn = Inline.Button(**btn)
                    row.append(btn)
            
            if len(row) > 0:
                tlrow = TLRPC.TL_keyboardButtonRow()
                for item in row:
                    tlrow.buttons.add(item)
                
                self._markup.rows.add(tlrow)
                self._json.append([Inline.to_json(item) for item in row])
            
            return self
        
        @classmethod
        def from_dict(cls, d, *args, **kwargs):
            """Создаёт Markup из словаря"""
            if isinstance(d, (dict, ArrayList)):
                return cls(*args, **kwargs).add_row(d)
            
            if isinstance(d, list):
                if not any(isinstance(item, list) for item in d):
                    return cls(*args, **kwargs).add_row(d)
                
                m = cls(*args, **kwargs)
                for item in d:
                    if isinstance(item, (dict, list, ArrayList)):
                        m.add_row(item)
                return m
        
        def to_url_with_data(self) -> str:
            """Конвертирует markup в URL"""
            data = json.dumps({"markup": self._json})
            encoded = compress_and_encode(data)
            return f"tg://mslib/mdata/{encoded}"
    
    @dataclass
    class CallbackParams:
        """Параметры callback от inline кнопки"""
        cell: Any  # ChatMessageCell
        message: MessageObject
        button: Optional[Any] = None
        is_long: bool = False
        
        def edit_message(self, text: str, **kwargs):
            """Редактирует сообщение"""
            fragment = kwargs.pop("fragment", get_last_fragment())
            edit_message(self.message, text, fragment=fragment, **kwargs)
            if kwargs.get("markup", None) is None and self.message.messageOwner.reply_markup:
                self.edit_markup()
        
        edit = edit_message
        
        def edit_markup(self, markup=None):
            """Редактирует markup"""
            edit_message_markup(self.cell, markup)
        
        def delete_message(self):
            """Удаляет сообщение"""
            dialog_id = self.message.getDialogId()
            chat = get_messages_controller().getChat(-dialog_id)
            if self.message.canDeleteMessage(
                    self.message.getChatMode() == 1,
                    chat
            ):
                topic_id = self.message.getTopicId()
                Telegram.delete_messages(
                    self.message.getRealId(),
                    dialog_id,
                    topic_id if topic_id != dialog_id and chat else 0
                )
        
        delete = delete_message
    
    @classmethod
    def on_click(cls, method: str, support_long_click: bool = False):
        """Декоратор для callback обработчиков"""
        def decorator(func):
            func.__is_inline_callback__ = True
            func.__support_long__ = support_long_click
            func.__data__ = method
            return func
        return decorator

# ==================== SpinnerAlertDialog context manager ====================
class SpinnerAlertDialog:
    """Context manager для показа spinner dialog"""
    def __init__(self, text: Optional[str] = None):
        self.text = text
        self.dialog_builder = None
    
    def __enter__(self):
        @run_on_ui_thread
        def show():
            try:
                frag = get_last_fragment()
                if not frag:
                    return
                act = frag.getParentActivity()
                if not act:
                    return
                
                self.dialog_builder = AlertDialogBuilder(act, AlertDialogBuilder.ALERT_TYPE_SPINNER)
                self.dialog_builder.set_cancelable(False)
                if self.text:
                    try:
                        self.dialog_builder.set_text(self.text)
                    except Exception:
                        pass
                self.dialog_builder.show()
            except Exception as e:
                logger.error(f"Failed to show spinner: {format_exc_only(e)}")
        
        show()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        @run_on_ui_thread
        def hide():
            if self.dialog_builder:
                try:
                    self.dialog_builder.dismiss()
                except Exception:
                    pass
        
        hide()
        return False

# ==================== Base Plugin Class (расширенный как в CactusLib) ====================
class MSLib(BasePlugin):
    """Базовый класс плагина с расширенным функционалом из CactusLib"""
    
    # Система локализации
    strings: Dict[str, Dict[str, str]] = {}
    
    # Минимальная версия библиотеки
    __min_lib_version__: Optional[str] = None
    
    # Данные для автообновления
    UPDATE_DATA = []
    
    def __init__(self):
        super().__init__()
        # Инициализация базы данных сразу в __init__
        self._db = None
        self._init_database()
    
    def _init_database(self):
        """Инициализирует базу данных для плагина"""
        try:
            if hasattr(self, 'id') and self.id:
                db_path = os.path.join(FileSystem.get_cache_dir("datastores"), f"{self.id}_db.json")
                os.makedirs(os.path.dirname(db_path), exist_ok=True)
                self._db = JsonDB(db_path)
                logger.debug(f"Database initialized for {self.id} at {db_path}")
        except Exception as e:
            logger.error(f"Failed to initialize database: {format_exc_only(e)}")
            self._db = None
    
    # ==================== Database methods ====================
    def get(self, key: str, default: Any = None) -> Any:
        """Получает значение из базы данных"""
        if not self._db:
            return default
        return self._db.get(key, default)
    
    def set(self, key: str, value: Any):
        """Сохраняет значение в базу данных"""
        if self._db:
            self._db.set(key, value)
    
    def pop(self, key: str, default: Any = None) -> Any:
        """Удаляет и возвращает значение из базы данных"""
        if not self._db:
            return default
        return self._db.pop(key, default)
    
    def clear_db(self):
        """Очищает базу данных"""
        if self._db:
            self._db.reset()
    
    # ==================== Localization methods ====================
    def lstrings(self) -> Dict[str, str]:
        """Возвращает словарь строк для текущей локали"""
        locale_dict = self.strings.get(LOCALE, self.strings.get("en", {}))
        return locale_dict
    
    def string(self, key: str, *args, default: Optional[str] = None, locale: Optional[str] = None, **kwargs) -> str:
        """Получает локализованную строку"""
        if key is None:
            return (default or "").format(*args, **kwargs)
        
        base_strings = self.strings.get("en", {})
        target_locale = locale if locale else LOCALE
        locale_dict = self.strings.get(target_locale, base_strings)
        
        string = locale_dict.get(key, base_strings.get(key, default)) or default or ""
        
        if args or kwargs:
            try:
                string = string.format(*args, **kwargs)
            except Exception as e:
                logger.error(f"Failed to format string '{key}': {format_exc_only(e)}")
        
        return string
    
    # ==================== Logging methods ====================
    def log(self, message: str, level: str = "INFO"):
        """Логирует сообщение с указанным уровнем"""
        level_name = level.upper()
        plugin_id = getattr(self, 'id', 'unknown')
        _log(f"[{level_name}] [{plugin_id}] {message}")
    
    def debug(self, message: str):
        """Логирует debug сообщение"""
        self.log(message, "DEBUG")
    
    def info(self, message: str):
        """Логирует info сообщение"""
        self.log(message, "INFO")
    
    def warn(self, message: str):
        """Логирует warning сообщение"""
        self.log(message, "WARN")
    
    def error(self, message: str):
        """Логирует error сообщение"""
        self.log(message, "ERROR")
    
    # ==================== Export/Import data methods ====================
    def export_data(self) -> Dict[str, Any]:
        """Экспортирует дополнительные данные плагина (переопределяется в наследниках)"""
        return {}
    
    def import_data(self, data: Dict[str, Any]):
        """Импортирует дополнительные данные плагина (переопределяется в наследниках)"""
        pass
    
    def _export_data(self) -> Dict[str, Any]:
        """Внутренний метод экспорта всех данных"""
        data = {}
        
        # Экспорт базы данных
        if self._db and len(self._db) > 0:
            data["db"] = dict(self._db)
        
        # Экспорт дополнительных данных
        other = self.export_data()
        if other:
            data["other"] = other
        
        return data
    
    def _import_data(self, data: Dict[str, Any]):
        """Внутренний метод импорта всех данных"""
        # Импорт базы данных
        if "db" in data and self._db:
            self._db.clear()
            self._db.update(data["db"])
            self._db.save()
        
        # Импорт дополнительных данных
        if "other" in data:
            self.import_data(data["other"])
    
    # ==================== Messaging helpers ====================
    def answer(self, params, text: str, *, parse_message: bool = True, parse_mode: str = "HTML", markup=None, **kwargs):
        """Отправляет ответное сообщение"""
        return send_message_with_entities(
            params.peer if hasattr(params, 'peer') else params,
            text,
            parse_message=parse_message,
            parse_mode=parse_mode,
            markup=markup,
            replyToMsg=getattr(params, "replyToMsg", None),
            replyToTopMsg=getattr(params, "replyToTopMsg", None),
            **kwargs
        )
    
    def answer_file(self, params, path: str, caption: Optional[str] = None, *, parse_markdown: bool = True, **kwargs):
        """Отправляет файл (обёртка)"""
        return answer_file(params, path, caption, parse_markdown=parse_markdown, **kwargs)
    
    def answer_photo(self, params, path: str, caption: Optional[str] = None, *, parse_message: bool = True, parse_mode: str = "HTML", **kwargs):
        """Отправляет фото"""
        try:
            from org.telegram.messenger import SendMessagesHelper  # type: ignore
            
            helper = get_messages_controller().getSendMessagesHelper()
            photo = helper.generatePhotoSizes(path, None)
            
            if not photo:
                raise Exception("Failed to generate photo sizes")
            
            entities = None
            if caption and parse_message:
                if parse_mode.upper() == "HTML":
                    parsed = HTML.parse(caption)
                elif parse_mode.upper() == "MARKDOWN":
                    parsed = Markdown.parse(caption)
                else:
                    parsed = None
                
                if parsed:
                    caption = parsed.text
                    entities = [e.to_tlrpc_object() for e in parsed.entities]
            
            # Создаём params для отправки
            send_params = {
                "peer": params.peer if hasattr(params, 'peer') else params,
                "replyToMsg": getattr(params, "replyToMsg", None),
                "replyToTopMsg": getattr(params, "replyToTopMsg", None),
                "caption": caption,
                "photo": photo,
                "path": path,
                **kwargs
            }
            
            if entities:
                send_params["entities"] = list_to_arraylist(entities)
            
            send_message(send_params)
            
        except Exception as e:
            logger.error(f"Failed to send photo: {format_exc()}")
    
    def open_plugin_settings(self):
        """Открывает настройки плагина"""
        try:
            from com.exteragram.messenger.plugins.ui import PluginSettingsActivity  # type: ignore
            frag = get_last_fragment()
            if frag:
                frag.presentFragment(PluginSettingsActivity.of(self.id))
        except Exception as e:
            logger.error(f"Failed to open settings: {format_exc_only(e)}")
    
    def on_plugin_load(self):
        _init_constants()
        
        # Проверка минимальной версии библиотеки
        if self.__min_lib_version__:
            try:
                min_ver = tuple(map(int, self.__min_lib_version__.split(".")))
                cur_ver = tuple(map(int, __version__.split(".")))
                if min_ver > cur_ver:
                    raise Exception(
                        f"Plugin requires MSLib version {self.__min_lib_version__} or higher, "
                        f"but {__version__} is installed"
                    )
            except Exception as e:
                logger.error(f"Version check failed: {format_exc_only(e)}")
        
        # Initialize companion file system
        companion.import_it()
        
        # Проверяем и переинициализируем базу данных если нужно
        if self._db is None:
            self._init_database()
        
        # Initialize dispatcher
        prefix = self.get_setting("command_prefix", ".")
        self._dispatcher = Dispatcher(__id__, prefix=prefix)
        logger.info(f"Command dispatcher initialized with prefix: {prefix}")
        
        # Load cached commands if available
        load_cached_dispatcher_commands(self._dispatcher)
        
        self.add_on_send_message_hook(priority=999999)
        
        logger.info(localise("loaded"))
        self.log("MSLib initialized")
        self.debug("Plugin loaded successfully")
        
        if self.get_setting("enable_autoupdater", False):
            from functools import partial
            run_on_ui_thread(partial(self._delayed_autoupdater_start))
            logger.info("AutoUpdater will be started via delayed callback")
        
        self._setup_addons()

    
    def _delayed_autoupdater_start(self):
        if not hasattr(self, '_autoupdater'):
            self._autoupdater = AutoUpdater(plugin_instance=self)
            
            # Load cached tasks before adding new one
            load_cached_autoupdater_tasks(self._autoupdater)
            
            add_autoupdater_task(__id__, MSLIB_AUTOUPDATE_CHANNEL_ID, MSLIB_AUTOUPDATE_MSG_ID, self._autoupdater)
            logger.info(f"MSLib self-update task added: channel={MSLIB_AUTOUPDATE_CHANNEL_ID}, message={MSLIB_AUTOUPDATE_MSG_ID}")
        
        if self._autoupdater.thread is None or not self._autoupdater.thread.is_alive():
            self._autoupdater.run()
            logger.info("AutoUpdater started on plugin load")
    
    
    def on_send_message_hook(self, account, params):
        if not hasattr(self, '_dispatcher') or not params or not params.message:
            return HookResult()
        
        message = params.message.strip()
        prefix = self._dispatcher.prefix
        
        if not message or not message.startswith(prefix):
            return HookResult()
        
        if message == prefix or (len(message) > 1 and ' ' not in message):
            commands = self._get_command_hints(message)
            
            if commands:
                markup = Inline.Markup()
                
                row_buttons = []
                for cmd_name, cmd_doc in commands[:12]:
                    btn_text = f"/{cmd_name}"
                    row_buttons.append(
                        Inline.button(
                            btn_text,
                            callback_data=Inline.CallbackData("mslib", "setCommand", cmd=cmd_name)
                        )
                    )
                    
                    if len(row_buttons) == 3:
                        markup.add_row(*row_buttons)
                        row_buttons = []
                
                if row_buttons:
                    markup.add_row(*row_buttons)
                
                hint_text = f"<b>💡 Available commands ({len(commands)}):</b>\n"
                hint_text += "\n".join([f"  <code>{prefix}{name}</code> - {doc}" for name, doc in commands[:5]])
                
                if len(commands) > 5:
                    hint_text += f"\n  <i>... and {len(commands) - 5} more</i>"
                
                peer_id = params.peer
                
                try:
                    # Parse HTML to get entities
                    parsed = HTML.parse(hint_text)
                    # Convert RawEntity to TLRPC entities
                    tlrpc_entities = [e.to_tlrpc_object() for e in parsed.entities]
                    
                    # Create params dict according to exteraGram documentation
                    msg_params = {
                        "peer": peer_id,
                        "message": parsed.text,
                        "entities": tlrpc_entities,
                        "reply_markup": markup.to_tlrpc()
                    }
                    send_message(msg_params)
                except Exception as e:
                    logger.error(f"Failed to send command hints: {format_exc_only(e)}")
                
                return HookResult(strategy=HookStrategy.CANCEL)
        
        return HookResult()
    
    def _get_command_hints(self, partial_text: str) -> List[Tuple[str, str]]:
        if not hasattr(self, '_dispatcher'):
            return []
        
        prefix = self._dispatcher.prefix
        search_text = partial_text[len(prefix):].lower()
        
        commands = []

        for cmd_name in self._dispatcher.listeners.keys():
            if not search_text or cmd_name.lower().startswith(search_text):
                cmd_obj = self._dispatcher.listeners[cmd_name]
                doc = "No description"
                
                if hasattr(cmd_obj.func, '__doc__') and cmd_obj.func.__doc__:
                    doc = cmd_obj.func.__doc__.strip().split('\n')[0][:50]
                
                commands.append((cmd_name, doc))
        
        commands.sort(key=lambda x: x[0])
        
        return commands
    
    @Inline.on_click("setCommand")
    def _set_command_callback(self, params, cmd: str):
        try:
            frag = get_last_fragment()
            if not frag:
                return
            
            from hook_utils import get_private_field # type: ignore
            chat_enter_view = get_private_field(frag, "chatActivityEnterView")
            
            if chat_enter_view:
                prefix = self._dispatcher.prefix if hasattr(self, '_dispatcher') else "."
                chat_enter_view.setFieldText(f"{prefix}{cmd} ")
                logger.debug(f"Set command in field: {prefix}{cmd}")
        except Exception as e:
            logger.error(f"Failed to set command: {format_exc_only(e)}")
    
    def _setup_addons(self):
        class ArticleViewerFixHook(MethodHook):
            def before_hooked_method(hook_self, param):
                if not self.get_setting("enable_article_viewer_fix", False):
                    return
                param.setResult(False)
        
        try:
            article_viewer_window_class = jclass("org.telegram.ui.ArticleViewer$WindowView")
            method = article_viewer_window_class.getClass().getDeclaredMethod("handleTouchEvent", MotionEvent)
            self.hook_method(method, ArticleViewerFixHook())
            logger.info("Article Viewer Fix hook registered")
        except Exception as e:
            logger.error(f"Failed to register Article Viewer Fix: {e}")
        
        class NoCallConfirmationHook(MethodHook):
            def before_hooked_method(hook_self, param):
                if not self.get_setting("enable_no_call_confirmation", False):
                    return
                param.args[6] = True
        
        try:
            from org.telegram.ui.Components.voip import VoIPHelper  # type: ignore
            from android.app import Activity # type: ignore
            voip_helper_class = VoIPHelper.getClass()
            method = voip_helper_class.getDeclaredMethod(
                "startCall",
                TLRPC.User, Boolean.TYPE, Boolean.TYPE, 
                Activity, TLRPC.UserFull, AccountInstance, Boolean.TYPE
            )
            self.hook_method(method, NoCallConfirmationHook())
            logger.info("No Call Confirmation hook registered")
        except Exception as e:
            logger.error(f"Failed to register No Call Confirmation: {e}")
        
        class OldBottomForwardHook(MethodHook):
            def before_hooked_method(hook_self, param):
                if not self.get_setting("enable_old_bottom_forward", False):
                    return
                param.args[0] = True
        
        try:
            chat_activity_class = ChatActivity.getClass()
            method = chat_activity_class.getDeclaredMethod("openForward", Boolean.TYPE)
            self.hook_method(method, OldBottomForwardHook())
            logger.info("Old Bottom Forward hook registered")
        except Exception as e:
            logger.error(f"Failed to register Old Bottom Forward: {e}")
        
        class HideProfileEditButtonHook(MethodHook):
            def after_hooked_method(hook_self, param):
                if not self.get_setting("enable_hide_profile_edit", False):
                    return
                
                try:
                    profileActivity = param.thisObject
                    set_private_field(profileActivity, "editItemVisible", False)
                    editItem = get_private_field(profileActivity, "editItem")
                    if editItem:
                        editItem.setVisibility(View.GONE)
                        logger.info("Profile edit button (editItem) hidden successfully")
                except Exception as e:
                    logger.error(f"Failed to hide profile edit button: {format_exc()}")
        
        try:
            from org.telegram.ui import ProfileActivity # type: ignore
            profile_activity_class = ProfileActivity.getClass()
            method = profile_activity_class.getDeclaredMethod("createActionBarMenu", Boolean.TYPE)
            self.hook_method(method, HideProfileEditButtonHook())
            logger.info("Hide Profile Edit Button hook registered")
        except Exception as e:
            logger.error(f"Failed to register Hide Profile Edit Button: {e}")
        
        logger.info("Additional features setup complete")


    def on_plugin_unload(self):
        logger.info(localise("unloaded"))
        self.log("MSLib unloaded")

        # Cache state before unloading
        try:
            # Cache AutoUpdater tasks
            if hasattr(self, '_autoupdater') and self._autoupdater:
                cache_all_autoupdater_tasks(self._autoupdater)
            
            # Cache dispatcher commands
            if hasattr(self, '_dispatcher') and self._dispatcher:
                cache_dispatcher_commands(self._dispatcher)
            
            logger.info("State cached to companion file")
        except Exception as e:
            logger.error(f"Failed to cache state: {format_exc_only(e)}")

        # Prefer stopping the instance AutoUpdater if present, fall back to global for compatibility
        stopped = False
        try:
            if hasattr(self, '_autoupdater') and self._autoupdater:
                try:
                    self._autoupdater.force_stop()
                    stopped = True
                    logger.info("AutoUpdater (instance) stopped")
                except Exception as e:
                    logger.error(f"Failed to stop instance AutoUpdater: {format_exc_only(e)}")

            # also check global variable for backwards compatibility
            global autoupdater
            if autoupdater and autoupdater is not getattr(self, '_autoupdater', None):
                try:
                    autoupdater.force_stop()
                    autoupdater = None
                    stopped = True
                    logger.info("AutoUpdater (global) stopped")
                except Exception as e:
                    logger.error(f"Failed to stop global AutoUpdater: {format_exc_only(e)}")

            if not stopped:
                logger.info("No AutoUpdater instance to stop")
        except Exception as e:
            logger.error(f"Error during plugin unload cleanup: {format_exc_only(e)}")
    

    def create_settings(self):
        def toggle_autoupdater(enabled: bool):
            logger.info(f"Toggle AutoUpdater: {enabled}")

            try:
                if enabled:
                    # prefer plugin-instance _autoupdater
                    if not hasattr(self, '_autoupdater') or not self._autoupdater:
                        self._autoupdater = AutoUpdater(plugin_instance=self)
                        # keep global in sync for compatibility
                        globals()['autoupdater'] = self._autoupdater
                        add_autoupdater_task(__id__, MSLIB_AUTOUPDATE_CHANNEL_ID, MSLIB_AUTOUPDATE_MSG_ID, self._autoupdater)
                        logger.info("MSLib self-update task added (instance)")

                    if self._autoupdater.thread is None or not self._autoupdater.thread.is_alive():
                        self._autoupdater.run()
                        _bulletin("success", localise("autoupdater_started"))
                        logger.info("AutoUpdater started (instance)")
                    else:
                        _bulletin("info", localise("autoupdater_already_running"))
                else:
                    stopped = False
                    if hasattr(self, '_autoupdater') and self._autoupdater:
                        try:
                            self._autoupdater.force_stop()
                            stopped = True
                            logger.info("AutoUpdater stopped (instance)")
                        except Exception as e:
                            logger.error(f"Failed to stop instance AutoUpdater: {format_exc_only(e)}")

                    # also stop global if present and different
                    if autoupdater and autoupdater is not getattr(self, '_autoupdater', None):
                        try:
                            autoupdater.force_stop()
                            globals()['autoupdater'] = None
                            stopped = True
                            logger.info("AutoUpdater stopped (global)")
                        except Exception as e:
                            logger.error(f"Failed to stop global AutoUpdater: {format_exc_only(e)}")

                    if stopped:
                        _bulletin("info", localise("autoupdater_stopped"))
                    else:
                        _bulletin("info", localise("autoupdater_already_stopped"))
            except Exception as e:
                logger.error(f"Error toggling AutoUpdater: {format_exc_only(e)}")
        
        def update_command_prefix(new_prefix: str):
            if hasattr(self, '_dispatcher') and new_prefix:
                if Dispatcher.validate_prefix(new_prefix):
                    self._dispatcher.set_prefix(new_prefix)
                    _bulletin("success", localise("command_prefix_updated").format(prefix=new_prefix))
                    logger.info(f"Command prefix updated to: {new_prefix}")
                else:
                    _bulletin("error", localise("invalid_prefix"))
        
        def force_update_check_onclick(_):
            if hasattr(self, '_autoupdater') and self._autoupdater.thread and self._autoupdater.thread.is_alive():
                self._autoupdater.force_update_check()
                _bulletin("success", localise("update_check_started"))
            else:
                _bulletin("error", localise("autoupdater_not_running"))
        
        def switch_debug_mode(new_value: bool):
            logger.setLevel(logging.DEBUG if new_value else logging.INFO)
            logger.info(f"Debug mode: {new_value}, level: {logging.getLevelName(logger.level)}")
        
        def toggle_plugin(plugin_name: str):
            def callback(value: bool):
                logger.info(f"[TOGGLE] Plugin: {plugin_name}, Value: {value}")
                
                status = "enabled" if value else "disabled"
                level = "success" if value else "info"
                message_key = f"addon-{plugin_name.replace('_', '-')}-{status}"
                message = localise(message_key)
                logger.info(f"[BULLETIN] Level: {level}, Message: {message}")
                _bulletin(level, message)
                logger.info("[TOGGLE] Complete - hook will check setting on next call")
            return callback
        
        return [
            Header(text=localise("commands_header")),
            Input(
                key="command_prefix",
                text=localise("command_prefix_label"),
                subtext=localise("command_prefix_hint"),
                default=".",
                icon="msg_limit_stories",
                on_change=update_command_prefix
            ),
            Divider(),
            Header(text=localise("autoupdater_header")),
            Switch(
                key="enable_autoupdater",
                text=localise("enable_autoupdater"),
                default=False,
                icon="msg_download_solar",
                on_change=toggle_autoupdater
            ),
            Text(
                text=localise("force_update_check"),
                icon="msg_photo_switch2",
                on_click=force_update_check_onclick
            ),
            Input(
                key="autoupdate_timeout",
                text=localise("autoupdate_timeout_title"),
                subtext=localise("autoupdate_timeout_hint"),
                default=DEFAULT_AUTOUPDATE_TIMEOUT,
                icon="msg2_autodelete"
            ),
            Switch(
                key="disable_timestamp_check",
                text=localise("disable_timestamp_check_title"),
                subtext=localise("disable_timestamp_check_hint"),
                default=DEFAULT_DISABLE_TIMESTAMP_CHECK,
                icon="msg_recent"
            ),
            Divider(),
            Header(text=localise("addons_header")),
            Switch(
                key="enable_article_viewer_fix",
                text=localise("addon_article_viewer_fix"),
                default=False,
                icon="msg_language_solar",
                on_change=toggle_plugin("article_viewer_fix")
            ),
            Switch(
                key="enable_no_call_confirmation",
                text=localise("addon_no_call_confirmation"),
                default=False,
                icon="msg_calls_solar",
                on_change=toggle_plugin("no_call_confirmation")
            ),
            Switch(
                key="enable_old_bottom_forward",
                text=localise("addon_old_bottom_forward"),
                default=False,
                icon="input_forward_solar",
                on_change=toggle_plugin("old_bottom_forward")
            ),
            Switch(
                key="enable_hide_profile_edit",
                text=localise("addon_hide_profile_edit"),
                default=False,
                icon="msg_edit",
                on_change=toggle_plugin("hide_profile_edit")
            ),
            Divider(),
            Header(text=localise("dev_header")),
            Switch(
                key="debug_mode",
                text=localise("debug_mode_title"),
                subtext=localise("debug_mode_hint"),
                default=DEFAULT_DEBUG_MODE,
                icon="msg_log_solar",
                on_change=switch_debug_mode
            ),
        ]
