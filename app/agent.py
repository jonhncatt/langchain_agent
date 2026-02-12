from __future__ import annotations

import json
import os
from typing import Any
from urllib.parse import urlparse, urlunparse

from pydantic import BaseModel, Field

from app.attachments import extract_document_text, summarize_file_payload
from app.config import AppConfig
from app.local_tools import LocalToolExecutor
from app.models import ChatSettings, ToolEvent


_STYLE_HINTS = {
    "short": "回答尽量简短，先给结论，再给最多3条关键点。",
    "normal": "回答清晰、可执行，避免冗长。",
    "long": "回答可适当详细，但要结构化并突出行动建议。",
}


class RunShellArgs(BaseModel):
    command: str = Field(description="Shell command, e.g. `ls -la` or `rg TODO .`")
    cwd: str = Field(default=".", description="Working directory relative to workspace")
    timeout_sec: int = Field(default=15, ge=1, le=30)


class ListDirectoryArgs(BaseModel):
    path: str = Field(default=".")
    max_entries: int = Field(default=200, ge=1, le=500)


class ReadTextFileArgs(BaseModel):
    path: str
    max_chars: int = Field(default=10000, ge=128, le=50000)


class CopyFileArgs(BaseModel):
    src_path: str
    dst_path: str
    overwrite: bool = True
    create_dirs: bool = True


class WriteTextFileArgs(BaseModel):
    path: str
    content: str
    overwrite: bool = True
    create_dirs: bool = True


class ReplaceInFileArgs(BaseModel):
    path: str
    old_text: str
    new_text: str
    replace_all: bool = False
    max_replacements: int = Field(default=1, ge=1, le=200)


class FetchWebArgs(BaseModel):
    url: str
    max_chars: int = Field(default=24000, ge=512, le=120000)
    timeout_sec: int = Field(default=12, ge=3, le=30)


class _UsageTracker:
    def __init__(self, callback_base_cls: type[Any]) -> None:
        class _Tracker(callback_base_cls):
            def __init__(self) -> None:
                super().__init__()
                self.input_tokens = 0
                self.output_tokens = 0
                self.total_tokens = 0
                self.llm_calls = 0

            def on_llm_end(self, response: Any, **kwargs: Any) -> None:
                usage = _extract_usage_from_llm_result(response)
                if usage["input_tokens"] <= 0 and usage["output_tokens"] <= 0 and usage["total_tokens"] <= 0:
                    return
                self.input_tokens += usage["input_tokens"]
                self.output_tokens += usage["output_tokens"]
                self.total_tokens += usage["total_tokens"]
                self.llm_calls += 1

            def snapshot(self) -> dict[str, int]:
                total_tokens = self.total_tokens
                if total_tokens <= 0:
                    total_tokens = self.input_tokens + self.output_tokens
                return {
                    "input_tokens": int(self.input_tokens),
                    "output_tokens": int(self.output_tokens),
                    "total_tokens": int(total_tokens),
                    "llm_calls": int(self.llm_calls),
                }

        self.instance = _Tracker()


def _extract_usage_from_mapping(mapping: dict[str, Any]) -> dict[str, int]:
    usage = {
        "input_tokens": int(mapping.get("prompt_tokens") or mapping.get("input_tokens") or 0),
        "output_tokens": int(mapping.get("completion_tokens") or mapping.get("output_tokens") or 0),
        "total_tokens": int(mapping.get("total_tokens") or 0),
    }
    if usage["total_tokens"] <= 0:
        usage["total_tokens"] = usage["input_tokens"] + usage["output_tokens"]
    return usage


def _extract_usage_from_llm_result(response: Any) -> dict[str, int]:
    llm_output = getattr(response, "llm_output", None)
    if isinstance(llm_output, dict):
        token_usage = llm_output.get("token_usage") or llm_output.get("usage")
        if isinstance(token_usage, dict):
            return _extract_usage_from_mapping(token_usage)

    generations = getattr(response, "generations", None) or []
    for generation_group in generations:
        for generation in generation_group:
            message = getattr(generation, "message", None)
            response_metadata = getattr(message, "response_metadata", None)
            if isinstance(response_metadata, dict):
                token_usage = response_metadata.get("token_usage")
                if isinstance(token_usage, dict):
                    return _extract_usage_from_mapping(token_usage)
                usage = response_metadata.get("usage")
                if isinstance(usage, dict):
                    return _extract_usage_from_mapping(usage)

            usage_metadata = getattr(message, "usage_metadata", None)
            if isinstance(usage_metadata, dict):
                return _extract_usage_from_mapping(usage_metadata)

    return {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
    }


class OfficeAgent:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.tools = LocalToolExecutor(config)

        try:
            from langchain.agents import create_agent
            from langchain_core.callbacks import BaseCallbackHandler
            from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
            from langchain_core.tools import StructuredTool
            from langchain_openai import ChatOpenAI
        except Exception as exc:
            raise RuntimeError(
                "Missing dependency: langchain/langchain-openai. Install with `pip install -r requirements.txt`."
            ) from exc

        self._create_agent = create_agent
        self._BaseCallbackHandler = BaseCallbackHandler
        self._AIMessage = AIMessage
        self._HumanMessage = HumanMessage
        self._SystemMessage = SystemMessage
        self._ToolMessage = ToolMessage
        self._StructuredTool = StructuredTool
        self._ChatOpenAI = ChatOpenAI
        self._lc_tools = self._build_langchain_tools()

    def maybe_compact_session(self, session: dict[str, Any], keep_last_turns: int) -> bool:
        turns = session.get("turns", [])
        if len(turns) <= self.config.summary_trigger_turns:
            return False

        keep = max(2, min(40, keep_last_turns))
        older = turns[:-keep]
        recent = turns[-keep:]
        if not older:
            return False

        existing_summary = session.get("summary", "")
        session["summary"] = self._summarize_turns(existing_summary, older)
        session["turns"] = recent
        return True

    def _summarize_turns(self, existing_summary: str, older_turns: list[dict[str, Any]]) -> str:
        transcript = []
        if existing_summary:
            transcript.append(f"已有摘要:\n{existing_summary}\n")

        for turn in older_turns:
            role = turn.get("role", "user")
            text = (turn.get("text") or "").strip()
            if not text:
                continue
            transcript.append(f"[{role}] {text}")

        raw = "\n".join(transcript)
        if not raw.strip():
            return existing_summary

        try:
            prompt_messages = [
                self._SystemMessage(
                    content=(
                        "你是会话摘要器。请把历史对话压缩成可供后续继续工作的摘要，"
                        "要保留目标、关键约束、已完成动作、未完成事项。"
                    )
                ),
                self._HumanMessage(content=raw),
            ]
            response = self._invoke_with_405_fallback(
                messages=prompt_messages,
                model=self.config.summary_model,
                max_output_tokens=450,
                enable_tools=False,
            )
            summarized = self._content_to_text(response.content).strip()
            if summarized:
                return summarized
        except Exception:
            pass

        lines: list[str] = []
        if existing_summary:
            lines.append(existing_summary)
        for turn in older_turns[-20:]:
            role = turn.get("role", "user")
            text = (turn.get("text") or "").replace("\n", " ").strip()
            if text:
                lines.append(f"[{role}] {text[:220]}")
        return "\n".join(lines)[:5000]

    def run_chat(
        self,
        history_turns: list[dict[str, Any]],
        summary: str,
        user_message: str,
        attachment_metas: list[dict[str, Any]],
        settings: ChatSettings,
    ) -> tuple[str, list[ToolEvent], str, list[str], list[str], dict[str, int]]:
        model = settings.model or self.config.default_model
        style_hint = _STYLE_HINTS.get(settings.response_style, _STYLE_HINTS["normal"])
        execution_plan = self._build_execution_plan(attachment_metas=attachment_metas, settings=settings)
        execution_trace: list[str] = []
        usage_total = self._empty_usage()
        allowed_roots_text = ", ".join(str(p) for p in self.config.allowed_roots)

        system_prompt = self._build_system_prompt(
            style_hint=style_hint,
            allowed_roots_text=allowed_roots_text,
            summary=summary,
        )

        execution_trace.append(f"工具开关: {'开启' if settings.enable_tools else '关闭'}。")
        execution_trace.append(f"可访问根目录: {allowed_roots_text}")
        if summary.strip():
            execution_trace.append("已加载历史摘要，减少上下文占用。")

        history_messages: list[Any] = []
        for turn in history_turns[-settings.max_context_turns :]:
            role = turn.get("role", "user")
            text = (turn.get("text") or "").strip()
            if not text:
                continue
            if role == "assistant":
                history_messages.append(self._AIMessage(content=text))
            else:
                history_messages.append(self._HumanMessage(content=text))
        execution_trace.append(f"已载入最近 {min(len(history_turns), settings.max_context_turns)} 条历史消息。")

        user_input, attachment_note, attachment_issues = self._build_user_input(user_message, attachment_metas)
        if attachment_metas:
            execution_trace.append(f"已处理 {len(attachment_metas)} 个附件输入。")
        for issue in attachment_issues:
            execution_trace.append(f"附件提示: {issue}")

        tool_events: list[ToolEvent] = []
        execution_trace.append("开始模型推理。")

        if settings.enable_tools:
            try:
                result, usage_total = self._invoke_agent_with_405_fallback(
                    system_prompt=system_prompt,
                    chat_history=history_messages,
                    user_input=user_input,
                    model=model,
                    max_output_tokens=settings.max_output_tokens,
                )
            except Exception as exc:
                execution_trace.append(f"模型请求失败: {exc}")
                return (
                    f"请求模型失败: {exc}",
                    tool_events,
                    attachment_note,
                    execution_plan,
                    execution_trace,
                    usage_total,
                )

            result_messages = result.get("messages") or []
            text = self._extract_text_from_result_messages(result_messages)
            if not text:
                text = "模型未返回可见文本。"

            tool_events = self._build_tool_events_from_messages(result_messages)
            for event in tool_events:
                execution_trace.append(f"执行工具: {event.name}")
        else:
            messages: list[Any] = [self._SystemMessage(content=system_prompt), *history_messages, self._HumanMessage(content=user_input)]
            try:
                response = self._invoke_with_405_fallback(
                    messages=messages,
                    model=model,
                    max_output_tokens=settings.max_output_tokens,
                    enable_tools=False,
                )
                usage_total = self._merge_usage(usage_total, self._extract_usage_from_message(response))
            except Exception as exc:
                execution_trace.append(f"模型请求失败: {exc}")
                return (
                    f"请求模型失败: {exc}",
                    tool_events,
                    attachment_note,
                    execution_plan,
                    execution_trace,
                    usage_total,
                )

            text = self._content_to_text(getattr(response, "content", ""))
            if not text.strip():
                text = "模型未返回可见文本。"

        execution_trace.append("已生成最终答复。")
        return text, tool_events, attachment_note, execution_plan, execution_trace, usage_total

    def _build_system_prompt(self, style_hint: str, allowed_roots_text: str, summary: str) -> str:
        prompt = (
            f"{self.config.system_prompt}\n\n"
            f"输出风格: {style_hint}\n"
            "处理本地文件请求时，先调用工具再下结论，不要凭空判断权限。\n"
            f"可访问路径根目录: {allowed_roots_text}\n"
            "读取文件优先使用 list_directory/read_text_file；"
            "复制文件优先使用 copy_file（不要用读写拼接，避免截断）；"
            "改写或新建文件优先使用 replace_in_file/write_text_file，尽量使用绝对路径。\n"
            "当联网抓取返回 warning（如脚本/反爬页面）时，不要给确定性结论，"
            "必须明确说明信息不足并建议改查权威来源。"
        )
        if summary.strip():
            prompt = f"{prompt}\n\n历史摘要:\n{summary.strip()}"
        return prompt

    def _build_execution_plan(self, attachment_metas: list[dict[str, Any]], settings: ChatSettings) -> list[str]:
        plan = ["理解你的目标和约束。"]
        if attachment_metas:
            plan.append(f"解析附件内容（{len(attachment_metas)} 个）。")
        plan.append(f"结合最近 {settings.max_context_turns} 条历史消息组织上下文。")
        if settings.enable_tools:
            plan.append("如有必要调用工具（读文件/列目录/执行命令/联网抓取）获取事实。")
        plan.append("汇总结论并按你选择的回答长度输出。")
        return plan

    def _build_llm(self, model: str, max_output_tokens: int, use_responses_api: bool | None = None):
        selected_use_responses = self.config.openai_use_responses_api if use_responses_api is None else use_responses_api
        kwargs: dict[str, Any] = {
            "model": model,
            "api_key": os.environ.get("OPENAI_API_KEY"),
            "max_tokens": max_output_tokens,
            "temperature": 0,
            "use_responses_api": selected_use_responses,
        }
        if self.config.openai_base_url:
            kwargs["base_url"] = self._normalize_base_url(self.config.openai_base_url)
        if self.config.openai_ca_cert_path:
            os.environ.setdefault("SSL_CERT_FILE", self.config.openai_ca_cert_path)
            os.environ.setdefault("REQUESTS_CA_BUNDLE", self.config.openai_ca_cert_path)

        try:
            return self._ChatOpenAI(**kwargs)
        except TypeError:
            kwargs.pop("use_responses_api", None)
            return self._ChatOpenAI(**kwargs)

    def _invoke_agent(
        self,
        system_prompt: str,
        chat_history: list[Any],
        user_input: str,
        model: str,
        max_output_tokens: int,
        use_responses_api: bool | None = None,
    ) -> tuple[dict[str, Any], dict[str, int]]:
        usage_tracker = _UsageTracker(self._BaseCallbackHandler).instance
        llm = self._build_llm(model=model, max_output_tokens=max_output_tokens, use_responses_api=use_responses_api)
        agent = self._create_agent(
            model=llm,
            tools=self._lc_tools,
            system_prompt=system_prompt,
        )
        result = agent.invoke(
            {
                "messages": [*chat_history, self._HumanMessage(content=user_input)],
            },
            config={
                "callbacks": [usage_tracker],
                "recursion_limit": 24,
            },
        )
        usage = usage_tracker.snapshot()
        return result, usage

    def _invoke_agent_with_405_fallback(
        self,
        system_prompt: str,
        chat_history: list[Any],
        user_input: str,
        model: str,
        max_output_tokens: int,
    ) -> tuple[dict[str, Any], dict[str, int]]:
        try:
            return self._invoke_agent(
                system_prompt=system_prompt,
                chat_history=chat_history,
                user_input=user_input,
                model=model,
                max_output_tokens=max_output_tokens,
                use_responses_api=None,
            )
        except Exception as exc:
            if not self._is_405_error(exc):
                raise

        fallback_use_responses = not self.config.openai_use_responses_api
        return self._invoke_agent(
            system_prompt=system_prompt,
            chat_history=chat_history,
            user_input=user_input,
            model=model,
            max_output_tokens=max_output_tokens,
            use_responses_api=fallback_use_responses,
        )

    def _invoke_with_405_fallback(
        self,
        messages: list[Any],
        model: str,
        max_output_tokens: int,
        enable_tools: bool,
    ) -> Any:
        llm = self._build_llm(model=model, max_output_tokens=max_output_tokens)
        runner = llm.bind_tools(self._lc_tools) if enable_tools else llm
        try:
            return runner.invoke(messages)
        except Exception as exc:
            if not self._is_405_error(exc):
                raise

        fallback_use_responses = not self.config.openai_use_responses_api
        llm_fb = self._build_llm(
            model=model,
            max_output_tokens=max_output_tokens,
            use_responses_api=fallback_use_responses,
        )
        runner_fb = llm_fb.bind_tools(self._lc_tools) if enable_tools else llm_fb
        return runner_fb.invoke(messages)

    def _build_langchain_tools(self) -> list[Any]:
        return [
            self._StructuredTool.from_function(
                name="run_shell",
                description="Run a safe shell command in workspace. Supports simple commands without pipes.",
                args_schema=RunShellArgs,
                func=self._run_shell_tool,
            ),
            self._StructuredTool.from_function(
                name="list_directory",
                description="List files in a workspace directory.",
                args_schema=ListDirectoryArgs,
                func=self._list_directory_tool,
            ),
            self._StructuredTool.from_function(
                name="read_text_file",
                description="Read a UTF-8 text file in workspace.",
                args_schema=ReadTextFileArgs,
                func=self._read_text_file_tool,
            ),
            self._StructuredTool.from_function(
                name="copy_file",
                description="Copy a file (binary-safe) from src_path to dst_path in allowed roots.",
                args_schema=CopyFileArgs,
                func=self._copy_file_tool,
            ),
            self._StructuredTool.from_function(
                name="write_text_file",
                description="Create or overwrite a UTF-8 text file in workspace.",
                args_schema=WriteTextFileArgs,
                func=self._write_text_file_tool,
            ),
            self._StructuredTool.from_function(
                name="replace_in_file",
                description="Replace target text in a UTF-8 text file in workspace.",
                args_schema=ReplaceInFileArgs,
                func=self._replace_in_file_tool,
            ),
            self._StructuredTool.from_function(
                name="fetch_web",
                description="Fetch web content from a URL for information lookup.",
                args_schema=FetchWebArgs,
                func=self._fetch_web_tool,
            ),
        ]

    def _run_shell_tool(self, command: str, cwd: str = ".", timeout_sec: int = 15) -> str:
        result = self.tools.run_shell(command=command, cwd=cwd, timeout_sec=timeout_sec)
        return json.dumps(result, ensure_ascii=False)

    def _list_directory_tool(self, path: str = ".", max_entries: int = 200) -> str:
        result = self.tools.list_directory(path=path, max_entries=max_entries)
        return json.dumps(result, ensure_ascii=False)

    def _read_text_file_tool(self, path: str, max_chars: int = 10000) -> str:
        result = self.tools.read_text_file(path=path, max_chars=max_chars)
        return json.dumps(result, ensure_ascii=False)

    def _copy_file_tool(
        self, src_path: str, dst_path: str, overwrite: bool = True, create_dirs: bool = True
    ) -> str:
        result = self.tools.copy_file(
            src_path=src_path,
            dst_path=dst_path,
            overwrite=overwrite,
            create_dirs=create_dirs,
        )
        return json.dumps(result, ensure_ascii=False)

    def _write_text_file_tool(
        self, path: str, content: str, overwrite: bool = True, create_dirs: bool = True
    ) -> str:
        result = self.tools.write_text_file(
            path=path,
            content=content,
            overwrite=overwrite,
            create_dirs=create_dirs,
        )
        return json.dumps(result, ensure_ascii=False)

    def _replace_in_file_tool(
        self,
        path: str,
        old_text: str,
        new_text: str,
        replace_all: bool = False,
        max_replacements: int = 1,
    ) -> str:
        result = self.tools.replace_in_file(
            path=path,
            old_text=old_text,
            new_text=new_text,
            replace_all=replace_all,
            max_replacements=max_replacements,
        )
        return json.dumps(result, ensure_ascii=False)

    def _fetch_web_tool(self, url: str, max_chars: int = 24000, timeout_sec: int = 12) -> str:
        result = self.tools.fetch_web(url=url, max_chars=max_chars, timeout_sec=timeout_sec)
        return json.dumps(result, ensure_ascii=False)

    def _build_user_input(self, user_message: str, attachment_metas: list[dict[str, Any]]) -> tuple[str, str, list[str]]:
        chunks = [user_message.strip()]
        notes: list[str] = []
        issues: list[str] = []

        for meta in attachment_metas:
            name = meta.get("original_name", "file")
            path = meta.get("path", "")
            kind = meta.get("kind", "other")

            if kind == "document":
                extracted = extract_document_text(path, self.config.max_attachment_chars)
                if extracted:
                    chunks.append(f"\n[附件文档: {name}]\n{extracted}")
                    notes.append(f"文档:{name}")
                    if extracted.startswith("[文档解析失败:"):
                        issues.append(f"{name} 文档解析失败，模型只收到错误信息。")
                else:
                    try:
                        preview = summarize_file_payload(path, max_bytes=768, max_text_chars=1200)
                        chunks.append(f"\n[附件文档: {name}] 未识别为结构化文本，已附带文件预览。\n{preview}")
                        notes.append(f"文档(预览):{name}")
                        issues.append(f"{name} 未结构化解析，已提供文件预览。")
                    except Exception as exc:
                        chunks.append(f"\n[附件文档: {name}] 读取失败: {exc}")
                        notes.append(f"文档(失败):{name}")
                        issues.append(f"{name} 文档读取失败: {exc}")
            elif kind == "image":
                size = meta.get("size", 0)
                mime = meta.get("mime", "image/*")
                chunks.append(
                    f"\n[附件图片: {name}] 已上传（mime={mime}, size={size} bytes）。"
                    "若需精确图像理解，请补充你想让我关注的区域/文字。"
                )
                notes.append(f"图片:{name}")
                issues.append(f"{name} 当前通过文本元数据注入，未直接做像素级解析。")
            else:
                try:
                    preview = summarize_file_payload(path, max_bytes=768, max_text_chars=1200)
                    chunks.append(f"\n[附件: {name}] 二进制/未知类型，已附带文件预览。\n{preview}")
                    notes.append(f"其他(预览):{name}")
                    issues.append(f"{name} 附件类型未知，已提供二进制预览。")
                except Exception as exc:
                    chunks.append(f"\n[附件: {name}] 读取失败: {exc}")
                    notes.append(f"其他(失败):{name}")
                    issues.append(f"{name} 附件读取失败: {exc}")

        return "\n".join(chunk for chunk in chunks if chunk), "；".join(notes), issues

    def _extract_text_from_result_messages(self, messages: list[Any]) -> str:
        for message in reversed(messages):
            if not isinstance(message, self._AIMessage):
                continue
            text = self._content_to_text(getattr(message, "content", ""))
            if text.strip():
                return text.strip()
        return ""

    def _build_tool_events_from_messages(self, messages: list[Any]) -> list[ToolEvent]:
        events: list[ToolEvent] = []
        pending: dict[str, tuple[str, dict[str, Any] | None]] = {}

        for message in messages:
            if isinstance(message, self._AIMessage):
                for tool_call in getattr(message, "tool_calls", []) or []:
                    call_id = str(tool_call.get("id") or f"call_{len(pending) + 1}")
                    name = str(tool_call.get("name") or "unknown")
                    args = tool_call.get("args")
                    if isinstance(args, dict):
                        parsed_input: dict[str, Any] | None = args
                    elif args is None:
                        parsed_input = None
                    else:
                        parsed_input = {"input": args}
                    pending[call_id] = (name, parsed_input)
                continue

            if not isinstance(message, self._ToolMessage):
                continue

            call_id = str(getattr(message, "tool_call_id", "") or "")
            pending_item = pending.pop(call_id, None)
            if pending_item is None:
                name = str(getattr(message, "name", None) or "unknown")
                event_input = None
            else:
                name, event_input = pending_item

            output_text = self._content_to_text(getattr(message, "content", ""))
            if not output_text:
                output_text = str(getattr(message, "content", ""))

            events.append(
                ToolEvent(
                    name=name,
                    input=event_input,
                    output_preview=output_text[:1200],
                )
            )

        return events

    def _content_to_text(self, content: Any) -> str:
        if isinstance(content, str):
            return content
        if not isinstance(content, list):
            return str(content or "")

        out: list[str] = []
        for item in content:
            if isinstance(item, str):
                out.append(item)
                continue
            if not isinstance(item, dict):
                out.append(str(item))
                continue

            item_type = item.get("type")
            if item_type in {"text", "output_text", "input_text"}:
                text = item.get("text")
                if isinstance(text, str) and text:
                    out.append(text)
        return "\n".join(out).strip()

    def _empty_usage(self) -> dict[str, int]:
        return {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "llm_calls": 0,
        }

    def _merge_usage(self, base: dict[str, int], extra: dict[str, int]) -> dict[str, int]:
        merged = dict(base)
        merged["input_tokens"] = int(merged.get("input_tokens", 0)) + int(extra.get("input_tokens", 0))
        merged["output_tokens"] = int(merged.get("output_tokens", 0)) + int(extra.get("output_tokens", 0))
        merged["total_tokens"] = int(merged.get("total_tokens", 0)) + int(extra.get("total_tokens", 0))
        merged["llm_calls"] = int(merged.get("llm_calls", 0)) + int(extra.get("llm_calls", 0))
        return merged

    def _extract_usage_from_message(self, message: Any) -> dict[str, int]:
        usage = self._empty_usage()

        usage_metadata = getattr(message, "usage_metadata", None)
        if isinstance(usage_metadata, dict):
            usage["input_tokens"] = int(usage_metadata.get("input_tokens") or usage_metadata.get("prompt_tokens") or 0)
            usage["output_tokens"] = int(
                usage_metadata.get("output_tokens") or usage_metadata.get("completion_tokens") or 0
            )
            usage["total_tokens"] = int(usage_metadata.get("total_tokens") or 0)

        response_metadata = getattr(message, "response_metadata", None)
        if isinstance(response_metadata, dict):
            token_usage = response_metadata.get("token_usage")
            if isinstance(token_usage, dict):
                if usage["input_tokens"] <= 0:
                    usage["input_tokens"] = int(token_usage.get("prompt_tokens") or token_usage.get("input_tokens") or 0)
                if usage["output_tokens"] <= 0:
                    usage["output_tokens"] = int(
                        token_usage.get("completion_tokens") or token_usage.get("output_tokens") or 0
                    )
                if usage["total_tokens"] <= 0:
                    usage["total_tokens"] = int(token_usage.get("total_tokens") or 0)

        if usage["total_tokens"] <= 0:
            usage["total_tokens"] = usage["input_tokens"] + usage["output_tokens"]

        usage["llm_calls"] = 1 if (usage["input_tokens"] > 0 or usage["output_tokens"] > 0 or usage["total_tokens"] > 0) else 0
        return usage

    def _normalize_base_url(self, raw_url: str) -> str:
        """
        Accept either base URL (..../v1) or full endpoint URL (..../v1/chat/completions).
        """
        url = raw_url.strip().strip("\"'").rstrip("/")
        parsed = urlparse(url)
        path = parsed.path or ""
        suffixes = ["/chat/completions", "/responses", "/v1/chat/completions", "/v1/responses"]
        lowered = path.lower()
        for suffix in suffixes:
            if lowered.endswith(suffix):
                path = path[: -len(suffix)] + ("/v1" if suffix.startswith("/v1/") else "")
                break
        normalized = urlunparse((parsed.scheme, parsed.netloc, path.rstrip("/"), parsed.params, parsed.query, parsed.fragment))
        return normalized

    def _is_405_error(self, exc: Exception) -> bool:
        text = str(exc).lower()
        return "405" in text or "method not allowed" in text
