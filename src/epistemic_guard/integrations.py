"""Drop-in Python agent integrations: sync/async `@signed_tool` decorator and
`verify_chat_turn(messages)` for OpenAI, Anthropic, MCP, PydanticAI, and LangGraph loops.
"""

from __future__ import annotations

import functools
import hashlib
import inspect
import json
import math
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any, ParamSpec, Protocol, TypeVar, overload, runtime_checkable

from epistemic_guard.exceptions import EpistemicGuardError
from epistemic_guard.provenance import DomainSigner
from epistemic_guard.schemas import SupervisionVerdict, ToolReceipt
from epistemic_guard.store import MerkleReceiptLog
from epistemic_guard.supervision import ReceiptSupervisor

P = ParamSpec("P")
R = TypeVar("R")


def _coerce_arg_scalar(val: object) -> str | int | float | bool:
    if isinstance(val, bool) or isinstance(val, (str, int)):
        return val
    if isinstance(val, float):
        return val if math.isfinite(val) else str(val)
    try:
        return json.dumps(val, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError):
        return str(val)


def _extract_bound_args(
    fn: Callable[..., Any], args: tuple[object, ...], kwargs: dict[str, object]
) -> dict[str, str | int | float | bool]:
    try:
        sig = inspect.signature(fn)
        bound = sig.bind_partial(*args, **kwargs)
        bound.apply_defaults()
        return {
            k: _coerce_arg_scalar(v)
            for k, v in bound.arguments.items()
            if k not in ("self", "cls")
        }
    except Exception:
        return {k: _coerce_arg_scalar(v) for k, v in kwargs.items()}


@runtime_checkable
class SupportsOtelSpan(Protocol):
    """Structural protocol matching OpenTelemetry `Span` / Langfuse observation objects."""

    def set_attribute(self, key: str, value: str | int | float | bool) -> object: ...


def receipt_to_otel_attributes(
    receipt: ToolReceipt,
    *,
    prefix: str = "epistemic_guard",
) -> dict[str, str | int | float | bool]:
    """Export a `ToolReceipt` into flat OpenTelemetry / Langfuse span event attributes."""
    out_sha = receipt.output_sha256 or hashlib.sha256(receipt.output.encode("utf-8")).hexdigest()
    aer1 = receipt.to_aer1()
    attrs: dict[str, str | int | float | bool] = {
        f"{prefix}.receipt.call_id": receipt.call_id,
        f"{prefix}.receipt.tool": receipt.tool,
        f"{prefix}.receipt.executed": receipt.executed,
        f"{prefix}.receipt.ok": receipt.ok,
        f"{prefix}.receipt.output_sha256": out_sha,
        f"{prefix}.receipt.aer1_id": str(aer1["id"]),
        f"{prefix}.receipt.aer1_output_hash": str(aer1["output_hash"]),
        f"{prefix}.receipt.signed": receipt.signature.startswith("ed25519:"),
    }
    if receipt.error_code:
        attrs[f"{prefix}.receipt.error_code"] = receipt.error_code
    if receipt.session_id:
        attrs[f"{prefix}.receipt.session_id"] = receipt.session_id
    if receipt.signature:
        attrs[f"{prefix}.receipt.signature"] = receipt.signature
    return attrs


def verdict_to_otel_attributes(
    verdict: SupervisionVerdict,
    *,
    merkle_root: str = "",
    session_id: str = "",
    prefix: str = "epistemic_guard",
) -> dict[str, str | int | float | bool]:
    """Export a `SupervisionVerdict` into flat OpenTelemetry / Langfuse span attributes."""
    attrs: dict[str, str | int | float | bool] = {
        f"{prefix}.narration.is_valid": verdict.is_valid,
        f"{prefix}.narration.device": verdict.device,
        f"{prefix}.narration.latency_ms": round(float(verdict.latency_ms), 3),
        f"{prefix}.narration.clauses_checked": verdict.clauses_checked,
        f"{prefix}.narration.unsupported_count": len(verdict.unsupported_claims),
    }
    if verdict.unsupported_claims:
        attrs[f"{prefix}.narration.unsupported_claims_json"] = json.dumps(
            list(verdict.unsupported_claims), separators=(",", ":")
        )
    if merkle_root:
        attrs[f"{prefix}.merkle_root"] = merkle_root
    if session_id:
        attrs[f"{prefix}.session_id"] = session_id
    return attrs


def attach_otel_evidence(
    span: SupportsOtelSpan,
    *,
    receipts: Sequence[ToolReceipt] = (),
    verdict: SupervisionVerdict | None = None,
    merkle_root: str = "",
    session_id: str = "",
    prefix: str = "epistemic_guard",
    record_events: bool = True,
) -> dict[str, str | int | float | bool]:
    """Stamp tamper-evident receipt digests and supervision verdicts onto an OTel/Langfuse span."""
    applied: dict[str, str | int | float | bool] = {
        f"{prefix}.receipt_count": len(receipts),
        f"{prefix}.signed_receipt_count": sum(
            1 for r in receipts if r.signature.startswith("ed25519:")
        ),
    }
    if merkle_root:
        applied[f"{prefix}.merkle_root"] = merkle_root
    active_sess = session_id or next((r.session_id for r in receipts if r.session_id), "")
    if active_sess:
        applied[f"{prefix}.session_id"] = active_sess
    if verdict is not None:
        applied.update(
            verdict_to_otel_attributes(
                verdict,
                merkle_root=merkle_root,
                session_id=active_sess,
                prefix=prefix,
            )
        )
    for key, val in applied.items():
        span.set_attribute(key, val)

    add_event_fn = getattr(span, "add_event", None)
    if record_events and callable(add_event_fn):
        for rcpt in receipts:
            add_event_fn(
                f"{prefix}.receipt",
                receipt_to_otel_attributes(rcpt, prefix=prefix),
            )
    return applied


class ReceiptRecorder:
    """Collects signed or unsigned `ToolReceipt` records across a single agent turn."""

    def __init__(
        self,
        *,
        signer: DomainSigner | None = None,
        merkle_log: MerkleReceiptLog | None = None,
        session_id: str = "",
        otel_span: SupportsOtelSpan | None = None,
    ) -> None:
        self._signer = signer
        self._merkle_log = merkle_log
        self._session_id = session_id
        self._otel_span = otel_span
        self._turn_receipts: list[ToolReceipt] = []

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def receipts(self) -> list[ToolReceipt]:
        return list(self._turn_receipts)

    def clear_turn(self) -> None:
        """Reset per-turn receipt list (while keeping `merkle_log` audit history intact)."""
        self._turn_receipts.clear()

    def record(
        self,
        tool: str,
        *,
        ok: bool,
        args: dict[str, str | int | float | bool],
        output: str,
        executed: bool = True,
        error_code: str = "",
        call_id: str | None = None,
    ) -> ToolReceipt:
        kwargs: dict[str, Any] = {
            "tool": tool,
            "ok": ok,
            "executed": executed,
            "error_code": error_code,
            "args": args,
            "output": output,
        }
        if call_id is not None:
            kwargs["call_id"] = call_id
        if self._session_id:
            kwargs["session_id"] = self._session_id
        raw = ToolReceipt(**kwargs)
        final = self._signer.sign_receipt(raw) if self._signer is not None else raw
        self._turn_receipts.append(final)
        if self._merkle_log is not None:
            self._merkle_log.append(final)
        if self._otel_span is not None:
            add_ev = getattr(self._otel_span, "add_event", None)
            if callable(add_ev):
                add_ev("epistemic_guard.receipt", receipt_to_otel_attributes(final))
        return final

    def record_rejection(
        self,
        tool: str,
        *,
        args: Mapping[str, object] | None = None,
        reason: str,
        error_code: str = "invalid_tool_arguments",
        call_id: str | None = None,
    ) -> ToolReceipt:
        """Record an explicit `executed=False, ok=False` receipt on pre-dispatch rejection."""
        coerced_args = {
            str(k): _coerce_arg_scalar(v) for k, v in (args or {}).items()
        }
        return self.record(
            tool,
            ok=False,
            executed=False,
            error_code=error_code,
            args=coerced_args,
            output=reason,
            call_id=call_id,
        )

    def _parse_dispatch_kwargs(
        self,
        tool_name: str,
        raw_args: str | Mapping[str, object],
        *,
        call_id: str | None = None,
    ) -> dict[str, object]:
        if isinstance(raw_args, str):
            try:
                parsed = json.loads(raw_args) if raw_args.strip() else {}
            except Exception as exc:
                self.record_rejection(
                    tool_name,
                    args={"_raw_arguments": raw_args},
                    reason=f"Malformed JSON tool arguments: {exc}",
                    error_code="invalid_json_arguments",
                    call_id=call_id,
                )
                raise ValueError(f"Malformed JSON arguments for '{tool_name}': {exc}") from exc
            if not isinstance(parsed, Mapping):
                self.record_rejection(
                    tool_name,
                    args={"_raw_arguments": raw_args},
                    reason="Tool arguments JSON must decode to an object.",
                    error_code="invalid_tool_arguments",
                    call_id=call_id,
                )
                raise TypeError(f"Tool arguments for '{tool_name}' must be a JSON object.")
            return {str(k): v for k, v in parsed.items()}
        return {str(k): v for k, v in raw_args.items()}

    def dispatch(
        self,
        fn: Callable[..., R],
        raw_args: str | Mapping[str, object],
        *,
        name: str | None = None,
        call_id: str | None = None,
    ) -> R:
        """Validate JSON/dict arguments against `fn` and record pre-dispatch rejections."""
        raw_fn: Callable[..., Any] = getattr(fn, "__wrapped__", fn)
        tool_name = name or raw_fn.__name__
        kwargs = self._parse_dispatch_kwargs(tool_name, raw_args, call_id=call_id)
        try:
            sig = inspect.signature(raw_fn)
            bound = sig.bind(**kwargs)
            bound.apply_defaults()
        except TypeError as exc:
            self.record_rejection(
                tool_name,
                args=kwargs,
                reason=str(exc),
                error_code="invalid_tool_arguments",
                call_id=call_id,
            )
            raise
        bound_args = {
            k: _coerce_arg_scalar(v) for k, v in bound.arguments.items() if k not in ("self", "cls")
        }
        try:
            res = raw_fn(**kwargs)
        except Exception as exc:
            self.record(
                tool_name,
                ok=False,
                executed=True,
                error_code=type(exc).__name__,
                args=bound_args,
                output=f"{type(exc).__name__}: {exc}",
                call_id=call_id,
            )
            raise
        self.record(
            tool_name,
            ok=True,
            executed=True,
            args=bound_args,
            output=str(res),
            call_id=call_id,
        )
        return res  # type: ignore[no-any-return]

    async def adispatch(
        self,
        fn: Callable[..., Awaitable[R]],
        raw_args: str | Mapping[str, object],
        *,
        name: str | None = None,
        call_id: str | None = None,
    ) -> R:
        """Async counterpart to `dispatch()` with pre-dispatch schema/JSON rejection receipts."""
        raw_fn: Callable[..., Awaitable[R]] = getattr(fn, "__wrapped__", fn)
        tool_name = name or raw_fn.__name__
        kwargs = self._parse_dispatch_kwargs(tool_name, raw_args, call_id=call_id)
        try:
            sig = inspect.signature(raw_fn)
            bound = sig.bind(**kwargs)
            bound.apply_defaults()
        except TypeError as exc:
            self.record_rejection(
                tool_name,
                args=kwargs,
                reason=str(exc),
                error_code="invalid_tool_arguments",
                call_id=call_id,
            )
            raise
        bound_args = {
            k: _coerce_arg_scalar(v) for k, v in bound.arguments.items() if k not in ("self", "cls")
        }
        try:
            res = await raw_fn(**kwargs)
        except Exception as exc:
            self.record(
                tool_name,
                ok=False,
                executed=True,
                error_code=type(exc).__name__,
                args=bound_args,
                output=f"{type(exc).__name__}: {exc}",
                call_id=call_id,
            )
            raise
        self.record(
            tool_name,
            ok=True,
            executed=True,
            args=bound_args,
            output=str(res),
            call_id=call_id,
        )
        return res

    @overload
    def tool(self, fn: Callable[P, R], *, name: str | None = None) -> Callable[P, R]: ...

    @overload
    def tool(
        self, fn: None = None, *, name: str | None = None
    ) -> Callable[[Callable[P, R]], Callable[P, R]]: ...

    def tool(
        self,
        fn: Callable[P, R] | None = None,
        *,
        name: str | None = None,
    ) -> Callable[P, R] | Callable[[Callable[P, R]], Callable[P, R]]:
        """Decorator recording an authentic `ToolReceipt` for sync or async tool functions."""
        return signed_tool(fn, recorder=self, name=name)


@overload
def signed_tool(  # noqa: UP047
    fn: Callable[P, R],
    *,
    recorder: ReceiptRecorder,
    name: str | None = None,
) -> Callable[P, R]: ...


@overload
def signed_tool(
    fn: None = None,
    *,
    recorder: ReceiptRecorder,
    name: str | None = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]: ...


def signed_tool(  # noqa: UP047
    fn: Callable[P, R] | None = None,
    *,
    recorder: ReceiptRecorder,
    name: str | None = None,
) -> Callable[P, R] | Callable[[Callable[P, R]], Callable[P, R]]:
    """Wrap a sync or async Python tool function so every execution emits a `ToolReceipt`.

    If the wrapped tool raises an exception, an explicit `ToolReceipt(ok=False, ...)` is
    recorded before re-raising so the LLM cannot claim a failed call succeeded.
    """

    def _decorator(func: Callable[P, R]) -> Callable[P, R]:
        tool_name = name or func.__name__
        sig = inspect.signature(func)

        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def _async_wrapper(*args: P.args, **kwargs: P.kwargs) -> object:
                arg_map = _extract_bound_args(func, args, kwargs)
                try:
                    sig.bind(*args, **kwargs)
                except TypeError as exc:
                    recorder.record(
                        tool_name,
                        ok=False,
                        executed=False,
                        error_code="invalid_tool_arguments",
                        args=arg_map,
                        output=f"TypeError: {exc}",
                    )
                    raise
                try:
                    res = await func(*args, **kwargs)
                except Exception as exc:
                    recorder.record(
                        tool_name,
                        ok=False,
                        executed=True,
                        error_code=type(exc).__name__,
                        args=arg_map,
                        output=f"{type(exc).__name__}: {exc}",
                    )
                    raise
                recorder.record(tool_name, ok=True, executed=True, args=arg_map, output=str(res))
                return res

            return _async_wrapper  # type: ignore[return-value]

        @functools.wraps(func)
        def _sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            arg_map = _extract_bound_args(func, args, kwargs)
            try:
                sig.bind(*args, **kwargs)
            except TypeError as exc:
                recorder.record(
                    tool_name,
                    ok=False,
                    executed=False,
                    error_code="invalid_tool_arguments",
                    args=arg_map,
                    output=f"TypeError: {exc}",
                )
                raise
            try:
                res = func(*args, **kwargs)
            except Exception as exc:
                recorder.record(
                    tool_name,
                    ok=False,
                    executed=True,
                    error_code=type(exc).__name__,
                    args=arg_map,
                    output=f"{type(exc).__name__}: {exc}",
                )
                raise
            recorder.record(tool_name, ok=True, executed=True, args=arg_map, output=str(res))
            return res

        return _sync_wrapper

    if fn is not None:
        return _decorator(fn)
    return _decorator


def _extract_text_from_message(msg: Mapping[str, Any]) -> str:
    """Extract plain assistant/user prose across OpenAI, Anthropic (`content` blocks),
    and Google Gemini (`parts` arrays) message formats.
    """
    content = msg.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for blk in content:
            if isinstance(blk, str):
                chunks.append(blk)
            elif isinstance(blk, Mapping) and blk.get("type") == "text":
                txt = blk.get("text")
                if isinstance(txt, str) and txt.strip():
                    chunks.append(txt)
        if chunks:
            return "\n".join(chunks)

    parts = msg.get("parts")
    if isinstance(parts, list):
        g_chunks: list[str] = []
        for part in parts:
            if isinstance(part, Mapping) and isinstance(part.get("text"), str):
                if part["text"].strip():
                    g_chunks.append(part["text"])
        if g_chunks:
            return "\n".join(g_chunks)
    return ""


def _parse_raw_args_map(raw_args: object) -> dict[str, str | int | float | bool]:
    if isinstance(raw_args, str):
        try:
            obj = json.loads(raw_args)
            if isinstance(obj, Mapping):
                return {str(k): _coerce_arg_scalar(v) for k, v in obj.items()}
        except Exception:
            return {}
    elif isinstance(raw_args, Mapping):
        return {str(k): _coerce_arg_scalar(v) for k, v in raw_args.items()}
    return {}


def _content_indicates_tool_failure(content_str: str) -> bool:
    """Return True if a tool message payload explicitly signals failure via text prefix or JSON."""
    stripped = content_str.strip()
    if stripped.lower().startswith(("error:", "traceback")):
        return True
    if stripped.startswith("{") and stripped.endswith("}"):
        try:
            obj = json.loads(stripped)
        except (TypeError, ValueError):
            return False
        if isinstance(obj, Mapping):
            if obj.get("ok") is False or obj.get("success") is False:
                return True
            if obj.get("isError") is True or obj.get("is_error") is True:
                return True
            err_val = obj.get("error")
            if err_val not in (None, False, "", [], {}):
                return True
            status_val = str(obj.get("status", "")).strip().lower()
            if status_val in {"error", "failed", "failure", "denied", "rejected"}:
                return True
    return False


def extract_receipts_from_messages(messages: Sequence[Mapping[str, Any]]) -> list[ToolReceipt]:
    """Extract `ToolReceipt` items across OpenAI (`tool_calls`/`tool`), Anthropic
    (`tool_use`/`tool_result`), and Google Gemini (`functionCall`/`functionResponse`) transcripts.
    """
    call_meta: dict[str, tuple[str, dict[str, str | int | float | bool]]] = {}
    gemini_pending_by_name: dict[str, list[dict[str, str | int | float | bool]]] = {}
    extracted: list[ToolReceipt] = []
    seq_counter = 0

    for msg in messages:
        role = str(msg.get("role", "")).lower()

        # 1. OpenAI assistant `tool_calls`
        if role == "assistant":
            tcalls = msg.get("tool_calls")
            if isinstance(tcalls, list):
                for tc in tcalls:
                    if not isinstance(tc, Mapping):
                        continue
                    cid = str(tc.get("id", ""))
                    fn_obj = tc.get("function")
                    if isinstance(fn_obj, Mapping):
                        tname = str(fn_obj.get("name", ""))
                        parsed_args = _parse_raw_args_map(fn_obj.get("arguments", "{}"))
                        if cid and tname:
                            call_meta[cid] = (tname, parsed_args)

        # 2. Anthropic `content` blocks (`tool_use` + `tool_result`)
        content_val = msg.get("content")
        if isinstance(content_val, list):
            for blk in content_val:
                if not isinstance(blk, Mapping):
                    continue
                btype = blk.get("type")
                if btype == "tool_use":
                    cid = str(blk.get("id", ""))
                    tname = str(blk.get("name", ""))
                    parsed_args = _parse_raw_args_map(blk.get("input", {}))
                    if cid and tname:
                        call_meta[cid] = (tname, parsed_args)
                elif btype == "tool_result":
                    seq_counter += 1
                    cid = str(blk.get("tool_use_id", ""))
                    raw_res = blk.get("content", "")
                    if isinstance(raw_res, list):
                        res_txt = "\n".join(
                            str(x.get("text", ""))
                            for x in raw_res
                            if isinstance(x, Mapping) and x.get("type") == "text"
                        )
                    elif isinstance(raw_res, str):
                        res_txt = raw_res
                    else:
                        res_txt = json.dumps(raw_res, sort_keys=True, separators=(",", ":"))
                    is_err = bool(blk.get("is_error", False)) or res_txt.lower().startswith(
                        ("error:", "traceback")
                    )
                    tname, parsed_args = call_meta.get(
                        cid, (str(blk.get("name", "tool")), {})
                    )
                    extracted.append(
                        ToolReceipt(
                            tool=tname,
                            ok=not is_err,
                            call_id=cid or f"anthropic-tool-{seq_counter}",
                            args=parsed_args,
                            output=res_txt,
                        )
                    )

        # 3. Google Gemini `parts` (`functionCall` + `functionResponse`)
        parts_val = msg.get("parts")
        if isinstance(parts_val, list):
            for part in parts_val:
                if not isinstance(part, Mapping):
                    continue
                fc = part.get("functionCall") or part.get("function_call")
                if isinstance(fc, Mapping):
                    tname = str(fc.get("name", ""))
                    parsed_args = _parse_raw_args_map(fc.get("args", {}))
                    cid = str(fc.get("id", ""))
                    if cid and tname:
                        call_meta[cid] = (tname, parsed_args)
                    elif tname:
                        gemini_pending_by_name.setdefault(tname, []).append(parsed_args)
                fr = part.get("functionResponse") or part.get("function_response")
                if isinstance(fr, Mapping):
                    seq_counter += 1
                    tname = str(fr.get("name", "tool"))
                    cid = str(fr.get("id", ""))
                    resp_obj = fr.get("response", {})
                    if cid and cid in call_meta:
                        tname, parsed_args = call_meta[cid]
                    else:
                        q = gemini_pending_by_name.get(tname)
                        parsed_args = q.pop(0) if q else {}
                    is_err = False
                    if isinstance(resp_obj, Mapping):
                        if bool(resp_obj.get("is_error", False)) or "error" in resp_obj:
                            is_err = True
                        out_txt = json.dumps(resp_obj, sort_keys=True, separators=(",", ":"))
                    else:
                        out_txt = str(resp_obj)
                    extracted.append(
                        ToolReceipt(
                            tool=tname,
                            ok=not is_err,
                            call_id=cid or f"gemini-fn-{seq_counter}",
                            args=parsed_args,
                            output=out_txt,
                        )
                    )

        # 4. OpenAI `role == "tool"` messages
        if role == "tool" and not isinstance(parts_val, list):
            seq_counter += 1
            cid = str(msg.get("tool_call_id", ""))
            content_str = (
                content_val
                if isinstance(content_val, str)
                else json.dumps(content_val, sort_keys=True, separators=(",", ":"))
            )
            is_err = bool(msg.get("is_error", False)) or _content_indicates_tool_failure(
                content_str
            )
            tname, parsed_args = call_meta.get(cid, (str(msg.get("name", "tool")), {}))
            extracted.append(
                ToolReceipt(
                    tool=tname,
                    ok=not is_err,
                    call_id=cid or f"tool-msg-{seq_counter}",
                    args=parsed_args,
                    output=content_str,
                )
            )
    return extracted


_extract_receipts_from_messages = extract_receipts_from_messages


def verify_chat_turn(
    messages: Sequence[Mapping[str, Any]],
    *,
    response: str | Mapping[str, Any] | None = None,
    receipts: Sequence[ToolReceipt] | None = None,
    supervisor: ReceiptSupervisor | None = None,
    verify_output_values: bool | None = None,
    device: str | None = None,
    warn_on_missing: bool = True,
) -> SupervisionVerdict:
    """Verify an OpenAI, Anthropic, or Google Gemini turn against executed `ToolReceipt`s."""
    sup = supervisor or ReceiptSupervisor(device=device, warn_on_missing=warn_on_missing)

    last_assistant = ""
    if isinstance(response, str):
        last_assistant = response
    elif isinstance(response, Mapping):
        # Support OpenAI ChatCompletion dict (`choices[0].message`) or direct message dict
        choices = response.get("choices")
        if isinstance(choices, list) and choices and isinstance(choices[0], Mapping):
            msg_obj = choices[0].get("message")
            if isinstance(msg_obj, Mapping):
                last_assistant = _extract_text_from_message(msg_obj)
        if not last_assistant:
            last_assistant = _extract_text_from_message(response)

    last_user = ""
    for msg in reversed(messages):
        role = str(msg.get("role", "")).lower()
        text = _extract_text_from_message(msg)
        if not last_assistant and role in {"assistant", "model"} and text.strip():
            last_assistant = text
        elif role == "user" and text.strip():
            last_user = text
            if last_assistant:
                break

    active_receipts = (
        list(receipts) if receipts is not None else extract_receipts_from_messages(messages)
    )
    return sup.verify_narration(
        last_assistant,
        receipts=active_receipts,
        user_prompt=last_user,
        verify_output_values=verify_output_values,
    )


verify_turn = verify_chat_turn


def _read_span_attributes(span: object) -> dict[str, Any]:
    if isinstance(span, Mapping):
        raw_attrs = span.get("attributes")
        if isinstance(raw_attrs, Mapping):
            return {str(k): v for k, v in raw_attrs.items()}
        return {str(k): v for k, v in span.items()}
    attrs_prop = getattr(span, "attributes", None)
    if isinstance(attrs_prop, Mapping):
        return {str(k): v for k, v in attrs_prop.items()}
    return {}


def extract_receipts_from_otel_spans(spans: Sequence[object]) -> list[ToolReceipt]:
    """Extract `ToolReceipt` records from OpenTelemetry `gen_ai.*` / OpenInference tool spans."""
    receipts: list[ToolReceipt] = []
    for idx, sp in enumerate(spans):
        attrs = _read_span_attributes(sp)
        op_name = str(attrs.get("gen_ai.operation.name") or "").lower()
        tool_name = str(
            attrs.get("gen_ai.tool.name")
            or attrs.get("epistemic_guard.receipt.tool")
            or attrs.get("tool.name")
            or ""
        )
        if not tool_name and op_name != "execute_tool":
            continue
        if not tool_name:
            span_name = getattr(sp, "name", None) or (
                sp.get("name") if isinstance(sp, Mapping) else ""
            )
            tool_name = str(span_name or "tool")

        call_id = str(
            attrs.get("gen_ai.tool.call.id")
            or attrs.get("epistemic_guard.receipt.call_id")
            or f"otel-tool-{idx + 1}"
        )
        raw_args = (
            attrs.get("gen_ai.tool.call.arguments")
            or attrs.get("tool.parameters")
            or attrs.get("input.value")
            or {}
        )
        args_map = _parse_raw_args_map(raw_args)
        raw_res = (
            attrs.get("gen_ai.tool.call.result")
            or attrs.get("output.value")
            or ""
        )
        out_str = (
            raw_res
            if isinstance(raw_res, str)
            else json.dumps(raw_res, sort_keys=True, separators=(",", ":"))
        )
        err_type = str(
            attrs.get("error.type") or attrs.get("epistemic_guard.receipt.error_code") or ""
        )
        status_code_attr = str(attrs.get("otel.status_code") or "").upper()
        ok_attr = attrs.get("epistemic_guard.receipt.ok")
        exec_attr = attrs.get("epistemic_guard.receipt.executed")
        is_ok = (
            bool(ok_attr)
            if isinstance(ok_attr, bool)
            else (not bool(err_type) and status_code_attr != "ERROR")
        )
        executed = bool(exec_attr) if isinstance(exec_attr, bool) else True
        receipts.append(
            ToolReceipt(
                tool=tool_name,
                ok=is_ok,
                executed=executed,
                error_code=err_type,
                call_id=call_id,
                args=args_map,
                output=out_str,
            )
        )
    return receipts


def verify_otel_spans(
    spans: Sequence[object],
    *,
    assistant_text: str | None = None,
    user_prompt: str = "",
    target_span: SupportsOtelSpan | None = None,
    supervisor: ReceiptSupervisor | None = None,
    verify_output_values: bool | None = None,
    device: str | None = None,
    warn_on_missing: bool = False,
) -> SupervisionVerdict:
    """Verify an assistant turn directly from OpenTelemetry `gen_ai.*` spans (Langfuse, Arize,
    Datadog) and optionally stamp verdict attributes onto `target_span`.
    """
    sup = supervisor or ReceiptSupervisor(device=device, warn_on_missing=warn_on_missing)
    receipts = extract_receipts_from_otel_spans(spans)
    reply_txt = assistant_text or ""
    prompt_txt = user_prompt
    if not reply_txt or not prompt_txt:
        for sp in reversed(spans):
            attrs = _read_span_attributes(sp)
            if not reply_txt:
                cand = (
                    attrs.get("gen_ai.completion")
                    or attrs.get("gen_ai.output.messages")
                    or attrs.get("output.value")
                )
                if isinstance(cand, str) and cand.strip() and not attrs.get("gen_ai.tool.name"):
                    reply_txt = cand
            if not prompt_txt:
                pcand = attrs.get("gen_ai.prompt") or attrs.get("input.value")
                if isinstance(pcand, str) and pcand.strip() and not attrs.get("gen_ai.tool.name"):
                    prompt_txt = pcand
    verdict = sup.verify_narration(
        reply_txt,
        receipts=receipts,
        user_prompt=prompt_txt,
        verify_output_values=verify_output_values,
    )
    if target_span is not None:
        attach_otel_evidence(target_span, receipts=receipts, verdict=verdict)
    return verdict


class EpistemicGuardLiteLLM:
    """Drop-in LiteLLM post-call guardrail (`BerriAI/litellm` `CustomGuardrail` compatible,
    without requiring `litellm` to be installed).
    """

    def __init__(
        self,
        *,
        supervisor: ReceiptSupervisor | None = None,
        action_on_fail: str = "raise",
        verify_output_values: bool = False,
        device: str | None = None,
        warn_on_missing: bool = False,
    ) -> None:
        if action_on_fail not in {"raise", "rewrite"}:
            raise ValueError("action_on_fail must be 'raise' or 'rewrite'.")
        self.supervisor = supervisor or ReceiptSupervisor(
            device=device,
            verify_output_values=verify_output_values,
            warn_on_missing=warn_on_missing,
        )
        self.action_on_fail = action_on_fail
        self.verify_output_values = verify_output_values

    def verify_completion(
        self,
        messages: Sequence[Mapping[str, Any]],
        response: object,
    ) -> SupervisionVerdict:
        reply_text = ""
        if isinstance(response, str):
            reply_text = response
        elif isinstance(response, Mapping):
            reply_text = ""
            verdict = verify_chat_turn(
                messages,
                response=response,
                supervisor=self.supervisor,
                verify_output_values=self.verify_output_values,
            )
            return verdict
        else:
            choices = getattr(response, "choices", None)
            if isinstance(choices, list) and choices:
                msg = getattr(choices[0], "message", None)
                content = getattr(msg, "content", None)
                if isinstance(content, str):
                    reply_text = content
        return verify_chat_turn(
            messages,
            response=reply_text,
            supervisor=self.supervisor,
            verify_output_values=self.verify_output_values,
        )

    async def async_post_call_success_hook(
        self,
        data: Mapping[str, Any],
        user_api_key_dict: Any,  # noqa: ANN401
        response: Any,  # noqa: ANN401
    ) -> Any:  # noqa: ANN401
        """LiteLLM `CustomGuardrail.async_post_call_success_hook` signature."""
        del user_api_key_dict
        raw_msgs = data.get("messages")
        msgs = [m for m in raw_msgs if isinstance(m, Mapping)] if isinstance(raw_msgs, list) else []
        verdict = self.verify_completion(msgs, response)
        if verdict.is_valid:
            return response
        receipts = extract_receipts_from_messages(msgs)
        retry_hint = verdict.to_retry_prompt(receipts)
        if self.action_on_fail == "raise":
            raise EpistemicGuardError(retry_hint)
        if isinstance(response, Mapping):
            mut = dict(response)
            choices = mut.get("choices")
            if isinstance(choices, list) and choices and isinstance(choices[0], Mapping):
                c0 = dict(choices[0])
                m0 = dict(c0.get("message") or {})
                m0["content"] = f"[UNGROUNDED CLAIM BLOCKED] {retry_hint}"
                c0["message"] = m0
                mut["choices"] = [c0, *choices[1:]]
            return mut
        choices_attr = getattr(response, "choices", None)
        if isinstance(choices_attr, list) and choices_attr:
            msg_attr = getattr(choices_attr[0], "message", None)
            if msg_attr is not None and hasattr(msg_attr, "content"):
                msg_attr.content = f"[UNGROUNDED CLAIM BLOCKED] {retry_hint}"
        return response
