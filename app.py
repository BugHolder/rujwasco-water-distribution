
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import traceback
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from groq import AsyncGroq
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from pydantic import BaseModel, Field


# =========================================================
# CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

load_dotenv(BASE_DIR / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger("rujwasco-ai")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv(
    "GROQ_MODEL",
    "openai/gpt-oss-120b",
)

PYTHON_EXECUTABLE = os.getenv(
    "PYTHON_EXECUTABLE",
    sys.executable,
)

MCP_SERVER_PATH = BASE_DIR / "server.py"

SYSTEM_INSTRUCTION = """
You are Rujwasco Distribution Intelligence.

You are an analytical assistant for water distribution operations.

RULES:

1. Use MCP tools whenever a question requires dataset facts
   or calculations.

2. Never invent figures, statistics or results.

3. Explain findings in practical operational language.

4. Distinguish correlation from causation.

5. State the date range and zone scope when available.

6. The dataset is synthetic practice data, not actual
   Rujwasco operational records.

7. UAV is not a verified NRW water balance.

8. If the dataset cannot answer a question, say so.

9. When using MCP tools, explain the results returned
   by the tools rather than simply listing raw output.
"""


# =========================================================
# REQUEST MODELS
# =========================================================

class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=12000)


class ChatResponse(BaseModel):
    reply: str
    model: str
    tools_used: list[str]


# =========================================================
# MCP UTILITIES
# =========================================================

def build_groq_tools(mcp_tools: list[Any]) -> list[dict]:
    """
    Convert MCP tool definitions into Groq/OpenAI-compatible
    function declarations.
    """

    declarations = []

    for tool in mcp_tools:
        declarations.append({
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or tool.name,
                "parameters": tool.inputSchema or {
                    "type": "object",
                    "properties": {},
                },
            },
        })

    return declarations


def serialize_tool_result(result: Any) -> str:
    """Convert MCP tool output into a JSON string."""

    if getattr(result, "isError", False):
        raise RuntimeError("MCP tool returned an error.")

    content = getattr(result, "content", None) or []

    text_parts = [
        item.text
        for item in content
        if getattr(item, "text", None)
    ]

    if text_parts:
        return "\n".join(text_parts)

    structured = getattr(
        result,
        "structuredContent",
        None,
    )

    if structured is not None:
        return json.dumps(structured, default=str)

    return "Tool completed without text output."


def exception_message(exc: BaseException) -> str:
    """Extract useful messages from nested exceptions."""

    if isinstance(exc, BaseExceptionGroup):
        return " | ".join(
            exception_message(error)
            for error in exc.exceptions
        )

    return f"{type(exc).__name__}: {exc}"


def log_full_exception(exc: BaseException) -> None:
    """Log complete exceptions and nested TaskGroup errors."""

    logger.error(
        "FULL EXCEPTION:\n%s",
        "".join(traceback.format_exception(exc)),
    )


# =========================================================
# GROQ + MCP ORCHESTRATION
# =========================================================

async def run_assistant(message: str) -> tuple[str, list[str]]:

    if not GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY is missing from your .env file."
        )

    if not MCP_SERVER_PATH.exists():
        raise FileNotFoundError(
            f"MCP server not found: {MCP_SERVER_PATH}"
        )

    client = AsyncGroq(
        api_key=GROQ_API_KEY,
    )

    server_params = StdioServerParameters(
        command=PYTHON_EXECUTABLE,
        args=[str(MCP_SERVER_PATH)],
        cwd=str(BASE_DIR),
        env={
            **os.environ,
            "PYTHONUNBUFFERED": "1",
        },
    )

    tools_used: list[str] = []

    messages = [
        {
            "role": "system",
            "content": SYSTEM_INSTRUCTION,
        },
        {
            "role": "user",
            "content": message,
        },
    ]

    try:

        logger.info("Starting MCP server...")

        async with stdio_client(server_params) as (
            read_stream,
            write_stream,
        ):

            logger.info("MCP transport connected.")

            async with ClientSession(
                read_stream,
                write_stream,
            ) as session:

                await session.initialize()

                discovered = await session.list_tools()

                logger.info(
                    "Discovered %s MCP tools: %s",
                    len(discovered.tools),
                    [tool.name for tool in discovered.tools],
                )

                available_tools = {
                    tool.name: tool
                    for tool in discovered.tools
                }

                groq_tools = build_groq_tools(
                    discovered.tools
                )

                # -----------------------------------------
                # GROQ FUNCTION CALLING LOOP
                # -----------------------------------------

                for round_number in range(6):

                    logger.info(
                        "Groq processing round %s",
                        round_number + 1,
                    )

                    response = await client.chat.completions.create(
                        model=GROQ_MODEL,
                        messages=messages,
                        tools=groq_tools,
                        tool_choice="auto",
                        temperature=0.2,
                    )

                    assistant_message = response.choices[0].message

                    # Preserve the assistant response,
                    # including any function calls.
                    messages.append(
                        assistant_message.model_dump(
                            exclude_none=True
                        )
                    )

                    tool_calls = assistant_message.tool_calls or []

                    # -------------------------------------
                    # NO MORE FUNCTION CALLS
                    # -------------------------------------

                    if not tool_calls:

                        reply = (
                            assistant_message.content
                            or "I couldn't generate a response."
                        )

                        return reply, tools_used

                    # -------------------------------------
                    # EXECUTE MCP FUNCTIONS
                    # -------------------------------------

                    for tool_call in tool_calls:

                        function_name = tool_call.function.name

                        call_id = tool_call.id

                        try:
                            arguments = json.loads(
                                tool_call.function.arguments or "{}"
                            )

                        except json.JSONDecodeError:

                            logger.exception(
                                "Invalid function arguments: %s",
                                function_name,
                            )

                            messages.append({
                                "role": "tool",
                                "tool_call_id": call_id,
                                "content": json.dumps({
                                    "error": "Invalid JSON arguments."
                                }),
                            })

                            continue

                        logger.info(
                            "Executing MCP tool: %s",
                            function_name,
                        )

                        if function_name not in available_tools:

                            result_text = json.dumps({
                                "error": (
                                    f"Unknown MCP tool: {function_name}"
                                )
                            })

                        else:

                            try:

                                result = await session.call_tool(
                                    function_name,
                                    arguments=arguments,
                                )

                                result_text = serialize_tool_result(
                                    result
                                )

                                tools_used.append(function_name)

                                logger.info(
                                    "Tool completed: %s",
                                    function_name,
                                )

                            except Exception as tool_error:

                                logger.exception(
                                    "MCP tool failed: %s",
                                    function_name,
                                )

                                result_text = json.dumps({
                                    "error": exception_message(
                                        tool_error
                                    )
                                })

                        # Return tool output to Groq.
                        messages.append({
                            "role": "tool",
                            "tool_call_id": call_id,
                            "content": result_text,
                        })

                return (
                    "Maximum analysis steps reached. "
                    "Please narrow your question.",
                    tools_used,
                )

    except BaseException as exc:

        log_full_exception(exc)

        raise RuntimeError(
            exception_message(exc)
        ) from exc


# =========================================================
# FASTAPI APPLICATION
# =========================================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    logger.info(
        "Rujwasco Distribution Intelligence starting."
    )

    yield

    logger.info(
        "Rujwasco Distribution Intelligence stopping."
    )


app = FastAPI(
    title="Rujwasco Distribution Intelligence",
    lifespan=lifespan,
)

app.mount(
    "/static",
    StaticFiles(directory=BASE_DIR / "static"),
    name="static",
)


# =========================================================
# ROUTES
# =========================================================

@app.get("/")
async def home():

    return FileResponse(
        BASE_DIR / "static" / "index.html"
    )


@app.get("/api/health")
async def health():

    return {
        "status": "ok",
        "provider": "Groq",
        "model": GROQ_MODEL,
        "api_key_configured": bool(GROQ_API_KEY),
        "mcp_server_exists": MCP_SERVER_PATH.exists(),
        "python_executable": PYTHON_EXECUTABLE,
    }


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):

    try:

        reply, tools_used = await run_assistant(
            request.message
        )

        return ChatResponse(
            reply=reply,
            model=GROQ_MODEL,
            tools_used=tools_used,
        )

    except Exception as exc:

        log_full_exception(exc)

        error = exception_message(exc)

        logger.error(
            "Assistant request failed: %s",
            error,
        )

        raise HTTPException(
            status_code=502,
            detail=f"Assistant request failed: {error}",
        ) from exc
