import asyncio
import json
import sys
from datetime import datetime
from functools import wraps
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from quart import Quart, Response, jsonify, request
from quart_cors import cors, route_cors
from webai_element_sdk import Context, Element
from webai_element_sdk.comms.messages import Frame, Preview
from webai_element_sdk.element import ElementInputs, ElementOutputs, Input, Output
from webai_element_sdk.element.settings import (
    ElementSettings,
    NumberSetting,
    TextSetting,
)
from werkzeug.exceptions import HTTPException

# Import metrics classes
from .metrics import TPS, TTFT, TPSSummary, TTFTSummary

# Import utility functions
from .utils import (
    PromptRequest,
    _handle_non_streaming_response,
    _parse_request_json,
    _validate_parameter,
    check_queue_capacity,
    generate_openai_error_response,
    queue_capacity_check,
    token_required,
    validate_base64_image,
)


class Inputs(ElementInputs):
    in1 = Input[Frame]()


class Outputs(ElementOutputs):
    preview = Output[Preview]()
    out1 = Output[Frame]()


class Settings(ElementSettings):
    api_key = TextSetting(
        name="api_key",
        display_name="API key",
        description="An optional key to help prevent unwanted network access. A default key is provided.",
        default="",
        sensitive=True,
        required=False,
        hints=["advanced"],
    )

    timeout = NumberSetting[float](
        name="timeout",
        display_name="Endpoint Timeout Seconds",
        description="Timeout for SSE endpoint, default is 0 or no timeout",
        default=0.0,
        min_value=0.0,
        hints=["advanced"],
    )

    max_concurrent_requests = NumberSetting[int](
        name="max_requests",
        display_name="Maximum Concurrent Requests",
        description="Maximum number of requests that can be processed simultaneously in parallel",
        default=1,
        min_value=1,
        hints=["advanced"],
    )

    max_queued_requests = NumberSetting[int](
        name="max_queued_requests",
        display_name="Maximum Queued Requests",
        description="Maximum number of requests that can be queued waiting for processing. Additional requests will be rejected with 429 status.",
        default=10,
        min_value=1,
        hints=["advanced"],
    )


element = Element(
    id=UUID("68f81646-53de-4952-b171-6ee7cdbd9fb0"),
    name="api",
    display_name="API",
    description="Facilitates communication with a Large Language Model. It sends a user-defined prompt (instruction or question) to the model, receives the model's generated response, and returns the response.",
    version="0.1.7",
    inputs=Inputs(),
    outputs=Outputs(),
    settings=Settings(),
    init_run=True,
)


class ElementPreviewServer(Quart):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.startup_event = asyncio.Event()

    async def startup(self):
        await super().startup()
        self.startup_event.set()


app = ElementPreviewServer(__name__)
app = cors(app, allow_origin="*")


# Middleware for common request handling
@app.before_request
async def handle_common_request_logic():
    """
    Middleware to handle CORS preflight requests globally.
    """
    # Handle CORS preflight requests globally
    if request.method == "OPTIONS":
        response = Response("", status=200)
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Headers"] = (
            "Content-Type, X-API-Key, Authorization"
        )
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        return response


requests: asyncio.Queue[PromptRequest] = asyncio.Queue()
curr_request: Optional[PromptRequest] = None

request_ids: asyncio.Queue[int] = asyncio.Queue()
curr_requests: List[Optional[PromptRequest]] = []
num_requests: int = 0
last_request_id: Optional[int] = None

api_key_list: List[str] = []
initialFrame: bool = True

prev_token_count: int = 0
api_tokens: List[str] = []
endpoint_timeout: float = 0.0
first_token: bool = True
max_queued_requests: int = 10

request_ttft: float = 0
ttft: List[TTFT] = []
tps: List[TPS] = []
tps_summary: TPSSummary = TPSSummary()
ttft_summary: TTFTSummary = TTFTSummary()


def set_api_token(secret_value: str):
    global api_tokens
    api_tokens = secret_value.split(",")


def get_api_tokens():
    """Helper function to get current API tokens for the decorator."""
    global api_tokens
    return api_tokens


async def check_queue_capacity_wrapper():
    """Wrapper function for queue capacity check that uses global variables."""
    global max_queued_requests, requests
    return await check_queue_capacity(requests, max_queued_requests)


# Create decorator instances with the helper functions
token_required_decorator = token_required(get_api_tokens)
queue_capacity_check_decorator = queue_capacity_check(check_queue_capacity_wrapper)


@element.startup
async def startup(ctx: Context[Inputs, Outputs, Settings]):
    global api_tokens, endpoint_timeout, request_ids, curr_requests, ttft, tps, first_token, max_queued_requests
    user_api_key = ctx.settings.api_key.value
    endpoint_timeout = ctx.settings.timeout.value
    max_queued_requests = ctx.settings.max_queued_requests.value
    first_token = True
    api_tokens = (
        []
        if user_api_key.strip() == ""
        else [t.strip() for t in user_api_key.split(",")]
    )

    tps = [TPS()] * ctx.settings.max_concurrent_requests.value
    ttft = [TTFT()] * ctx.settings.max_concurrent_requests.value
    curr_requests = [None] * ctx.settings.max_concurrent_requests.value

    for i in range(ctx.settings.max_concurrent_requests.value):
        await request_ids.put(i)

    port = ctx.preview_port
    print(f"Serving on {port}")
    asyncio.create_task(app.run_task(host="0.0.0.0", port=port, debug=False))


@element.executor
async def run(ctx: Context[Inputs, Outputs, Settings]):
    """
    OpenAI Response Structure
    {
    "choices": [
        {
        "finish_reason": "stop",
        "index": 0,
        "message": {
            "content": "The 2020 World Series was played in Texas at Globe Life Field in Arlington.",
            "role": "assistant"
        },
        "logprobs": null
        }
    ],
    "created": 1677664795,
    "id": "chatcmpl-7QyqpwdfhqwajicIEznoc6Q47XAyW",
    "model": "gpt-4o-mini",
    # we can add these later if they need them
    "usage": {
        "completion_tokens": 17,
        "prompt_tokens": 57,
        "total_tokens": 74
    }
    }
    """
    global initialFrame, prev_token_count, curr_request, first_token, ttft, request_ttft, tps, curr_requests, request_ids, num_requests, tps_summary, ttft_summary, last_request_id

    while True:
        if num_requests == 0:
            request_id = await request_ids.get()
            last_request_id = request_id

            curr_requests[request_id] = await requests.get()
            num_requests += 1

            await tps[request_id].start()
            await ttft[request_id].start()

            current_request = curr_requests[request_id]
            yield ctx.outputs.out1(
                Frame(
                    ndframe=None,
                    other_data={
                        "api": (
                            current_request.messages
                            if current_request is not None
                            else []
                        ),
                        "requestId": request_id,
                    },
                )
            )

        else:
            # there's capacity for another client, check if there is one waiting
            if (num_requests > 0) and (
                num_requests < ctx.settings.max_concurrent_requests.value
            ):
                try:
                    req = requests.get_nowait()
                    request_id = await request_ids.get()
                    curr_requests[request_id] = req
                    num_requests += 1

                    await ttft[request_id].start()

                    current_request = curr_requests[request_id]
                    yield ctx.outputs.out1(
                        Frame(
                            ndframe=None,
                            other_data={
                                "api": (
                                    current_request.messages
                                    if current_request is not None
                                    else []
                                ),
                                "requestId": request_id,
                            },
                        )
                    )
                except asyncio.QueueEmpty:
                    pass

            frame = ctx.inputs.in1.value  # TODO: make sure we don't propogate old data
            if (
                frame is None
                or "init" in frame.other_data
                or "message" not in frame.other_data
                or (frame.other_data["token_number"] == prev_token_count)
            ):
                return

            request_id = frame.other_data.get("requestId", last_request_id)

            if frame.other_data.get("token_number") == 1:
                ttft_value = await ttft[request_id].stop()
                await tps[request_id].start()
                await ttft_summary.add(ttft_value)

            output = {
                "choices": [
                    {
                        "finish_reason": (
                            None if not frame.other_data["done"] else "stop"
                        ),
                        "index": 0,
                        "message": {
                            "content": frame.other_data["message"],
                            "role": "assistant",
                        },
                        "logprobs": None,
                    }
                ],
                "created": 1677664795,
                "id": "webaichat-6ee7cdbd9fb0",
                "model": "nav",
                "object": "chat.completion",
            }

            # Only include usage when the frame is done (final frame)
            if frame.other_data["done"]:
                usage_data = frame.other_data.get("usage", {})
                output["usage"] = usage_data
                if usage_data:  # Only print if there's actual usage data
                    print(f"API: Including usage data: {usage_data}")

            if frame.other_data["done"] == True:
                tps_value = await tps[request_id].stop(prev_token_count - 1)
                await tps_summary.add(tps_value)

                output["metrics"] = {"tps": tps_value, "ttft": ttft[request_id].value}

                if curr_requests[request_id] is not None:
                    await curr_requests[request_id].queue.put(output)
                await request_ids.put(request_id)
                num_requests -= 1

                curr_request = None
                first_token = True
                request_ttft = 0
                prev_token_count = frame.other_data["token_number"]
                continue
            else:
                prev_token_count = frame.other_data["token_number"]
                if curr_requests[request_id] is not None:
                    await curr_requests[request_id].queue.put(output)
                return


@app.route("/")
async def index():
    return "Please use the API endpoint /prompt"


@app.route("/config")
@route_cors(
    allow_headers=["Content-Type", "Access-Control-Allow-Origin"],
    allow_methods=["GET"],
    allow_origin=["*"],
)
async def config():
    return jsonify({"type": "prompt", "endpoint": "/prompt"})


@app.route("/prompt", methods=["POST", "OPTIONS"])
@route_cors(
    allow_headers=["Content-Type", "X-API-Key"],
    allow_methods=["POST"],
)
@queue_capacity_check_decorator
async def prompt():
    @token_required_decorator
    async def handle_prompt():
        global endpoint_timeout

        # TODO: Check the key in the header and make sure it's in the setting's list
        # api_key = request.headers.get("x-api-key")
        # if api_key != user_api_key:
        #   abort(401, description="Invalid API key")

        data = await request.get_json()
        if "message" not in data:
            return generate_openai_error_response(
                status_code=400,
                message="Missing 'message' field in JSON.",
                error_type="invalid_request_error",
                error_code="missing_field",
                param="message",
            )

        # Pass the prompt
        # should come in as a list in the openAI style
        promptRequest = PromptRequest(data["message"])
        queue = promptRequest.queue
        await requests.put(promptRequest)

        # Wait for responses and stream them out as they are received
        # TODO: Are these the correct headers?
        headers = {
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }

        async def event_stream(queue: asyncio.Queue):
            while True:
                try:
                    response = await queue.get()

                    if "choices" not in response or not isinstance(
                        response["choices"], list
                    ):
                        print("Invalid response format")
                        continue

                    # TODO: Structure response as OpenAPI chunk messages before sending back
                    yield json.dumps(response)
                    if response["choices"][-1].get("finish_reason") is not None:
                        break
                except Exception as e:
                    print(f"API: Error: {e}")
                    continue

        resp = Response(event_stream(queue), headers=headers)
        resp.timeout = endpoint_timeout if endpoint_timeout > 0 else None
        return resp

    return await handle_prompt()


@app.route("/metrics", methods=["GET"])
async def metrics():
    global ttft
    return {
        "ttft": await ttft_summary.summary(),
        "tps": await tps_summary.summary(),
    }


@app.route("/v1/chat/completions", methods=["POST", "OPTIONS"])
@route_cors(
    allow_headers=["Content-Type", "X-API-Key", "Authorization"],
    allow_methods=["POST", "OPTIONS"],
)
@queue_capacity_check_decorator
async def chat_completions():
    """OpenAI-compatible chat completions endpoint."""

    @token_required_decorator
    async def handle_chat_completion():
        global endpoint_timeout

        data, error_resp = await _parse_request_json(request)
        if error_resp:
            return error_resp

        # Validate 'model'
        model_val, error_resp = _validate_parameter(
            data, "model", required=True, expected_type=str, can_be_empty_str=False
        )
        if error_resp:
            return error_resp

        # Validate 'messages' (already partially validated, enhancing here)
        messages_val = data.get("messages")

        if messages_val is None:
            return generate_openai_error_response(
                status_code=400,
                message="Missing 'messages' field in request body.",
                error_type="invalid_request_error",
                error_code="missing_field",
                param="messages",
            )
        if not isinstance(messages_val, list):
            return generate_openai_error_response(
                status_code=400,
                message="'messages' must be a list.",
                error_type="invalid_request_error",
                error_code="invalid_type",
                param="messages",
            )
        if not messages_val:  # Check if list is empty
            return generate_openai_error_response(
                status_code=400,
                message="'messages' list cannot be empty.",
                error_type="invalid_request_error",
                error_code="invalid_value",
                param="messages",
            )

        validated_messages = []
        for i, msg_dict in enumerate(messages_val):
            param_path_prefix = f"messages.[{i}]"
            if not isinstance(msg_dict, dict):
                return generate_openai_error_response(
                    status_code=400,
                    message=f"Each message in 'messages' list must be an object. Error at index {i}.",
                    error_type="invalid_request_error",
                    error_code="invalid_type",
                    param=param_path_prefix,
                )

            role, error_resp = _validate_parameter(
                msg_dict,
                "role",
                required=True,
                expected_type=str,
                allowed_values=["system", "user", "assistant"],
                param_path=f"{param_path_prefix}.role",
            )
            if error_resp:
                return error_resp

            content, error_resp = _validate_parameter(
                msg_dict,
                "content",
                required=True,
                expected_type=(str, list),  # Ensure content can be str or list
                can_be_empty_str=True,
                param_path=f"{param_path_prefix}.content",
            )
            if error_resp:
                return error_resp

            # Normalize content based on type
            if isinstance(content, str):
                # String content - keep as is
                normalized_msg = {"role": role, "content": content}
            elif isinstance(content, list):
                # List content - normalize input_text and input_image types
                normalized_content = []
                for item in content:
                    if item.get("type") == "input_text":
                        # Convert input_text to text format
                        normalized_content.append(
                            {"type": "text", "text": item.get("text", "")}
                        )
                    elif item.get("type") == "input_image":
                        # Convert input_image to image_url format
                        normalized_content.append(
                            {
                                "type": "image_url",
                                "image_url": {"url": item.get("image_url", "")},
                            }
                        )
                    else:
                        # Keep other formats as-is (text, image_url, etc.)
                        normalized_content.append(item)
                normalized_msg = {"role": role, "content": normalized_content}
            else:
                # Other content types - keep as is
                normalized_msg = {"role": role, "content": content}

            # Validate base64 image data in normalized message
            if isinstance(normalized_msg.get("content"), list):
                for content_idx, content_item in enumerate(normalized_msg["content"]):
                    if (
                        isinstance(content_item, dict)
                        and content_item.get("type") == "image_url"
                        and isinstance(content_item.get("image_url"), dict)
                    ):

                        image_url = content_item["image_url"].get("url", "")
                        if image_url:
                            param_path = f"{param_path_prefix}.content.[{content_idx}].image_url.url"
                            base64_err = validate_base64_image(image_url, param_path)
                            if base64_err:
                                return base64_err

            validated_messages.append(normalized_msg)

        # Parameters not yet supported - return 501
        # Note skipped properties are not supported yet, but we're not blocking
        # them as some 3rd party applications pass them in. For now they are just ignored
        unsupported_params = [
            "audio",
            "function_call",
            "functions",
            "logit_bias",
            "logprobs",
            "metadata",
            "modalities",
            "parallel_tool_calls",
            "prediction",
            "reasoning_effort",
            "response_format",
            "seed",
            "service_tier",
            "stop",
            "store",
            "tool_choice",
            "tools",
            "top_logprobs",
            "user",
            "web_search_options",
            "stream_options",  # stream_options is only used when stream is true, but not supported yet
        ]

        for param_name in unsupported_params:
            if param_name in data and data.get(param_name) is not None:
                return generate_openai_error_response(
                    status_code=501,
                    message=f"The parameter '{param_name}' is not yet supported.",
                    error_type="api_error",
                    error_code="feature_not_implemented",
                    param=param_name,
                )

        # Validate supported optional parameters
        stream_val, error_resp = _validate_parameter(
            data, "stream", required=False, expected_type=bool, default_value=False
        )
        if error_resp:
            return error_resp

        temperature_val, error_resp = _validate_parameter(
            data,
            "temperature",
            required=False,
            expected_type=(int, float),
            default_value=1.0,
            min_value=0.0,
            max_value=2.0,
        )
        if error_resp:
            return error_resp

        top_p_val, error_resp = _validate_parameter(
            data,
            "top_p",
            required=False,
            expected_type=(int, float),
            default_value=1.0,
            min_value=0.0,
            max_value=1.0,
        )  # OpenAI spec implies 0-1 range for top_p
        if error_resp:
            return error_resp

        # Validate max_output_tokens parameter
        max_output_tokens_val, error_resp = _validate_parameter(
            data,
            "max_output_tokens",
            required=False,
            expected_type=int,
            min_value=1,
            max_value=100000,  # Reasonable upper limit
        )
        if error_resp:
            return error_resp

        print(
            f"API: Normalized messages for /v1/chat/completions: {validated_messages}"
        )

        try:
            promptRequest = PromptRequest(validated_messages)
            queue = promptRequest.queue
            await requests.put(promptRequest)
        except Exception as e:
            print(f"API: Error creating or queuing prompt request: {e}")
            return generate_openai_error_response(
                status_code=500,
                message=f"Error processing request: {str(e)}",
                error_type="api_error",
                error_code="internal_error",
            )

        if not stream_val:
            return await _handle_non_streaming_response(queue, "chat.completion")

        # For streaming responses, set up SSE headers
        headers = {
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }

        async def event_stream(queue: asyncio.Queue):
            while True:
                try:
                    response = await queue.get()
                    if "choices" not in response or not isinstance(
                        response["choices"], list
                    ):
                        print("Invalid response format")
                        continue

                    # Format response to match OpenAPI's structure
                    openai_response = {
                        "id": response["id"],
                        "object": "chat.completion",
                        "created": response["created"],
                        "model": response["model"],
                        "choices": response["choices"],
                        "usage": response.get("usage", {}),
                    }

                    if "metrics" in response:
                        openai_response["metrics"] = response["metrics"]

                    # Format as proper SSE with data: prefix
                    yield f"data: {json.dumps(openai_response)}\n\n"
                    if response["choices"][-1].get("finish_reason") is not None:
                        break
                except Exception as e:
                    print(f"API: Error: {e}")
                    continue

        resp = Response(event_stream(queue), headers=headers)
        resp.timeout = endpoint_timeout if endpoint_timeout > 0 else None
        return resp

    return await handle_chat_completion()


# https://platform.openai.com/docs/api-reference/responses
@app.route("/v1/responses", methods=["POST", "OPTIONS"])
@route_cors(
    allow_headers=["Content-Type", "X-API-Key", "Authorization"],
    allow_methods=["POST", "OPTIONS"],
)
@queue_capacity_check_decorator
async def responses():
    @token_required_decorator
    async def handle_response():
        global endpoint_timeout

        data, error_resp = await _parse_request_json(request)
        if error_resp:
            return error_resp

        # Required fields validation
        input_val, error_resp = _validate_parameter(
            data,
            "input",
            required=True,
            expected_type=(str, list),
            can_be_empty_str=False,
        )
        if error_resp:
            return error_resp

        model_val, error_resp = _validate_parameter(
            data, "model", required=True, expected_type=str, can_be_empty_str=False
        )
        if error_resp:
            return error_resp

        # Check for parameters that are not yet supported
        unsupported_params = [
            "parallel_tool_calls",
            "background",
            "include",
            "instructions",
            "reasoning",
            "previous_response_id",
            "service_tier",
            "store",
            "tools",
            "tool_choice",
            "user",
            "truncation",
        ]

        for param_name in unsupported_params:
            if param_name in data and data.get(param_name) is not None:
                return generate_openai_error_response(
                    status_code=501,
                    message=f"The '{param_name}' parameter is not yet supported.",
                    error_type="api_error",
                    error_code="feature_not_implemented",
                    param=param_name,
                )

        # Optional fields validation
        stream_val = data.get("stream", False)
        if not isinstance(stream_val, bool):
            return generate_openai_error_response(
                status_code=400,
                message="'stream' must be a boolean.",
                error_type="invalid_request_error",
                error_code="invalid_type",
                param="stream",
            )

        temperature_val = data.get("temperature", 1.0)
        if not isinstance(temperature_val, (int, float)) or not (
            0.0 <= temperature_val <= 2.0
        ):
            return generate_openai_error_response(
                status_code=400,
                message="'temperature' must be a number between 0.0 and 2.0.",
                error_type="invalid_request_error",
                error_code="invalid_value",
                param="temperature",
            )

        text_val = data.get("text")
        if text_val is not None and not isinstance(text_val, dict):
            return generate_openai_error_response(
                status_code=400,
                message="'text' must be an object.",
                error_type="invalid_request_error",
                error_code="invalid_type",
                param="text",
            )

        top_p_val = data.get("top_p", 1.0)
        if not isinstance(top_p_val, (int, float)):
            return generate_openai_error_response(
                status_code=400,
                message="'top_p' must be a number.",
                error_type="invalid_request_error",
                error_code="invalid_type",
                param="top_p",
            )

        # Create a prompt request with a single message
        messages = []
        if isinstance(input_val, str):
            messages = [{"role": "user", "content": input_val}]
        elif isinstance(input_val, list):
            # Normalize the input format for downstream compatibility
            normalized_messages = []
            for msg in input_val:
                if isinstance(msg, dict) and "role" in msg and "content" in msg:
                    # Handle message object format
                    normalized_msg = {"role": msg["role"]}
                    content = msg["content"]

                    # Normalize content items to the format downstream element expects
                    normalized_content = []
                    for item in content:
                        if item.get("type") == "input_text":
                            # Convert input_text to text format
                            normalized_content.append(
                                {"type": "text", "text": item.get("text", "")}
                            )
                        elif item.get("type") == "input_image":
                            # Convert input_image to image_url format
                            normalized_content.append(
                                {
                                    "type": "image_url",
                                    "image_url": {"url": item.get("image_url", "")},
                                }
                            )
                        else:
                            # Keep other formats as-is (text, image_url, etc.)
                            normalized_content.append(item)
                    normalized_msg["content"] = normalized_content

                    # Validate base64 image data in normalized message
                    if isinstance(normalized_msg.get("content"), list):
                        for content_idx, content_item in enumerate(
                            normalized_msg["content"]
                        ):
                            if (
                                isinstance(content_item, dict)
                                and content_item.get("type") == "image_url"
                                and isinstance(content_item.get("image_url"), dict)
                            ):

                                image_url = content_item["image_url"].get("url", "")
                                if image_url:
                                    param_path = f"input.[{input_val.index(msg)}].content.[{content_idx}].image_url.url"
                                    base64_err = validate_base64_image(
                                        image_url, param_path
                                    )
                                    if base64_err:
                                        return base64_err

                    normalized_messages.append(normalized_msg)
                else:
                    # Handle direct content item format (fallback)
                    normalized_messages.append(msg)

            messages = normalized_messages
            print(f"API: Normalized messages for /v1/responses: {messages}")
        else:
            return generate_openai_error_response(
                status_code=400,
                message="'input' must be a string or a list.",
                error_type="invalid_request_error",
                error_code="invalid_type",
                param="input",
            )

        try:
            promptRequest = PromptRequest(messages)
            queue = promptRequest.queue
            await requests.put(promptRequest)
        except Exception as e:
            print(f"API: Error creating or queuing prompt request: {e}")
            return generate_openai_error_response(
                status_code=500,
                message=f"Error processing request: {str(e)}",
                error_type="api_error",
                error_code="internal_error",
            )

        if not stream_val:
            # For non-streaming, collect all responses and return a single response
            return await _handle_non_streaming_response(queue, "response")

        # For streaming responses, set up SSE headers
        headers = {
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }

        async def event_stream(queue: asyncio.Queue):
            while True:
                try:
                    response = await queue.get()
                    if "choices" not in response or not isinstance(
                        response["choices"], list
                    ):
                        print("Invalid response format")
                        continue

                    # Format response to match OpenAI's structure
                    openai_response = {
                        "id": response["id"],
                        "object": "response",
                        "created": response["created"],
                        "model": response["model"],
                        "choices": [
                            {
                                "text": response["choices"][0]["message"]["content"]
                                or "",
                                "index": 0,
                                "logprobs": None,
                                "finish_reason": response["choices"][0].get(
                                    "finish_reason"
                                ),
                            }
                        ],
                        "usage": response.get("usage", {}),
                    }

                    if "metrics" in response:
                        openai_response["metrics"] = response["metrics"]

                    # Format as proper SSE with data: prefix
                    yield f"data: {json.dumps(openai_response)}\n\n"
                    if response["choices"][-1].get("finish_reason") is not None:
                        break
                except Exception as e:
                    print(f"API: Error: {e}")
                    continue

        resp = Response(event_stream(queue), headers=headers)
        resp.timeout = endpoint_timeout if endpoint_timeout > 0 else None
        return resp

    return await handle_response()


@app.route("/v1/completions", methods=["POST", "OPTIONS"])
@route_cors(
    allow_headers=["Content-Type", "X-API-Key", "Authorization"],
    allow_methods=["POST", "OPTIONS"],
)
@queue_capacity_check_decorator
async def completions():
    """Handle requests to the /v1/completions endpoint (OpenAI legacy)."""

    @token_required_decorator
    async def handle_completion():
        data, error_response = await _parse_request_json(request)
        if error_response:
            return error_response
        if not data:
            return generate_openai_error_response(
                400,
                "Request body is missing or not valid JSON.",
                "invalid_request_error",
            )

        # Define supported parameters for the legacy /v1/completions endpoint
        SUPPORTED_COMPLETIONS_PARAMS = [
            "model",
            "prompt",
            "temperature",
            "top_p",
            "stream",
            "user",
            "suffix",
            "max_tokens",
            "n",
            "logprobs",
            "echo",
            "stop",
            "presence_penalty",
            "frequency_penalty",
            "best_of",
            "logit_bias",
            "seed",
            # Parameters like 'functions', 'function_call', 'tools', 'tool_choice' are NOT supported
        ]

        # Check for unsupported parameters
        for param_name in data.keys():
            if param_name not in SUPPORTED_COMPLETIONS_PARAMS:
                return generate_openai_error_response(
                    status_code=501,
                    message=f"The parameter `{param_name}` is not supported by this endpoint.",
                    error_type="api_error",
                    error_code="feature_not_implemented",
                    param=param_name,
                )

        # Validate required and optional parameters
        model, error_response = _validate_parameter(
            data, "model", required=True, expected_type=str, can_be_empty_str=False
        )
        if error_response:
            return error_response

        prompt, error_response = _validate_parameter(
            data, "prompt", required=True, expected_type=str, can_be_empty_str=True
        )
        if error_response:
            return error_response

        temperature, error_response = _validate_parameter(
            data,
            "temperature",
            required=False,
            expected_type=(int, float),
            default_value=1.0,
            min_value=0.0,
            max_value=2.0,
        )
        if error_response:
            return error_response

        top_p, error_response = _validate_parameter(
            data,
            "top_p",
            required=False,
            expected_type=(int, float),
            default_value=1.0,
            min_value=0.0,
            max_value=1.0,
        )
        if error_response:
            return error_response

        stream, error_response = _validate_parameter(
            data, "stream", required=False, expected_type=bool, default_value=False
        )
        if error_response:
            return error_response

        user, error_response = _validate_parameter(
            data, "user", required=False, expected_type=str, can_be_empty_str=True
        )
        if error_response:
            return error_response

        suffix, error_response = _validate_parameter(
            data, "suffix", required=False, expected_type=str, can_be_empty_str=True
        )
        if error_response:
            return error_response

        max_tokens, error_response = _validate_parameter(
            data,
            "max_tokens",
            required=False,
            expected_type=int,
            min_value=1,
            max_value=1000,
        )
        if error_response:
            return error_response

        n, error_response = _validate_parameter(
            data, "n", required=False, expected_type=int, min_value=1, max_value=100
        )
        if error_response:
            return error_response

        logprobs, error_response = _validate_parameter(
            data, "logprobs", required=False, expected_type=bool, default_value=False
        )
        if error_response:
            return error_response

        echo, error_response = _validate_parameter(
            data, "echo", required=False, expected_type=bool, default_value=False
        )
        if error_response:
            return error_response

        stop, error_response = _validate_parameter(
            data, "stop", required=False, expected_type=str, can_be_empty_str=True
        )
        if error_response:
            return error_response

        presence_penalty, error_response = _validate_parameter(
            data,
            "presence_penalty",
            required=False,
            expected_type=(int, float),
            min_value=-2.0,
            max_value=2.0,
        )
        if error_response:
            return error_response

        frequency_penalty, error_response = _validate_parameter(
            data,
            "frequency_penalty",
            required=False,
            expected_type=(int, float),
            min_value=-2.0,
            max_value=2.0,
        )
        if error_response:
            return error_response

        best_of, error_response = _validate_parameter(
            data,
            "best_of",
            required=False,
            expected_type=int,
            min_value=1,
            max_value=100,
        )
        if error_response:
            return error_response

        logit_bias, error_response = _validate_parameter(
            data,
            "logit_bias",
            required=False,
            expected_type=dict,
            can_be_empty_str=True,
        )
        if error_response:
            return error_response

        seed, error_response = _validate_parameter(
            data,
            "seed",
            required=False,
            expected_type=int,
            min_value=1,
            max_value=1000000,
        )
        if error_response:
            return error_response

        # Create a prompt request with a single user message containing the prompt
        messages = [{"role": "user", "content": prompt}]  # Use validated prompt
        try:
            promptRequest = PromptRequest(messages)
            queue = promptRequest.queue
            await requests.put(promptRequest)
        except Exception as e:
            print(f"API: Error creating or queuing prompt request: {e}")
            return generate_openai_error_response(
                status_code=500,
                message=f"Error processing request: {str(e)}",
                error_type="api_error",
                error_code="internal_error",
            )

        if not stream:
            # For non-streaming, collect all responses and return a single response
            return await _handle_non_streaming_response(queue, "text_completion")

        # For streaming responses, set up SSE headers
        headers = {
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }

        async def event_stream(queue: asyncio.Queue):
            while True:
                try:
                    response = await queue.get()
                    if "choices" not in response or not isinstance(
                        response["choices"], list
                    ):
                        print("Invalid response format")
                        continue

                    # Format response to match OpenAI's structure
                    openai_response = {
                        "id": response["id"],
                        "object": "text_completion",
                        "created": response["created"],
                        "model": response["model"],
                        "choices": [
                            {
                                "text": response["choices"][0]["message"]["content"]
                                or "",
                                "index": 0,
                                "logprobs": None,
                                "finish_reason": response["choices"][0].get(
                                    "finish_reason"
                                ),
                            }
                        ],
                        "usage": response.get("usage", {}),
                    }

                    if "metrics" in response:
                        openai_response["metrics"] = response["metrics"]

                    # Format as proper SSE with data: prefix
                    yield f"data: {json.dumps(openai_response)}\n\n"
                    if response["choices"][-1].get("finish_reason") is not None:
                        break
                except Exception as e:
                    print(f"API: Error: {e}")
                    continue

        resp = Response(event_stream(queue), headers=headers)
        resp.timeout = endpoint_timeout if endpoint_timeout > 0 else None
        return resp

    return await handle_completion()


@app.route("/v1/models", methods=["GET", "OPTIONS"])
@route_cors(
    allow_headers=["Content-Type", "X-API-Key", "Authorization"],
    allow_methods=["GET", "OPTIONS"],
)
async def get_models():
    @token_required_decorator
    async def handle_get_models():
        # TODO: This endpoint should eventually return a list of available running models
        # that inference can be generated against, similar to OpenAI's /v1/models response.
        # For example: {"object": "list", "data": [{"id": "model-name", "object": "model", ...}]}
        # https://platform.openai.com/docs/api-reference/models
        return (
            jsonify(
                {
                    "data": [
                        {
                            "id": "webai/llm",
                            "object": "model",
                            "created": 1747937728,
                            "owned_by": "webai",
                        }
                    ],
                    "object": "list",
                }
            ),
            200,
        )

    return await handle_get_models()


@app.route("/v1/models/<string:model_id>", methods=["GET", "OPTIONS"])
@route_cors(
    allow_headers=["Content-Type", "X-API-Key", "Authorization"],
    allow_methods=["GET", "OPTIONS"],
)
async def get_specific_model(model_id: str):
    @token_required_decorator
    async def handle_get_specific_model(model_id_inner: str):
        # TODO: This endpoint should eventually return a running model (or reject the request if the model is not running)
        # that inference can be generated against, similar to OpenAI's /v1/models/:model response.
        # For example: {"object": "list", "data": [{"id": "model-name", "object": "model", ...}]}
        # https://platform.openai.com/docs/api-reference/models/retrieve
        if model_id_inner == "webai":
            return (
                jsonify(
                    {
                        "id": "webai",
                        "object": "model",
                        "created": 1747937728,
                        "owned_by": "webai",
                    }
                ),
                200,
            )
        else:
            return generate_openai_error_response(
                status_code=404,
                message=f"The model '{model_id_inner}' does not exist.",
                error_type="invalid_request_error",
                error_code="model_not_found",
            )

    return await handle_get_specific_model(model_id)


@app.route("/v1/embeddings", methods=["POST", "OPTIONS"])
@route_cors(
    allow_headers=["Content-Type", "X-API-Key", "Authorization"],
    allow_methods=["POST", "OPTIONS"],
)
async def create_embeddings():
    @token_required_decorator
    async def handle_create_embeddings():
        # This endpoint is not yet implemented, so we return an error.
        # https://platform.openai.com/docs/api-reference/embeddings
        return generate_openai_error_response(
            status_code=501,
            message="The embeddings endpoint is not yet supported. Please check back later.",
            error_type="api_error",
            error_code="endpoint_not_implemented",
        )

    return await handle_create_embeddings()


# https://platform.openai.com/docs/api-reference/usage/audio_transcriptions
@app.route("/v1/audio/transcriptions", methods=["POST", "OPTIONS"])
@route_cors(
    allow_headers=["Content-Type", "X-API-Key", "Authorization"],
    allow_methods=["POST", "OPTIONS"],
)
async def create_audio_transcription():
    @token_required_decorator
    async def handle_create_audio_transcription():
        # TODO: Add request body validation as per OpenAI spec when implementing
        # https://platform.openai.com/docs/api-reference/audio/createTranscription
        return generate_openai_error_response(
            status_code=501,
            message="Audio transcriptions endpoint is not yet supported. Please check back later.",
            error_type="api_error",
            error_code="endpoint_not_implemented",
        )

    return await handle_create_audio_transcription()


# https://platform.openai.com/docs/api-reference/usage/audio_transcriptions
@app.route("/v1/audio/translations", methods=["POST", "OPTIONS"])
@route_cors(
    allow_headers=["Content-Type", "X-API-Key", "Authorization"],
    allow_methods=["POST", "OPTIONS"],
)
async def create_audio_translation():
    @token_required_decorator
    async def handle_create_audio_translation():
        # TODO: Add request body validation as per OpenAI spec when implementing
        # https://platform.openai.com/docs/api-reference/audio/createTranslation
        return generate_openai_error_response(
            status_code=501,
            message="Audio translations endpoint is not yet supported. Please check back later.",
            error_type="api_error",
            error_code="endpoint_not_implemented",
        )

    return await handle_create_audio_translation()
