import asyncio
import base64
import time
from typing import Any, Literal, TypedDict
from uuid import UUID

import requests
from openai import OpenAI
from webai_element_sdk.comms.messages import ColorFormat, Frame
from webai_element_sdk.element import Context, Element
from webai_element_sdk.element.settings import (
    BoolSetting,
    ElementSettings,
    NumberSetting,
    TextSetting,
)
from webai_element_sdk.element.variables import (
    ElementInputs,
    ElementOutputs,
    Input,
    Output,
)

from webai_element_utils.logs import setup_element_logger

logger = setup_element_logger("Llama4")

class Settings(ElementSettings):
    model = TextSetting(
        name="base_model_arch",
        display_name="Base Model Architecture",
        description="The fundamental structure or design of a machine learning model.",
        default="Meta/Llama-4-Maverick",
        valid_values=[
            "Meta/Llama-4-Maverick",
            "Meta/Llama-4-Scout",
        ],
        required=True,
        hints=["dropdown"],
    )
    api_key = TextSetting(
        name="api_key",
        display_name="API key",
        description="Llama 4 API key",
        default="",
        sensitive=True,
        required=True,
    )
    temperature = NumberSetting[float](
        name="temperature",
        display_name="Temperature",
        description="Temperature for the model",
        default=0.7,
        min_value=0.0,
        max_value=1.0,
        hints=["advanced"],
    )
    max_completion_tokens = NumberSetting[int](
        name="max_completion_tokens",
        display_name="Max Completion Tokens",
        description="Maximum number of completion tokens",
        default=300,
        max_value=10000,
        min_value=1,
        hints=["advanced"],
    )

    chat_history = BoolSetting(
        name="chat_history",
        display_name="Chat History",
        description="Whether to use chat history",
        default=True,
    )


class Inputs(ElementInputs):
    in1 = Input[Frame]()


class Outputs(ElementOutputs):
    out1 = Output[Frame]()


element = Element(
    id=UUID("e54b5bf8-f954-4dba-a111-c45728c46e8e"),
    name="llama4",
    version="0.0.4",
    display_name="Llama4",
    description="Llama4 API with media ingestion support",
    settings=Settings(),
    inputs=Inputs(),
    outputs=Outputs(),
    is_inference=True,
)

model = None
model_value = None


class ChatEntry(TypedDict):
    role: Literal["user", "assistant", "system"]
    content: Any


chat_history: list[ChatEntry] = []


def image_to_base64(image_input):
    """
    Accepts either:
    - a dict with 'url' as a remote image URL
    - a base64 data URL string
    Returns just the base64 portion (no data:image/... prefix).
    """
    if isinstance(image_input, dict) and "url" in image_input:
        url = image_input["url"]

        if url.startswith("data:image/") and ";base64," in url:
            return url.split(";base64,")[1]

        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; MyApp/1.0; +https://example.com/bot)"
        }

        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return base64.b64encode(response.content).decode("utf-8")

    raise ValueError("Invalid image input. Must be a dict with a 'url' key.")


def _ingest_media_data(input_frame: Frame):
    """Ingest media data from the video audio processor and add to chat history"""
    global chat_history
    
    other_data = input_frame.other_data
    media_type = other_data.get("media_type")
    
    if media_type == "video_transcription":
        transcript = other_data.get("transcript", "")
        source_file = other_data.get("source_file", "")
        whisper_model = other_data.get("whisper_model", "")
        
        # Add transcript to chat history as user message
        chat_history.append({
            "role": "user",
            "content": f"[Video Transcript from {source_file}]\n{transcript}"
        })
        
        logger.info(f"Added transcript from {source_file} to chat history: {len(transcript)} characters")
        
    elif media_type == "video_frame":
        frame_data = other_data.get("frame_data", {})
        source_file = other_data.get("source_file", "")
        frame_index = other_data.get("frame_index", 0)
        timestamp = frame_data.get("timestamp", 0)
        
        # Add video frame to chat history as user message with multimodal content
        chat_history.append({
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"[Video frame {frame_index} from {source_file} at {timestamp}s]"
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{frame_data.get('image_base64', '')}"}
                }
            ]
        })
        
        logger.info(f"Added video frame {frame_index} from {source_file} to chat history")
        
    elif media_type == "image_file":
        image_base64 = other_data.get("image_base64", "")
        source_directory = other_data.get("source_directory", "")
        image_index = other_data.get("image_index", 0)
        
        # Add image to chat history as user message with multimodal content
        chat_history.append({
            "role": "user", 
            "content": [
                {
                    "type": "text",
                    "text": f"[Image {image_index} from {source_directory}]"
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
                }
            ]
        })
        
        logger.info(f"Added image {image_index} from {source_directory} to chat history")


@element.startup
async def startup(ctx: Context[Inputs, Outputs, Settings]):
    global model, model_value
    model_arch_dict = {
        "Meta/Llama-4-Maverick": "Llama-4-Maverick-17B-128E-Instruct-FP8",
        "Meta/Llama-4-Scout": "Llama-4-Scout-17B-16E-Instruct-FP8",
    }

    try:
        model_value = model_arch_dict.get(ctx.settings.model.value)
        model = OpenAI(
            api_key=ctx.settings.api_key.value,
            base_url="https://api.llama.com/compat/v1/",
        )
        logger.info(f"Model initialized: {model_value}")
    except Exception as e:
        logger.exception(f"Error during startup: {e}")


@element.executor
async def llm_inference(ctx: Context[Inputs, Outputs, Settings]):
    global model, model_value, chat_history

    use_chat_history = ctx.settings.chat_history.value
    input_frame = ctx.inputs.in1.value
    
    # Check if this is media data that should be ingested without response
    if input_frame.other_data.get("media_type") in ["video_transcription", "video_frame", "image_file"]:
        _ingest_media_data(input_frame)
        # Don't yield any response for media ingestion
        return
    
    # Only process API messages for actual LLM inference
    api_messages = input_frame.other_data.get("api", [])

    if not api_messages:
        api_messages = [
            {"role": "user", "content": input_frame.other_data.get("message")}
        ]

    if not isinstance(api_messages, list) or not api_messages:
        logger.warning("API input is missing or not a list.")
        return

    # Start with existing chat history if enabled
    current_history: list[ChatEntry] = []
    if use_chat_history:
        current_history.extend(chat_history)

    # Process and add new API messages
    for turn in api_messages:
        role = turn.get("role")
        content = turn.get("content")

        if isinstance(content, list):
            processed_content = []
            for part in content:
                if part.get("type") == "image_url":
                    image_url = part.get("image_url")
                    base64_str = image_to_base64(image_url)
                    processed_content.append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_str}"
                            },
                        }
                    )
                else:
                    processed_content.append(part)
            current_history.append({"role": role, "content": processed_content})
        else:
            current_history.append({"role": role, "content": content})

    if not use_chat_history and current_history:
        current_history = [current_history[-1]]
    elif not use_chat_history:
        logger.warning("No user message found to send to the model.")
        return

    # Limit context window size (keep last N messages to manage memory)
    max_context_messages = 50  # Adjust based on your needs
    if len(current_history) > max_context_messages:
        current_history = current_history[-max_context_messages:]

    logger.info("Prompt to model:\n" + str(current_history))

    try:
        outputs = model.chat.completions.create(
            model=model_value,
            messages=current_history,
            temperature=ctx.settings.temperature.value,
            max_tokens=ctx.settings.max_completion_tokens.value,
            stream=True,
        )

        full_output = ""
        count = 0
        is_first_token = True
        start_time = time.time()

        for chunk in outputs:
            delta = chunk.choices[0].delta
            message_text = delta.content or ""
            if message_text:
                full_output += message_text
                count += 1

                if is_first_token:
                    logger.info(f"Time to first token: {time.time() - start_time:.2f}s")
                    is_first_token = False

                yield ctx.outputs.out1(
                    Frame(
                        ndframe=None,
                        rois=[],
                        color_space=ColorFormat.RGB,
                        frame_id=None,
                        headers=None,
                        other_data={
                            "message": message_text,
                            "token_number": count,
                            "done": False,
                        },
                    )
                )
                await asyncio.sleep(0.0025)

        if use_chat_history:
            # Add new API messages to chat history
            chat_history.extend(api_messages)
            # Add assistant response to chat history
            chat_history.append({"role": "assistant", "content": full_output.strip()})

        citation_string = ""
        if len(input_frame.other_data.get("citations", [])) > 0:
            citation_string = (
                "\n\nCitation(s):\n" if any(input_frame.other_data["citations"]) else ""
            )
            references = set()
            for citation in input_frame.other_data["citations"]:
                if citation:
                    document_name = citation["source"].split("/")[-1]
                    if "page" in citation:
                        page = citation["page"]
                        if f"{document_name}_{page}" in references:
                            continue
                        references.add(f"{document_name}_{page}")
                        citation_string += f"\n{document_name}, page {page}\n"
                    else:
                        if document_name in references:
                            continue
                        references.add(document_name)
                        citation_string += f"\n{document_name}, "

        yield ctx.outputs.out1(
            Frame(
                ndframe=None,
                rois=[],
                color_space=ColorFormat.RGB,
                frame_id=None,
                headers=None,
                other_data={
                    "message": citation_string,
                    "token_number": count + 1,
                    "done": True,
                    "usage": {
                        "completion_tokens": None,
                        "prompt_tokens": None,
                        "total_tokens": None,
                    },
                },
            )
        )

    except Exception as e:
        logger.exception(f"Exception during inference: {e}")
