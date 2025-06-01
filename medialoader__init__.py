import time
from pathlib import Path
from typing import Any, AsyncIterator
from uuid import UUID

import cv2
import numpy as np
from webai_element_sdk.comms.messages import ColorFormat, Frame
from webai_element_sdk.element import Context, Element
from webai_element_sdk.element.settings import (
    BoolSetting,
    ElementSettings,
    NumberSetting,
    TextSetting,
)
from webai_element_sdk.element.variables import ElementOutputs, Output


class Settings(ElementSettings):
    video_file = TextSetting(
        name="video_file",
        display_name="Video File",
        description="The path to the video file to be loaded.",
        default="",
        hints=["file_path"],
        required=False,
    )
    image_directory = TextSetting(
        name="image_directory",
        display_name="Image Directory",
        description="The path to the image directory to be loaded.",
        default="",
        hints=["folder_path"],
        required=False,
    )
    frame_rate = NumberSetting[int](
        name="frame_rate",
        display_name="Frame Rate",
        description="The amount of frames per second (FPS) that should be processed.",
        default=0,
        hints=["advanced"],
    )
    stay_alive = BoolSetting(
        name="stay_alive",
        display_name="Stay Alive",
        description="Toggle to keep element running indefinitely after files complete.",
        default=False,
        hints=["advanced"],
    )


class Outputs(ElementOutputs):
    default = Output[Frame]()


element = Element(
    id=UUID("1916c9ba-fca7-4ed3-b773-11f400def123"),
    name="media_loader",
    display_name="Media Loader",
    version="0.3.9",
    description="Imports videos and images into the application so that AI models can use them for inference",
    outputs=Outputs(),
    settings=Settings(),
)


def _load_video_file(video: cv2.VideoCapture, frame_rate: int):
    # total_frames: int = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    counter: int = 0

    while video.isOpened():
        ret, frame = video.read()

        if not ret:
            print("End of file reached.")
            break

        counter += 1
        # print(f"Loading frame {counter}/{total_frames} ({frame_rate}fps)...")

        yield frame

    video.release()


def _load_images_from_directory(filepath: Path):
    for file in filepath.iterdir():
        if file.is_file() and file.suffix.lower() in [
            ".jpg",
            ".png",
            ".jpeg",
            ".npy",
            ".raw",
        ]:
            time.sleep(0.5)
            yield cv2.imread(str(file))


@element.executor  # type: ignore
async def run(ctx: Context[None, Outputs, Settings]) -> AsyncIterator[Any]:
    frame_rate: int = ctx.settings.frame_rate.value
    media_path: str

    if ctx.settings.video_file.value != "":
        media_path = ctx.settings.video_file.value
    elif ctx.settings.image_directory.value != "":
        media_path = ctx.settings.image_directory.value
    else:
        raise ValueError("No media path provided. Quitting...")

    media_path_obj = Path(media_path).resolve()

    if media_path_obj.is_dir():
        generator = _load_images_from_directory(media_path_obj)
    elif media_path_obj.suffix.lower() in [".mp4", ".avi", ".mov"]:
        video = cv2.VideoCapture(str(media_path_obj))

        if frame_rate == 0:
            frame_rate = int(video.get(cv2.CAP_PROP_FPS))

        generator = _load_video_file(video, frame_rate)
    else:
        raise ValueError(f"{media_path} is not a supported type or format")

    next_frame_time: float = time.perf_counter()

    for img in generator:
        if frame_rate != 0:
            time_to_next_frame = next_frame_time - time.perf_counter()

            if time_to_next_frame > 0:
                # print(f"Waiting {time_to_next_frame:.2f}s until next frame...")
                time.sleep(time_to_next_frame)

            next_frame_time += 1 / frame_rate

        if img is None:  # type: ignore
            continue

        image_rgb = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)

        # Removing "ColorFormat.BGR" parameter here (or using "ColorFormat.RGB")
        # breaks bounding box UI/output preview element with error:
        # "cv2.error: OpenCV(4.9.0) [...]/loadsave.cpp:803: error:
        # (-215:Assertion failed) buf.checkVector(1, CV_8U) > 0 in function
        # 'imdecode_'""
        yield ctx.outputs.default(Frame(ndframe=np.asarray(image_rgb), rois=[], color_space=ColorFormat.BGR))  # type: ignore

    while ctx.settings.stay_alive.value:
        continue
