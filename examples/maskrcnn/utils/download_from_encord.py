import os
import shutil
from pathlib import Path

import requests
from encord import EncordUserClient
from encord.constants.enums import DataType
from tqdm.contrib.concurrent import thread_map


def download_data_from_encord_project(project_hash: str, ssh_key: str, target_folder: str):
    def download_image_from_data_unit(data_unit: dict, target_path: Path):
        file_name = data_unit["data_hash"] + Path(data_unit["data_title"]).suffix
        image_target_path = target_path / file_name

        response = requests.get(data_unit["data_link"], stream=True)
        response.raise_for_status()

        with open(image_target_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)

    def download_video_frames_from_data_unit(data_unit: dict, target_path: Path):
        response = requests.get(data_unit["data_link"], stream=True)
        response.raise_for_status()

        video_target_folder = target_path / data_unit["data_hash"]
        video_target_folder.mkdir()
        video_file = video_target_folder / data_unit["data_title"]

        with open(video_file.as_posix(), "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)

        os.system(
            f"ffmpeg -i {video_file.as_posix()} -start_number 0 -loglevel panic {os.path.join(video_target_folder.as_posix(), '%d.jpg')}"
        )

        os.remove(video_file.as_posix())

    target_path = Path(target_folder)

    if target_path.is_dir():
        print("Project data is already downloaded.")
        return

    target_path.mkdir(parents=True, exist_ok=True)

    user_client = EncordUserClient.create_with_ssh_private_key(Path(ssh_key).read_text())
    project = user_client.get_project(project_hash)
    label_rows = project.list_label_rows_v2()

    all_image_data_units = []
    all_video_data_units = []
    for label in label_rows:
        fetched_label = project.get_label_row(label.label_hash)
        data_type = fetched_label["data_type"]

        if data_type in [DataType.IMAGE.value, DataType.IMG_GROUP.value]:
            for data_unit in fetched_label["data_units"].values():
                all_image_data_units.append(data_unit)
        elif data_type == DataType.VIDEO.value:
            for data_unit in fetched_label["data_units"].values():
                all_video_data_units.append(data_unit)

    print("Downloading images...")
    images_target_path = target_path / "images"
    images_target_path.mkdir()
    thread_map(
        lambda data_unit: download_image_from_data_unit(data_unit, images_target_path),
        all_image_data_units,
        max_workers=os.cpu_count(),
    )
    print("Downloading images finished!")

    print("Downloading videos")
    videos_target_path = target_path / "videos"
    videos_target_path.mkdir()
    thread_map(
        lambda data_unit: download_video_frames_from_data_unit(data_unit, videos_target_path),
        all_video_data_units,
        max_workers=os.cpu_count(),
    )

    print("Downloading videos finished!")
