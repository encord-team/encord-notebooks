from utils.download_from_encord import download_data_from_encord_project
from utils.provider import get_config

params = get_config("config.ini")
download_data_from_encord_project(
    params.encord.project_hash, params.encord.ssh_key, params.encord.data_folder
)
