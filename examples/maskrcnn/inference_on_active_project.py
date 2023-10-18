import json
import pickle
import uuid
from pathlib import Path

import cv2
import numpy as np
import torch
from encord_active.db.models import get_engine
from encord_active.lib.db.predictions import (
    BoundingBox,
    Format,
    ObjectDetection,
    Prediction,
)
from encord_active.public.active_project import ActiveContext
from tqdm import tqdm
from utils.model_libs import get_model_instance_segmentation
from utils.provider import get_config, get_transform

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
params = get_config("config.ini")

COCO_categories = json.loads(Path(params.data.train_ann).read_text())

predictions_to_store = []
file_paths = []
data_hashes = []


path = Path(params.inference.ea_database)
engine = get_engine(path)

model = get_model_instance_segmentation(len(COCO_categories["categories"]) + 1)
model.load_state_dict(torch.load(params.inference.model_checkpoint_path, map_location=device))
model.to(device)

img_transformer = get_transform(train=False)

active_context = ActiveContext(path.parent)
active_project = active_context.get_project(uuid.UUID(params.encord.project_hash))
all_active_projects = active_context.list_projects()
project_ontology = active_project.ontology.to_dict()

model.eval()
with torch.no_grad():
    for counter, du in enumerate(tqdm(active_project.iter_frames())):

        image = du.image
        img, _ = img_transformer(image, None)
        prediction = model([img[:3,:,:].to(device)])

        scores_filter = prediction[0]["scores"] > params.inference.confidence_threshold
        masks = prediction[0]["masks"][scores_filter].detach().cpu().numpy()
        labels = prediction[0]["labels"][scores_filter].detach().cpu().numpy()
        scores = prediction[0]["scores"][scores_filter].detach().cpu().numpy()

        for ma, la, sc in zip(masks, labels, scores):
            contours, _ = cv2.findContours((ma[0] > 0.5).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

            for contour in contours:
                if project_ontology["objects"][la.item() - 1]["shape"] == "bounding_box":
                    (x, y, w, h) = cv2.boundingRect(contour)
                    obj = ObjectDetection(
                        format=Format.BOUNDING_BOX,
                        data=BoundingBox(x=x / ma.shape[2], y=y / ma.shape[1], w=w / ma.shape[2], h=h / ma.shape[1]),
                        feature_hash=project_ontology["objects"][la.item() - 1]["featureNodeHash"],
                    )
                elif project_ontology["objects"][la.item() - 1]["shape"] == "polygon":
                    contour = contour.reshape(contour.shape[0], 2) / np.array([[ma.shape[2], ma.shape[1]]])
                    obj = ObjectDetection(
                        format=Format.POLYGON,
                        data=contour,
                        feature_hash=project_ontology["objects"][la.item() - 1]["featureNodeHash"],
                    )
                else:
                    continue

                prediction = Prediction(
                    data_hash=str(du.data_hash),
                    frame=du.frame,
                    confidence=sc.item(),
                    object=obj,
                )
                predictions_to_store.append(prediction)


with open(Path(params.encord.data_folder).parent / f"predictions_{params.inference.wandb_id}.pkl", "wb") as f:
    pickle.dump(predictions_to_store, f)
