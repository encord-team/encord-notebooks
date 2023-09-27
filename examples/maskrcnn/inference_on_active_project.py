import json
from pathlib import Path
import os
import pickle
import cv2
import torch
from encord_active.db.models import ProjectDataUnitMetadata, get_engine, Project
from encord_active.lib.db.predictions import BoundingBox, Format, ObjectDetection, Prediction
from PIL import Image
from sqlmodel import Session, select
from utils.model_libs import get_model_instance_segmentation
from utils.provider import get_config, get_transform
import numpy as np

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

with Session(engine) as sess:
    project = sess.exec(
        select(Project).where(Project.project_hash == params.encord.project_hash)
    ).first()
    project_ontology = project.ontology

model.eval()
with torch.no_grad():
    with Session(engine) as sess:
        all_du = sess.exec(
            select(ProjectDataUnitMetadata).where(ProjectDataUnitMetadata.project_hash == params.encord.project_hash)
        ).fetchall()
        for du in all_du:
            img = img_transformer(Image.open(du["data_uri"]).convert("RGB"))
            prediction = model([img.to(device)])

            scores_filter = prediction[0]["scores"] > params.inference.confidence_threshold
            masks = prediction[0]["masks"][scores_filter].detach().cpu().numpy()
            labels = prediction[0]["labels"][scores_filter].detach().cpu().numpy()
            scores = prediction[0]["scores"][scores_filter].detach().cpu().numpy()

            for ma, la, sc in zip(masks, labels, scores):
                contours, _ = cv2.findContours(
                    (ma[0] > 0.5).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
                )

                for contour in contours:
                    contour = contour.reshape(contour.shape[0], 2) / np.array([[ma.shape[2], ma.shape[1]]])
                    if project_ontology["objects"][la.item() - 1]['shape'] == 'bounding_box':
                        (x, y, w, h) = cv2.boundingRect(contour)
                        obj = ObjectDetection(
                            format=Format.BOUNDING_BOX,
                            data=BoundingBox(x=x, y=y, w=w, h=h),
                            feature_hash=project_ontology["objects"][la.item() - 1]["featureNodeHash"],
                        )
                    elif project_ontology["objects"][la.item() - 1]['shape'] == 'polygon':
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

with open(Path(params.inference.target_data_folder).parent / f"predictions_{params.inference.wandb_id}.pkl", "wb") as f:
    pickle.dump(predictions_to_store, f)
