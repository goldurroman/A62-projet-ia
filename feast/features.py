from datetime import timedelta
from feast import Entity, Field, FeatureView, FileSource, ValueType
from feast.types import Float32

lesion = Entity(
    name="lesion_id",
    value_type=ValueType.STRING,  # Toujours l'ancien ValueType pour Entity
    description="Identifiant unique de la lésion",
)

lesion_morphology_source = FileSource(
    name="lesion_morphology_source",
    path="s3://feast-offline/lesion_morphology.parquet",
    timestamp_field="event_timestamp",
)

lesion_morphology = FeatureView(
    name="lesion_morphology",
    entities=[lesion],  # MODIFICATION ICI : On passe directement l'objet 'lesion' au lieu de ["lesion_id"]
    ttl=timedelta(days=365),
    schema=[
        Field(name="compactness", dtype=Float32),
        Field(name="asymmetry", dtype=Float32),
        Field(name="diameter_px", dtype=Float32),
        Field(name="suspicion_score", dtype=Float32),
    ],
    source=lesion_morphology_source,
    tags={"domain": "skin_lesion", "type": "morphology"},
)