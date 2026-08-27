from pathlib import Path


class ProjectConfig:
    ROOT = Path(__file__).resolve().parent.parent
    Repository_METABRIC=ROOT / "DataRepository" / "METABRIC"
    TRAIN_PATH = Repository_METABRIC / "train_brca_metabric.parquet"
    TEST_PATH = Repository_METABRIC / "test_brca_metabric.parquet"