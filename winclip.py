from anomalib import TaskType
from anomalib.data.image.folder import Folder, FolderDataset
from anomalib.models import WinClip
from anomalib.engine import Engine

dataset_root = 'datasets/datasets_opto/bottiglie_back/'

datamodule = Folder(
        name="bottiglie_back",
        root=dataset_root,
        normal_dir="test_normal",
        abnormal_dir="test_abnormal",
        task=TaskType.CLASSIFICATION,
        )

datamodule.setup()

model = WinClip()
model.setup("bottle")

engine = Engine(
        image_metrics=["AUROC","F1Score","PRECISION"]
        )

engine.test(datamodule=datamodule,model=model)
