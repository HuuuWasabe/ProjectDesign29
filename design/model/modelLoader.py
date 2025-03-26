from pytorchvideo.models.x3d import create_x3d
import torch, torchmetrics, pytorch_lightning
from sklearn.metrics import classification_report
from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    ShortSideScale,
    UniformTemporalSubsample,
)

from torchvision.transforms import (
    Compose,
    Lambda,
    CenterCrop,
)

# Model Configurations
clip_duration = 5
sub_clip_duration = 30
batch_size = 6
crop_size = 224
categories = sorted(["BarbellCurl", "Deadlift", "LateralRaises", "OverheadPress", "Squat"])

def make_kinetics_x3d():
    return create_x3d(
        input_channel=3,
        input_crop_size=crop_size,
        model_num_class=len(categories),
        dropout_rate=0.5,
        width_factor=2.4,
        depth_factor=2.2,
        norm=torch.nn.BatchNorm3d,
        activation=torch.nn.ReLU
    )


class VideoClassificationLightningModule(pytorch_lightning.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = make_kinetics_x3d()
        self.mca = torchmetrics.Accuracy(task='multiclass', num_classes=len(categories))
        
        self.test_preds = []
        self.test_targets = []

    def forward(self, x):
        return self.model(x)
    
    def on_train_epoch_start(self):
        epoch = self.trainer.current_epoch

    def training_step(self, batch, batch_idx):
        y_hat = self.model(batch["video"])
        loss = torch.nn.functional.cross_entropy(y_hat, batch["label"])
        acc = self.mca(torch.nn.functional.softmax(y_hat, dim=-1), batch["label"])
        self.log("train_loss", loss.item())
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        y_hat = self.model(batch["video"])
        loss = torch.nn.functional.cross_entropy(y_hat, batch["label"])
        acc = self.mca(torch.nn.functional.softmax(y_hat, dim=-1), batch["label"])
        self.log("val_loss", loss)
        self.log("val_acc", acc, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        y_hat = self.model(batch["video"])
        loss = torch.nn.functional.cross_entropy(y_hat, batch["label"])
        acc = self.mca(torch.nn.functional.softmax(y_hat, dim=-1), batch["label"])
        
        preds = torch.argmax(y_hat, dim=1)
        self.test_preds.append(preds.detach())
        self.test_targets.append(batch["label"].detach())
        self.log("test_loss", loss)
        self.log(
            "test_acc", acc, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True
        )
        return loss

    def on_test_epoch_end(self):
        all_preds = torch.cat(self.test_preds).cpu().numpy()
        all_targets = torch.cat(self.test_targets).cpu().numpy()

        report = classification_report(all_targets, all_preds, target_names=categories)
        print("Classification Report:\n", report) # Add a classification Report

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(),
                                      lr=1e-3,
                                      #weight_decay=1e-4
                                    )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=60, eta_min=1e-6, last_epoch=-1)
        return [optimizer], [scheduler]


class PredictionModule:
    def __init__(self):
        self.model_path = "/home/joker/Documents/devProjects/AppDev/Sample API/model/model.ckpt"
        self.device = 'cpu'
        self.video_transforms = Compose(
            [
            ApplyTransformToKey(
              key="video",
              transform=Compose(
                  [
                    UniformTemporalSubsample(sub_clip_duration),
                    Lambda(lambda x: x / 255.0),
                    Normalize((0.45, 0.45, 0.45), (0.225, 0.225, 0.225)),
                    ShortSideScale(248),
                    CenterCrop(crop_size)
                  ]
                ),
              ),
            ]
        )

    def make_model(self):
        model = VideoClassificationLightningModule.load_from_checkpoint(self.model_path)
        model = model.eval()
        model = model.to(self.device)

        return model

    def get_labels(self):
        id_to_classnames = {}
        for i, k in enumerate(categories):
            id_to_classnames[i] = str(k)

        return id_to_classnames
    
    def predict(self, video_path, start_sec, end_sec):
        # Initial Steps
        model = self.make_model()
        id_to_classnames = self.get_labels()

        #start_sec = 0
        #end_sec = start_sec + (clip_duration*sub_clip_duration)/30
        #end_sec = 2

        test_video = EncodedVideo.from_path(video_path)
        video_data = test_video.get_clip(start_sec=start_sec, end_sec=end_sec)

        video_data = self.video_transforms(video_data)
        inputs = video_data['video']
        inputs = inputs.to(self.device)

        preds = model(inputs[None, ...])

        num = 3
        post_act = torch.nn.Softmax(dim=1)
        preds = post_act(preds)
        pred_classes = preds.topk(k=num).indices[0]

        predicted_class = [id_to_classnames[int(i)] for i in pred_classes]
        #print(f"Top {num} predicted labels: %s" % ", ".join(predicted_class))
        return predicted_class[0]