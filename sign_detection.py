from vision.ssd.mobilenetv1_ssd import (
    create_mobilenetv1_ssd,
    create_mobilenetv1_ssd_predictor,
)


class SignDetection:
    def __init__(
        self,
        candidate_size: int = 200,
        model_path: str = "mb1-ssd-Epoch-99-Loss-7.836331605911255.pth",
        label_path: str = "labels.txt",
    ):
        class_names = [name.strip() for name in open(label_path).readlines()]
        net = create_mobilenetv1_ssd(len(class_names), is_test=True)
        net.load(model_path)
        self.predictor = create_mobilenetv1_ssd_predictor(net, candidate_size)

    def predict_func(self, image):
        boxes, labels, probs = self.predictor.predict(image, 10, 0.1)
        return boxes, labels, probs