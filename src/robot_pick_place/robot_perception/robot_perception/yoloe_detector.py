"""Shared YOLOE inference used by object_detect.py and the optional ROS adapter.

No torch/ultralytics imports until AI inference is requested.
"""


def best_per_class(result):
    """Keep the highest-confidence instance of each class, including its mask."""
    best = {}
    for index, box in enumerate(result.boxes):
        label = int(box.cls.item())
        confidence = float(box.conf.item())
        if label not in best or confidence > best[label][0]:
            best[label] = (confidence, index)
    return result[[value[1] for value in best.values()]]


class YOLOEDetector:
    def __init__(self, model, classes, device='cpu', confidence=0.05, image_size=960):
        if not classes or any(not text.strip() for text in classes):
            raise ValueError('At least one nonempty AI class is required')
        if not 0.0 < confidence <= 1.0 or image_size <= 0:
            raise ValueError('Invalid AI confidence or image size')
        try:
            from ultralytics import YOLOE
        except ImportError as error:
            raise RuntimeError('AI dependencies unavailable. Activate the documented YOLOE virtual environment.') from error
        self.model = YOLOE(model)
        self.classes = None
        self.set_classes(classes)
        self.device = device
        self.confidence = confidence
        self.image_size = image_size

    def set_classes(self, classes):
        classes = [text.strip() for text in classes]
        if not classes or any(not text for text in classes):
            raise ValueError('At least one nonempty AI class is required')
        if classes != self.classes:
            self.model.set_classes(classes)
            self.classes = list(classes)

    def predict(self, source):
        results = self.model.predict(source=source, imgsz=self.image_size, conf=self.confidence,
                                     device=self.device, retina_masks=True, verbose=False)
        if not results:
            raise RuntimeError('YOLOE returned no image results')
        return best_per_class(results[0])
