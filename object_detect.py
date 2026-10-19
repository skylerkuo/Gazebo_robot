"""Standalone image demo using the same YOLOE detector as the ROS AI mode."""
import argparse
from pathlib import Path
import sys

try:
    from robot_perception.yoloe_detector import YOLOEDetector
except ModuleNotFoundError:
    # Keep the standalone workflow usable before building/sourcing ROS packages.
    sys.path.insert(0, str(Path(__file__).resolve().parent / 'src/robot_pick_place/robot_perception'))
    from robot_perception.yoloe_detector import YOLOEDetector


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='yoloe-26n-seg.pt')
    parser.add_argument('--source', default='gazebo_scene.png')
    parser.add_argument('--classes', nargs='+', default=['cube'])
    parser.add_argument('--device', default='0')
    parser.add_argument('--conf', type=float, default=0.05)
    parser.add_argument('--imgsz', type=int, default=960)
    parser.add_argument('--output', default='detected.png')
    parser.add_argument('--no-show', action='store_true')
    args = parser.parse_args()
    detector = YOLOEDetector(args.model, args.classes, args.device, args.conf, args.imgsz)
    result = detector.predict(args.source)
    if not args.no_show:
        result.show()
    result.save(filename=args.output)
    for box in result.boxes:
        label = result.names[int(box.cls.item())]
        confidence = float(box.conf.item())
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        print(f'\nObject: {label}\nConfidence: {confidence:.3f}')
        print(f'Center: ({(x1+x2)/2:.1f}, {(y1+y2)/2:.1f})')
        print(f'BBox: {[round(x1), round(y1), round(x2), round(y2)]}')


if __name__ == '__main__':
    main()
