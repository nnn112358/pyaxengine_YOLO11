import axengine as axe
import numpy as np
import cv2
import matplotlib.pyplot as plt
from dataclasses import dataclass
import time

# 設定パラメータ
#MODEL_PATH = '/opt/m5stack/data/yolo11n/yolo11n.axmodel'            # YOLO11 AXモデルのパス
#MODEL_PATH = 'model/yolo11n_320x320_base.axmodel'            
MODEL_PATH = 'model/yolo11n_640x640_base.axmodel'            
IMAGE_PATH = 'input.jpg'                 # 検出対象の画像パス
OUT_IMAGE_PATH = 'output_ax.jpg'                 # 検出対象の画像パス
CONFIDENCE_THRESHOLD = 0.45            # 確信度の閾値
NMS_THRESHOLD = 0.45                   # 非最大値抑制の閾値
#INPUT_SIZE = (320, 320)                # 入力サイズ
INPUT_SIZE = (640, 640)                # 入力サイズ
REG_MAX = 16                          # DFL（Distribution Focal Loss）の最大分布数

# COCOクラス名
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
    'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
    'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
    'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
    'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
    'teddy bear', 'hair drier', 'toothbrush'
]


@dataclass
class Object:
    bbox: list  # [x0, y0, width, height]のバウンディングボックス座標
    label: int  # クラスID
    prob: float  # 検出確率


def sigmoid(x):
    """シグモイド関数の実装"""
    return 1 / (1 + np.exp(-x))


def softmax(x, axis=-1):
    """
    数値的に安定なソフトマックス関数の実装
    入力値から最大値を引くことで、指数関数の演算時のオーバーフローを防ぐ
    """
    x = x - np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


def decode_distributions(feat, reg_max=16):
    """
    Distribution Focal Loss (DFL)の出力をデコードする関数

    引数:
    feat: [num_bboxes, 4, reg_max]形状の配列
    reg_max: 分布の最大値（デフォルト16）

    戻り値:
    [num_bboxes, 4]形状のデコードされた距離
    """
    # ソフトマックスを適用
    prob = softmax(feat, axis=-1)  # [num_bboxes, 4, reg_max]
    # 期待値を計算
    dis = np.sum(prob * np.arange(reg_max), axis=-1)  # [num_bboxes, 4]
    return dis


def preprocess(image_path, input_size):
    """
    入力画像を読み込み、前処理を行います
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"画像ファイルを読み取れませんでした: {image_path}")
    original_shape = image.shape[:2]  # (高さ, 幅)
    # 色空間変換 BGRからRGBへ
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # モデルの入力サイズにリサイズ
    resized_image = cv2.resize(image, input_size)
    # uint8型のまま保持し、正規化は行いません
    input_tensor = np.expand_dims(resized_image, axis=0).astype(np.uint8)  # [1, 320, 320, 3]

    """
    print("\nPreprocessed Image:")
    print(f"  Shape: {input_tensor.shape}")
    print(f"  Min: {np.min(input_tensor)}")
    print(f"  Max: {np.max(input_tensor)}")
    print(f"  Mean: {np.mean(input_tensor):.3f}")
    print("===========================\n")
    """

    return input_tensor, original_shape, image


def postprocess(outputs, original_shape, input_size, confidence_threshold, nms_threshold, reg_max=16):
    """
    モデル出力を処理し、検出結果に変換する

    引数:
    outputs: モデルからの出力
    original_shape: 元の画像サイズ
    input_size: モデルの入力サイズ
    confidence_threshold: 確信度の閾値
    nms_threshold: 非最大値抑制の閾値
    reg_max: DFLの最大分布数

    戻り値:
    最終的な検出結果のリスト
    """
    # 各特徴マップレベルの出力情報
    heads = [
        {'output': outputs[0], 'grid_size': input_size[0] // 8, 'stride': 8},
        {'output': outputs[1], 'grid_size': input_size[0] // 16, 'stride': 16},
        {'output': outputs[2], 'grid_size': input_size[0] // 32, 'stride': 32}
    ]

    detections = []
    num_classes = 80  # COCOデータセットのクラス数
    bbox_channels = 4 * reg_max  # バウンディングボックス関連の出力チャネル数
    class_channels = num_classes  # クラススコアのチャネル数

    for head in heads:
        output = head['output']  # [1, grid_h, grid_w, 144]
        batch_size, channels, height, width = output.shape
        stride = head['stride']
        """
        print(f"ストライド{stride}、出力形状{output.shape}の特徴マップを処理中")
        """

        if batch_size != 1:
            raise ValueError("現在はバッチサイズ1のみサポートしています")
        batch_size, grid_h, grid_w, channels = output.shape
        # ONNXモデルの出力形状を調整
        # 出力を[batch_size, height, width, bbox_channels + class_channels]に変換
        #output = np.transpose(output, (0, 2, 3, 1))
        #grid_h = height
        #grid_w = width

        """
        print(f"  Shape: {output.shape}")
        print(f"  height: {height}")
        print(f"  width: {width}")

        print(f"  Shape: {output.shape}")
        print(f"  grid_h: {grid_h}")
        print(f"  grid_w: {grid_w}")
        """

        # バウンディングボックスとクラス部分を分割
        bbox_part = output[:, :, :, :bbox_channels]  # [1, 80, 80, 64]
        class_part = output[:, :, :, bbox_channels:]  # [1, 80, 80, 80]

        """
        print(f" bbox_channels: {bbox_channels}")
        print(f" bbox_part Shape: {bbox_part.shape}")
        print(f" class_part Shape: {class_part.shape}")
        """

        # bbox_partの処理
        num_bbox_params = 4  # 左、上、右、下の4パラメータ
        if bbox_channels != num_bbox_params * reg_max:
            raise ValueError(
                f"bbox_channels ({bbox_channels}) が4*reg_max ({4 * reg_max})と一致しません")

        # bbox_partを[1, grid_h, grid_w, 4, 16]にリシェイプ
        try:
            bbox_part = bbox_part.reshape(
                batch_size, grid_h, grid_w, num_bbox_params, reg_max)
        except ValueError as e:
            print(f"bbox_partのリシェイプに失敗: {e}")
            raise

        # [grid_h * grid_w, 4, 16]にリシェイプ
        bbox_part = bbox_part.reshape(
            grid_h * grid_w, num_bbox_params, reg_max)  # [1600,4,16]

        # class_partの処理
        # [1, grid_h * grid_w, 80]にリシェイプ
        class_part = class_part.reshape(
            batch_size, grid_h * grid_w, class_channels)  # [1,1600,80]

        # グリッドセルごとに処理
        for b in range(batch_size):
            for i in range(grid_h * grid_w):
                h = i // grid_w
                w = i % grid_w
                # クラススコアの取得
                class_scores = class_part[b, i, :]  # [80]
                class_id = np.argmax(class_scores)
                class_score = class_scores[class_id]
                box_prob = sigmoid(class_score)

                if box_prob < confidence_threshold:
                    continue

                # バウンディングボックスパラメータの取得
                bbox = bbox_part[i, :, :]  # [4, 16]

                # バウンディングボックスパラメータのデコード
                dis_left = decode_distributions(bbox[0, :], reg_max)
                dis_top = decode_distributions(bbox[1, :], reg_max)
                dis_right = decode_distributions(bbox[2, :], reg_max)
                dis_bottom = decode_distributions(bbox[3, :], reg_max)

                # 中心座標の計算
                pb_cx = (w + 0.5) * stride
                pb_cy = (h + 0.5) * stride

                # バウンディングボックス座標の計算
                x0 = pb_cx - dis_left * stride
                y0 = pb_cy - dis_top * stride
                x1 = pb_cx + dis_right * stride
                y1 = pb_cy + dis_bottom * stride

                # 座標を元の画像サイズにスケーリング
                scale_x = original_shape[1] / input_size[0]
                scale_y = original_shape[0] / input_size[1]
                x0 = np.clip(x0 * scale_x, 0, original_shape[1] - 1)
                y0 = np.clip(y0 * scale_y, 0, original_shape[0] - 1)
                x1 = np.clip(x1 * scale_x, 0, original_shape[1] - 1)
                y1 = np.clip(y1 * scale_y, 0, original_shape[0] - 1)

                # 幅と高さの計算
                width = x1 - x0
                height = y1 - y0

                # 検出結果に追加
                detections.append(Object(
                    bbox=[float(x0), float(y0), float(width), float(height)],
                    label=int(class_id),
                    prob=float(box_prob)
                ))

    # 非最大値抑制（NMS）の適用
    if len(detections) == 0:
        return []
    """
    print(f"detections Length: {len(detections)}")
    #print(f"First detection Shape: {detections[0].shape}")
    """

    boxes = np.array([d.bbox for d in detections])  # [num_detections, 4]
    scores = np.array([d.prob for d in detections])  # [num_detections]
    class_ids = np.array([d.label for d in detections])  # [num_detections]

    final_detections = []

    # クラスごとにNMSを適用
    unique_classes = np.unique(class_ids)
    for cls in unique_classes:
        idxs = np.where(class_ids == cls)[0]
        cls_boxes = boxes[idxs]
        cls_scores = scores[idxs]

        # NMSの計算
        x1_cls = cls_boxes[:, 0]
        y1_cls = cls_boxes[:, 1]
        x2_cls = cls_boxes[:, 0] + cls_boxes[:, 2]
        y2_cls = cls_boxes[:, 1] + cls_boxes[:, 3]

        areas = (x2_cls - x1_cls) * (y2_cls - y1_cls)
        order = cls_scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            if order.size == 1:
                break

            # IoU（Intersection over Union）の計算
            xx1 = np.maximum(x1_cls[i], x1_cls[order[1:]])
            yy1 = np.maximum(y1_cls[i], y1_cls[order[1:]])
            xx2 = np.minimum(x2_cls[i], x2_cls[order[1:]])
            yy2 = np.minimum(y2_cls[i], y2_cls[order[1:]])

            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            intersection = w * h
            iou = intersection / (areas[i] + areas[order[1:]] - intersection)

            # IoUが閾値以下の検出結果を保持
            inds = np.where(iou <= nms_threshold)[0]
            order = order[inds + 1]

        # 最終的な検出結果に追加
        for idx in keep:
            final_detections.append(Object(
                bbox=cls_boxes[idx].tolist(),
                label=int(cls),
                prob=float(cls_scores[idx])
            ))

    return final_detections


def main():

    timing = {}

    # 前処理の実行
    try:
        # 前処理の時間計測
        preprocess_start = time.time()
        input_tensor, original_shape, original_image = preprocess(
            IMAGE_PATH, INPUT_SIZE)
        timing['preprocess'] = time.time() - preprocess_start

    except FileNotFoundError as e:
        print(e)
        return

    # AXモデルの読み込み
    try:
        session = axe.InferenceSession(MODEL_PATH)
    except Exception as e:
        print(f"モデルの読み込み中にエラーが発生しました: {e}")
        return

    # モデルの入力名を取得
    input_name = session.get_inputs()[0].name

    # モデルの出力名を取得
    output_names = [output.name for output in session.get_outputs()]

    # モデルの入力サイズを取得
    #input_shape = session.get_inputs()[0].shape
    #INPUT_SIZE = (input_shape[2], input_shape[1])  # NHWC形式なのでW,Hの順


    # セッション情報の出力
    print("\n=== ONNX Session Information ===")

    # 入力情報の表示
    print("\nInput Information:")
    for i, input_detail in enumerate(session.get_inputs()):
        print(f"\nInput {i}:")
        print(f"  Name: {input_detail.name}")
        print(f"  Shape: {input_detail.shape}")
        print(f"  Type: {input_detail.dtype}")

    # 出力情報の表示
    print("\nOutput Information:")
    for i, output_detail in enumerate(session.get_outputs()):
        print(f"\nOutput {i}:")
        print(f"  Name: {output_detail.name}")
        print(f"  Shape: {output_detail.shape}")
        print(f"  Type: {output_detail.dtype}")

    # 推論の実行
    # バックボーンモデルの実行時間計測
    outputs_start = time.time()
    try:
        outputs = session.run(output_names, {input_name: input_tensor})
    except Exception as e:
        print(f"推論実行中にエラーが発生しました: {e}")
        return
    timing['outputs'] = time.time() - outputs_start

    # 後処理の実行
    try:
        postprocess_start = time.time()

        detections = postprocess(
            outputs,
            original_shape,
            INPUT_SIZE,
            CONFIDENCE_THRESHOLD,
            NMS_THRESHOLD,
            reg_max=REG_MAX
        )
        timing['postprocess'] = time.time() - postprocess_start

    except Exception as e:
        print(f"後処理中にエラーが発生しました: {e}")
        return

    # 検出結果の可視化
    for det in detections:
        bbox = det.bbox
        score = det.prob
        class_id = det.label
        # クラスIDがCOCO_CLASSESの範囲を超える場合の処理
        if class_id >= len(COCO_CLASSES):
            label = f"cls{class_id}: {score:.2f}"
        else:
            label = f"{COCO_CLASSES[class_id]}: {score:.2f}"
        # バウンディングボックスと検出ラベルの描画
        print(f"Detected: {label} at bbox {bbox}")
        x, y, w, h = map(int, bbox)
        cv2.rectangle(original_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(original_image, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 合計時間を計算
    timing['total'] = sum(timing.values())
    print("\n処理時間の内訳:")
    print(f"前処理時間: {timing['preprocess']:.3f} 秒")
    print(f"outputs推論時間: {timing['outputs']:.3f} 秒")
    print(f"後処理時間: {timing['postprocess']:.3f} 秒")
    print(f"合計時間: {timing['total']:.3f} 秒")

    cv2.imwrite(OUT_IMAGE_PATH, cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR))
#    cv2.imshow('Detection Result', cv2.cvtColor(cv2.resize(original_image, (640, 480)), cv2.COLOR_RGB2BGR))
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
