import csv
import cv2
import os
import requests
import shutil
import json
import math
from tqdm import tqdm


def read_csv_file(file_path):
    with open(file_path, mode="r", encoding="utf-8") as file:
        return [row for row in csv.DictReader(file)]


def extract_frames(video_path, output_folder, max_frames=200):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 获取视频的总帧数

    # 如果视频的总帧数少于 max_frames，就设置为视频的总帧数
    frames_to_extract = min(total_frames, max_frames)

    if total_frames <= frames_to_extract:
        frame_interval = 1  # 如果总帧数小于等于 max_frames，提取所有帧
    else:
        # 否则根据总帧数和 max_frames 计算合适的帧间隔
        frame_interval = math.ceil(total_frames / frames_to_extract)

    if not cap.isOpened():
        print("Extract frames of video error: Could not open video.")
        return

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)

        # 只在满足提取条件时提取帧
        if current_frame % frame_interval == 0 and frame_count < frames_to_extract:
            frame_path = os.path.join(output_folder, f"frame{frame_count:06d}.jpg")
            cv2.imwrite(frame_path, frame)
            frame_count += 1

        # 如果已提取足够的帧，退出循环
        if frame_count >= frames_to_extract:
            break

    cap.release()
    return frame_count - 1



def download_video(url, save_path):
    response = requests.get(url, stream=True)
    response.raise_for_status()

    with open(save_path, "wb") as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)


def remake_dir(dir_path):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)


def save_json(obj, json_path):
    with open(json_path, "w", encoding="utf-8") as json_file:
        json.dump(obj, json_file, ensure_ascii=False, indent=4)

    print(f"Saved {os.path.join(os.getcwd(), json_path)}")


def read_webvid_10m(webvid_10m_path, split, limit=None):
    if split == "train":
        webvid_10m_path = os.path.join(webvid_10m_path, "data", "train")
    elif split == "val":
        webvid_10m_path = os.path.join(webvid_10m_path, "data", "val")
    elif split == "test":
        webvid_10m_path = os.path.join(webvid_10m_path, "data", "test")
    else:
        raise NotImplementedError
    webvid_10m_path = os.path.join(webvid_10m_path, "partitions")
    partitions_num = len(os.listdir(webvid_10m_path))
    limit = min(limit, partitions_num) if limit is not None else partitions_num
    items = []
    for i in range(0,limit):
        items.append(
            os.path.join(webvid_10m_path, f"data_{split}_partitions_{i:04d}.csv")
        )
    return items


def main(webvid_10m_path, split, max_item_count):
    print(f"Max item count: {max_item_count}")
    base_path = os.path.join("webvid_10m-convert-output", split)
    remake_dir(base_path)
    videos_temp_path = os.path.join(base_path, "videos_temp")
    base_frames_path = os.path.join(base_path, "frames")
    os.makedirs(videos_temp_path)
    os.makedirs(base_frames_path)

    partition_paths = read_webvid_10m(webvid_10m_path, split)

    gt_items = {}
    annotations = []
    images =[]
    items = {}
    item_count = 0
    remain_item_count = max_item_count
    for partition_path in partition_paths:
        rows = read_csv_file(partition_path)
        for row in tqdm(rows[:remain_item_count], desc=f"Processing {partition_path}"):
            video_id = row["videoid"]
            contentUrl = row["contentUrl"]
            name = row["name"].strip()
            video_filename = video_id + ".mp4"
            video_path = os.path.join(videos_temp_path, video_filename)
            try:
                download_video(contentUrl, video_path)
            except Exception as e:
                print(f"Error downloading video '{name}': {e}")
            else:
                video_frames_path = os.path.join(base_frames_path, video_id)
                frame_length = extract_frames(video_path, video_frames_path)-1
                items[video_id] = {
                    "image_id": video_filename,
                    "caption": name,
                    "id": int(video_id),
                    "video": video_id,
                    "frame_length": frame_length,
                    "num_frames": frame_length,
                }
                annotations.append(
                  {
                    "image_id": video_id,
                    "caption": name,
                    "id": int(video_id)
                  }
                )
                images.append({
                  "id":video_id
                })
                item_count += 1
        remain_item_count -= len(rows)
        if remain_item_count <= 0:
            break
    gt_items["annotations"] = annotations
    gt_items["images"] = images
    save_json(items, os.path.join(base_path, f"cap_{split}.json"))
    save_json(gt_items, os.path.join(base_path, f"cap_{split}_gt.json"))


if __name__ == "__main__":
    val_max_item_count = 500
    train_max_item_count = 0
    print("Start processing train:")
    main(
        webvid_10m_path="webvid_csv", split="train", max_item_count=train_max_item_count
    )
    print("Start processing val:")
    main(webvid_10m_path="webvid_csv", split="val", max_item_count=val_max_item_count)
