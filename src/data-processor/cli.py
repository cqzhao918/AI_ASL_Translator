import mediapipe as mp
import numpy as np
import pandas as pd
import json
import cv2
from google.cloud import storage
import os
import glob
import tqdm
import argparse


def transform_video(video_file):
    cap = cv2.VideoCapture(video_file)
    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.1)

    video_df = []
    frame_no=0

    while cap.isOpened():
        print('\r',frame_no,end='')
        success, image = cap.read()

        if not success: break
        image = cv2.resize(image, dsize=None, fx=4, fy=4)
        height,width,_ = image.shape

        #print(image.shape)
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = holistic.process(image)

        data = [] 
        fy = height/width

        if result.face_landmarks is None:
            for i in range(468): #
                data.append({
                    'type' : 'face',
                    'landmark_index' : i,
                    'x' : np.nan,
                    'y' : np.nan,
                    'z' : np.nan,
                })
        else:
            assert(len(result.face_landmarks.landmark)==468)
            for i in range(468): #
                xyz = result.face_landmarks.landmark[i]
                data.append({
                    'type' : 'face',
                    'landmark_index' : i,
                    'x' : xyz.x,
                    'y' : xyz.y *fy,
                    'z' : xyz.z,
                })

        if result.left_hand_landmarks is None:
            for i in range(21):  #
                data.append({
                    'type': 'left_hand',
                    'landmark_index': i,
                    'x': np.nan,
                    'y': np.nan,
                    'z': np.nan,
                })
        else:
            assert (len(result.left_hand_landmarks.landmark) == 21)
            for i in range(21):  #
                xyz = result.left_hand_landmarks.landmark[i]
                data.append({
                    'type': 'left_hand',
                    'landmark_index': i,
                    'x': xyz.x,
                    'y': xyz.y *fy,
                    'z': xyz.z,
                })

        #if result.pose_world_landmarks is None:
        if result.pose_landmarks is None:
            for i in range(33):  #
                data.append({
                    'type': 'pose',
                    'landmark_index': i,
                    'x': np.nan,
                    'y': np.nan,
                    'z': np.nan,
                })
        else:
            assert (len(result.pose_landmarks.landmark) == 33)
            for i in range(33):  #
                xyz = result.pose_landmarks.landmark[i]
                data.append({
                    'type': 'pose',
                    'landmark_index': i,
                    'x': xyz.x,
                    'y': xyz.y *fy,
                    'z': xyz.z,
                })

        if result.right_hand_landmarks is None:
            for i in range(21):  #
                data.append({
                    'type': 'right_hand',
                    'landmark_index': i,
                    'x': np.nan,
                    'y': np.nan,
                    'z': np.nan,
                })
        else:
            assert (len(result.right_hand_landmarks.landmark) == 21)
            for i in range(21):  #
                xyz = result.right_hand_landmarks.landmark[i]
                data.append({
                    'type': 'right_hand',
                    'landmark_index': i,
                    'x': xyz.x,
                    'y': xyz.y *fy,
                    'z': xyz.z,
                })
            zz=0

        frame_df = pd.DataFrame(data)
        frame_df.loc[:,'frame'] =  frame_no
        frame_df.loc[:, 'height'] = height/width
        frame_df.loc[:, 'width'] = width/width
        #print(frame_df)
        video_df.append(frame_df)
        frame_no +=1

    video_df = pd.concat(video_df)
    holistic.close()
    return video_df 

def clean_format(video_df):
   #print(video_df)
    video_df['row_id'] = video_df['frame'].astype('str')+'-'+video_df['type']+'-'+video_df['landmark_index'].astype('str')
    video_df.drop(['height', 'width'], axis=1, inplace=True)
    return video_df



def main(args):

    print("========Processing Data=========")

    client = storage.Client()
    bucket = client.bucket('capy-data')
    # videofile_dir = [file.name for file in bucket.list_blobs(prefix="data/WLASL-data/wlasl_videos/") if '.mp4' in file.name]
    # videofiles = [os.path.join('gs://capy-data',videofile) for videofile in videofile_dir if '.mp4' in videofile]
    # video_filenames = videofiles

    item = args.file

    if not os.path.exists('wlasl_deploy_video'):
        os.makedirs('wlasl_deploy_video')
    if not os.path.exists('wlasl_deploy_parquet'):
        os.makedirs('wlasl_deploy_parquet')

    bucket_train_filename = f'data/WLASL-data/wlasl_videos/{item}.mp4'
    blob = bucket.blob(bucket_train_filename)
    train_filename = f'./wlasl_deploy_video/{item}.mp4'
    blob.download_to_filename(train_filename)
    print('transform_video')
    train_video_df = transform_video(train_filename)
    print('clean_format')
    cleaned_train_vdf  = clean_format(train_video_df)  
    print('to_parquet')
    cleaned_train_vdf.to_parquet(f'./wlasl_deploy_parquet/{item}.parquet')
    
    gcs_new_prefix = 'data/WLASL-data/wlasl_parquet_deploy/'

    storage_client = storage.Client.from_service_account_json("../secrets/model-deployment.json")
    bucket = storage_client.get_bucket("capy-data")

    print('upload')
    for input_file in tqdm.tqdm(glob.glob(os.path.join("./wlasl_deploy_parquet", '*.parquet'))):
        print(input_file)
        gcs_object_name = os.path.join(gcs_new_prefix, os.path.basename(input_file))
        blob = bucket.blob(gcs_object_name)
        blob.upload_from_filename(input_file)


    # blob_df = bucket.blob("data/WLASL-data/wlasl_train_new.csv")
    # train_df = pd.read_csv(blob_df.open(), dtype='str')
    # train_videos_id = list(train_df['sequence_id'])

    # blob_df = bucket.blob("data/WLASL-data/wlasl_test_new.csv")
    # test_df = pd.read_csv(blob_df.open(), dtype='str')
    # test_videos_id = list(test_df['sequence_id'])

    # if not os.path.exists('wlasl_train_video'):
    #     os.makedirs('wlasl_train_video')
    # if not os.path.exists('wlasl_train_parquet'):
    #     os.makedirs('wlasl_train_parquet')
    # for item in train_videos_id:
    #     print(item)
    #     bucket_train_filename = f'data/WLASL-data/wlasl_videos/{item}.mp4'
    #     blob = bucket.blob(bucket_train_filename)
    #     train_filename = f'./wlasl_train_video/{item}.mp4'
    #     # blob.download_to_filename(train_filename)
    #     print(1)
    #     train_video_df = transform_video(train_filename)
    #     print(2)
    #     cleaned_train_vdf  = clean_format(train_video_df) 
    #     print(3)   
    #     cleaned_train_vdf.to_parquet(f'./wlasl_train_parquet/{item}_olivia.parquet')
    #     print("Success 1")
    #     break

    # if not os.path.exists('wlasl_test_video'):
    #     os.makedirs('wlasl_test_video')
    # if not os.path.exists('wlasl_test_parquet'):
    #     os.makedirs('wlasl_test_parquet')
    # for item in test_videos_id:
    #     print(item)
    #     bucket_test_filename = f'data/WLASL-data/wlasl_videos/{item}.mp4'
    #     blob = bucket.blob(bucket_test_filename)
    #     test_filename = f'./wlasl_test_video/{item}.mp4'
    #     blob.download_to_filename(test_filename)
    #     test_video_df = transform_video(test_filename)
    #     cleaned_test_vdf  = clean_format(test_video_df)    
    #     cleaned_test_vdf.to_parquet(f'./wlasl_test_parquet/{item}_olivia.parquet')
    #     print("Success 2")
    #     break

    # # gcs_bucket_name = 'capy-data'
    # gcs_train_prefix = 'data/WLASL-data/wlasl_parquet_train/'
    # gcs_test_prefix = 'data/WLASL-data/wlasl_parquet_test/'

    # storage_client = storage.Client.from_service_account_json("./psychic-bedrock-398320-e41cc1b33701.json")
    # bucket = storage_client.get_bucket("capy-data")

    # print("========Connect to bucket=========")

    # for input_file in tqdm.tqdm(glob.glob(os.path.join("./wlasl_train_parquet", '*_olivia.parquet'))):
    #     print(input_file)
    #     gcs_object_name = os.path.join(gcs_train_prefix, os.path.basename(input_file))
    #     blob = bucket.blob(gcs_object_name)
    #     blob.upload_from_filename(input_file)
    #     print("Success 3")
    #     break

    # for input_file in tqdm.tqdm(glob.glob(os.path.join("./wlasl_test_parquet", '*_olivia.parquet'))):
    #     print(input_file)
    #     gcs_object_name = os.path.join(gcs_test_prefix, os.path.basename(input_file))
    #     blob = bucket.blob(gcs_object_name)
    #     blob.upload_from_filename(input_file)
    #     print("Success 4")
    #     break

# def main_():
#     print("========Processing Data=========")
#     print("========Connect to bucket=========")
#     print("========Done=========")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data Collector CLI")

    parser.add_argument(
        "-f",
        "--file",
        default='70349', 
        help="Video file name",
    )

    args = parser.parse_args()

    main(args)