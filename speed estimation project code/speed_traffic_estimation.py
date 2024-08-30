from ultralytics import YOLO
import pandas as pd
import numpy as np
import time
import regex as re
import os
import pickle
from joblib import dump, load
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
import cv2
import datetime

parser = argparse.ArgumentParser(description='Get arguments.')
parser.add_argument('--video_dir', type=str,
                    help='path to video')

parser.add_argument('--fps',default=25, type=int,
                    help='video framerate')

parser.add_argument('--out_dir',default='./', type=str,
                    help='output directory')

args = parser.parse_args()
src =  Path(args.video_dir)
fps = args.fps
out_dir = args.out_dir

speed_model_params = './weights/speed_reg_model.joblib'
trafficDensity_model_params = './weights/traffic_density_clustering_model.joblib'
speed_model = load(speed_model_params) 
traffic_density_model = load(trafficDensity_model_params)


def tracklets(src):
    model = YOLO("yolov8m.pt")

    results =model.track(source=src, conf=0.3, iou=0.5, show=False,verbose=False,stream=True,tracker="bytetrack.yaml",persist=True)
    columns = ['frame_id','ids','bbox_x','bbox_y','area','labels','ids','x1','y1','x2','y2']
    dataframe = pd.DataFrame(columns = columns)
    imgs = []

    for ind,r in enumerate(results):
        boxes = r.boxes.xywh.numpy()  # Boxes object for bbox outputs
        imgs.append(r.orig_img)
        boxes_xyxy = r.boxes.xyxy.numpy()
        x1 ,y1, x2, y2 = boxes_xyxy[:,0], boxes_xyxy[:,1], boxes_xyxy[:,2], boxes_xyxy[:,3]
        frame = [ind+1]*len(boxes)
        x = boxes[:,0]
        y = boxes[:,1]
        area = boxes[:,2]*boxes[:,3]
        label = r.boxes.cls.numpy()
        ids = r.boxes.id.numpy()
    
        mini_df = pd.DataFrame(list(zip(frame,ids, x, y,area,label, ids,x1,y1,x2,y2)),columns = columns)
        dataframe = pd.concat([dataframe,mini_df])
    instances = len(dataframe)
    dataframe['instances']=np.arange(instances)
    dataframe=dataframe[(dataframe['labels']==2) | (dataframe['labels']==5) | (dataframe['labels']==7)]
    dataframe['labels'] = dataframe['labels'].replace([2, 5,7], [0,2,3])
    
    first_df = dataframe.iloc[:,:7]
    second_df = dataframe.iloc[:,7:]
    
    first_df['instances']= second_df['instances'] 
    return first_df.values, second_df, np.array(imgs)


def speed_dist(full_tracklets, fps = 25):    
    u_ids = np.unique(full_tracklets[:,1])

    complete_frame = np.zeros((1,10))
    for ind,i in enumerate(u_ids):

        dist = [] #distance empty array
        p_s = [] #pixel distance array
        tracklet = full_tracklets[np.where(i==full_tracklets[:,1])] # get the full path of a tracklet/object
        if len(tracklet)==1:
            d = 0
            s = 0
            dist.append(d)
            p_s.append(s)

            tracklet=np.column_stack((tracklet,dist,p_s))

            if ind==0:
                complete_frame = complete_frame+tracklet
            else:
                complete_frame = np.append(complete_frame,tracklet, axis=0)
            continue
        x_y_cen = tracklet[:,2:4] #the x and y coordinates

        for p in range(1,len(x_y_cen)):
            j=p-1
            d = np.linalg.norm(x_y_cen[p] - x_y_cen[j])
            s = (d*fps)/2
            dist.append(d)
            p_s.append(s)


        dist.append(np.nan)
        p_s.append(np.nan)

        dist =np.array(dist)
        p_s = np.array(p_s)

        dist=pd.Series(dist)
        dist =dist.interpolate()
        dist = dist.values

        p_s=pd.Series(p_s)
        p_s =p_s.interpolate()
        p_s = p_s.values
        tracklet=np.column_stack((tracklet,dist,p_s))
        if ind==0:
            complete_frame = complete_frame+tracklet
        else:
            complete_frame = np.append(complete_frame,tracklet, axis=0)
        
    complete_frame = complete_frame[complete_frame[:, 7].argsort()]
    
    complete_frame = np.delete(complete_frame,[1,7],1)
    complete_frame[:,[5,6]] = complete_frame[:,[6,5]] 
    complete_frame[:,[6,7]] = complete_frame[:,[7,6]] 

    return complete_frame


def traffic_density_module(dataframe, model_cluster = traffic_density_model):
    a = dataframe.copy()
    road_lengths = []
    objects_ = []
   
  
    road_length_list = []

    unique_id = dataframe['id'].unique()
    for id_ in unique_id:
        df_id = dataframe[dataframe['id']==id_]
        first_occurence = df_id.iloc[[0]]
        last_occurence = df_id.iloc[[-1]]
        coor_last = last_occurence[['x_center','y_center']].values
        coor_first = first_occurence[['x_center','y_center']].values
        max_dist = np.linalg.norm(coor_last - coor_first)
        road_length_list.append(max_dist)
    road_length = max(road_length_list)

    unique_frame = dataframe['frame_num'].unique()
    for frame_ind,frame in enumerate(unique_frame):
        df_frame = dataframe[dataframe['frame_num']==frame]
        object_count_per_frame = df_frame.shape[0]
        objects_.append(object_count_per_frame)
        ind_max = np.where(df_frame['area']==np.max(df_frame['area']))[0][0]
       
        ind_min = np.where(df_frame['area']==np.min(df_frame['area']))[0][0]
       
        road_len2 = np.linalg.norm(df_frame.iloc[ind_max,1:3] - df_frame.iloc[ind_min,1:3])
        
    zipped_vehicle_count = dict(zip(unique_frame,objects_))
    a['road length'] = road_length 
    a['vehicle count'] = a['frame_num'].map(zipped_vehicle_count)
    a['traffic density'] = a['vehicle count']/a['road length']
    a['traffic density2'] = a['vehicle count']/road_len2
    #a = a.groupby('frame_num').mean()
    t_density= a['traffic density'].values
    t_density2= a['traffic density2'].values
    
    cluster = model_cluster.predict(t_density.reshape(-1,1))
    cluster2 = model_cluster.predict(t_density2.reshape(-1,1))
    
    cluster_mapping = ['Heavy','Medium','Medium','Light','Light','Light']
    cluster_num = [5,3,0,4,1,2]
    zipped_cluster = dict(zip(cluster_num,cluster_mapping))
    a['cluster_num'] = cluster
    a['cluster_num2'] = cluster2
    a['Traffic Density Cat'] =a['cluster_num'].map(zipped_cluster)
    a['Traffic Density Cat2'] =a['cluster_num2'].map(zipped_cluster)

    
    return a

def save_video(df, ims,out=out_dir, fps = fps, video_name = args.video_dir):
    ts = time.time()
    timestamp = datetime.datetime.fromtimestamp(ts).strftime('%Y_%m_%d_%H_%M_%S')

    #curr_time = datetime.datetime.now()

    video_name = os.path.join(out_dir, video_name.split('/')[-1][:-4]+'_'+timestamp+'_footage.mp4') #name outputvideo
    h, w = ims.shape[1],ims.shape[2]

    video = cv2.VideoWriter(video_name,cv2.VideoWriter_fourcc('m','p','4','v'), fps, (w,h))
    for i, img in enumerate(ims):
        
        df_new = df[df['frame_num']==i+1]
        for ind, row in df_new.iterrows():    
            x1 = row['x1']
            y1 = row['y1']
            x2 =row['x2']
            y2 = row['y2']
            ids = row['id']
            density = row['Traffic Density Cat']
            density2 = row['Traffic Density Cat2']
            start = (int(x1),int(y1))
            end = (int(x2), int(y2))
        

            speed = row['pred']   
            cv2.putText(
            img, #numpy array on which text is written
            'speed:' +"{0:.3f}".format(speed), #text
            (start[0],start[1]+33), #position at which writing has to start
            cv2.FONT_HERSHEY_SIMPLEX, #font family
            0.45, #font size
            (0, 255, 255), #font color
            1)

            cv2.putText(
            img, #numpy array on which text is written
            "Traffic: "+density, #text
            (20,20), #position at which writing has to start
            cv2.FONT_HERSHEY_SIMPLEX, #font family
            0.45, #font size
            (255, 125, 62), #font color
            2)


            cv2.putText(
            img, #numpy array on which text is written
            "Traffic2: "+density2, #text
            (20,75), #position at which writing has to start
            cv2.FONT_HERSHEY_SIMPLEX, #font family
            0.45, #font size
            (255, 125, 62), #font color
            2)
        
            
            cv2.putText(
            img, #numpy array on which text is written
            'IDS:' +"{0:.3f}".format(ids), #text
            (start[0],start[1]+56), #position at which writing has to start
            cv2.FONT_HERSHEY_SIMPLEX, #font family
            0.45, #font size
            (0, 255, 255), #font color
            1)
            
            cv2.rectangle(img, (start[0],start[1]), (end[0],start[1]+10), (55,125,255), -1)
        
            cv2.rectangle(img,start,end,(55,125,255),1)
            
           
        video.write(img)
    cv2.destroyAllWindows()
    video.release()
    return video_name

if __name__ == "__main__":
    start= time.time()
    df1,visual_df, imgs = tracklets(src)
    full_tracklets = df1
    ans=speed_dist(full_tracklets, fps)
    ans_for_prediction = ans[:,[0,3,4,5,6,7]]
    pred = speed_model.predict(ans_for_prediction)
    cols1 = ["frame_num","x_center","y_center","area","class","pixel_dist","pixel_speed","id","pred","x1","y1","x2","y2"]
    ans_check=np.column_stack((ans,pred,visual_df.iloc[:,0] ,visual_df.iloc[:,1],visual_df.iloc[:,2], visual_df.iloc[:,3] ))
    df=pd.DataFrame(ans_check,columns = cols1)
    df_ = traffic_density_module(df)
    out_file = save_video(df_, imgs)
    
    end= time.time()
    print(f"total time taken to render:  {end-start}s") 
    print(f'comleted, video saved as {out_file}')