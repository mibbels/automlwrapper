import sys
import json
import os
import shutil
import zipfile

import pandas as pd
import numpy as np

from pyspark.sql.session import SparkSession
from pywebhdfs.webhdfs import PyWebHdfsClient
from PIL import Image
from sedarapi import SedarAPI
from sklearn.model_selection import train_test_split



class SedarDataLoader:
    __slots__ = ['spark_session', 'sedarapi', 'hdfs']

    # --------------------------------------------------------------------------------------------#
    def __init__(self, sedar : SedarAPI):
        self.spark_session = None
        self.sedarapi = sedar

        self.hdfs = PyWebHdfsClient(host='192.168.220.107', port='9870')

    # --------------------------------------------------------------------------------------------#
    def create_spark_session(self):
        jars = ["io.delta:delta-core_2.12:1.0.0"]
 
        configs = {
            #"spark.master": "spark://192.168.220.107:7077",
            "spark.master": "local",
            "spark.app.name": "7777778-c47a-48d0-a3e4-9c91cdd92c3f1",
           # "spark.jars": tiledb_jar,
            "spark.submit.deployMode": "client",
            "spark.jars.packages": ",".join(jars),
            "spark.driver.bindAddress": "0.0.0.0",
            #######################
            "spark.cores.max": "2",
            "spark.executor.memory": "16G",
            "spark.driver.memory": "16G",
            "spark.driver.maxResultSize": "0",
            "spark.pyspark.driver.python": sys.executable,
            #########################
            "spark.executor.heartbeatInterval": "60s",
            "spark.network.timeoutInterval": "300s",
            "spark.network.timeout": "300s",
            "spark.sql.broadcastTimeout": "10000",
            #########################
            "fs.defaultFS": "hdfs://192.168.220.107:9000",
            "dfs.client.use.datanode.hostname": "false",
            "dfs.datanode.use.datanode.hostname": "false",
        }
        builder = SparkSession.builder
        for k, v in configs.items():
            builder = builder.config(k, v)
        spark_session = builder.getOrCreate()

        return spark_session

    # --------------------------------------------------------------------------------------------#
    def spark_df_from_json(self, json, is_local_path=False):
        if not self.spark_session:
            self.spark_session = self.create_spark_session()


        if is_local_path:
            df = self.spark_session.read.json(json)
        else:
            df = self.spark_session.createDataFrame(json)

        return df

    # --------------------------------------------------------------------------------------------#
    def query_data(self, sedar_workspace_id, sedar_dataset_id, query=None, file_save_location=None):
        QUERY_SPARK = True
        QUERY_RAW = False

        ws = self.sedarapi.get_workspace(sedar_workspace_id)
        ds = ws.get_dataset(sedar_dataset_id)

        if ds.content['schema']['type'] == 'UNSTRUCTURED':
            QUERY_SPARK = False
            QUERY_RAW = True
        elif ds.content['schema']['type'] == 'STRUCTURED':
            pass
        elif ds.content['schema']['type'] == 'SEMISTRUCTURED':
            pass
        elif ds.content['schema']['type'] == 'IMAGE':
            QUERY_SPARK = False
            QUERY_RAW = True
        else:
            raise Exception(f'Recieved unknown schema "{ds.content["schema"]["type"]}" type for dataset.')


        if QUERY_SPARK and query is None:
            query = f'SELECT * FROM {ds.id}'

        if QUERY_SPARK:
            query_result = ds.query_sourcedata(query)

            df = pd.DataFrame.from_dict(query_result['body'])

            print('INFO: the data has been returned as a pandas dataframe')
            return df
        
        elif QUERY_RAW:
            """ TODO: implement raw query in sedarapi 
            """
            if not file_save_location:
                raise Exception('file_save_location must be provided for raw query.')
            
            if not os.path.exists(file_save_location):
                os.makedirs(file_save_location)
            
            ws_id = ws.content['id']
            source_id = ds.content['datasource']['id']
            filenames = ds.content['datasource']['revisions'][-1]['source_files']
            for file in filenames:
                path = f"/datalake/{ws_id}/data/unstructured/{source_id}/{file[5:]}"
                data = self.hdfs.read_file(path)

                if not data:
                    raise Exception(f'Could not read file {path} from HDFS.')
                
                with open(os.path.join(file_save_location, file[5:]), 'wb') as f:
                    f.write(data)
            
            write_paths = "\n".join([os.path.join(file_save_location, file[5:]) for file in filenames])
            print(f'INFO: the data has been written to: {write_paths}')   

            return [os.path.join(file_save_location, file[5:]) for file in filenames]

    # --------------------------------------------------------------------------------------------#
    def zip_to_segmentation_df_gluon(self, zip_path, unzip_path):
        file_paths = self.extract_zip(zip_path, unzip_path)

        image_list = []
        mask_list = []

        for path in file_paths:
            full_path = os.path.join(unzip_path, path)
            #file_name, file_ext = os.path.splitext(os.path.basename(path))

            if 'images/' in path or 'image/' in path or 'data/' in path or 'img/' in path:
                image_list.append(full_path)
            elif 'mask/' in path or 'masks/' in path or 'label/' in path or 'labels/' in path:
                mask_list.append(full_path)
        
        image_list = sorted(image_list)
        mask_list = sorted(mask_list)

        df = pd.DataFrame(data={'image': image_list, 'mask': mask_list})
        return df

    # --------------------------------------------------------------------------------------------#
    def zip_to_coco_df_gluon(self, zip_path, unzip_path):
        file_paths = self.extract_zip(zip_path, unzip_path)

        label_file = [path for path in file_paths if path.endswith('.json')][0]
        image_location = [path.split('/')[:-1] for path in file_paths if path.endswith('.jpg') or path.endswith('.png')][0]
        image_location = '/'.join(image_location)

        annotation_json = None
        with open(os.path.join(unzip_path, label_file)) as f:
            annotation_json = json.load(f)
         
        image_list = []
        image_id_list = []
        for image_info in annotation_json['images']:
            image_id = image_info['id']
            file_name = image_info['file_name']

            image_full_path = os.path.join(unzip_path, file_name)
            if not os.path.exists(image_full_path):
                image_full_path = os.path.join(unzip_path, image_location, file_name.split('/')[-1])

            image_list.append(image_full_path)
            image_id_list.append(image_id)

        df = pd.DataFrame(data={'tmp_id': image_id_list, 'image': image_list})

        bbox_mapping = {}
        for annotation in annotation_json["annotations"]:
            image_id = annotation["image_id"]
            bbox = annotation["bbox"]
            if image_id not in bbox_mapping:
                bbox_mapping[image_id] = []
            bbox_mapping[image_id].append(bbox)

        df['label'] = df['tmp_id'].apply(lambda x: bbox_mapping.get(x, []))
        df = df.drop(columns=['tmp_id'])

        return df

    # --------------------------------------------------------------------------------------------#
    def zip_to_class_df_gluon(self, zip_path, unzip_path):
        file_paths = self.extract_zip(zip_path, unzip_path)

        class_mapping = {}
        data = []
    
        for path in file_paths:
            
            directory, file_name = os.path.split(path)
            
            if directory and file_name:  # FILE INSIDE FOLDER
                if directory not in class_mapping:
                    class_mapping[directory] = len(class_mapping)
                
                full_path = os.path.join(unzip_path, path)
                class_id = class_mapping[directory]
                data.append([full_path, class_id])
    
        df = pd.DataFrame(data, columns=['image', 'label'])
        return df, class_mapping

    
    # --------------------------------------------------------------------------------------------#
    def zip_to_class_np(self, zip_path, unzip_path):
        file_paths = self.extract_zip(zip_path, unzip_path)

        class_mapping = {}
        data = []
    
        for path in file_paths:
            
            directory, file_name = os.path.split(path)
            
            if directory and file_name:  # FILE INSIDE FOLDER
                if directory not in class_mapping:
                    class_mapping[directory] = len(class_mapping)
                
                full_path = os.path.join(unzip_path, path)
                class_id = class_mapping[directory]
                data.append([full_path, class_id])
    
        images = []
        labels = []
        for image_path, label in data:
            try:
                with Image.open(image_path) as img:
                    img_arr = np.array(img)
                    if img.mode != 'RGB':
                        img_arr = img_arr.reshape(img_arr.shape[0], img_arr.shape[1], 1) 
                    images.append(img_arr)
                    labels.append(label)
            except IOError:
                print(f"Could nwe design th eot read image: {image_path}")

        x = np.array(images)
        y = np.array(labels)

        return x, y, class_mapping
    
    # --------------------------------------------------------------------------------------------#
    def zip_to_segmentation_np(self, zip_path, unzip_path):
        file_paths = self.extract_zip(zip_path, unzip_path)

        image_paths = []
        mask_paths = []

        for path in file_paths:
            base_name = os.path.basename(path)
            name, ext = os.path.splitext(base_name)
            if ext not in ['.jpg', '.png']:
                continue

            if 'images/' in path or 'image/' in path or 'data/' in path or 'img/' in path:
                image_paths.append((name, os.path.join(unzip_path, path)))
            elif 'mask/' in path or 'masks/' in path or 'label/' in path or 'labels/' in path:
                mask_paths.append((name, os.path.join(unzip_path, path)))

        image_paths.sort(key=lambda x: x[0])
        mask_paths.sort(key=lambda x: x[0])

        X, Y = [], []
        for (img_name, img_path), (mask_name, mask_path) in zip(image_paths, mask_paths):
            if img_name == mask_name: 
                with Image.open(img_path) as img:
                    img_arr = np.array(img)
                    if img.mode != 'RGB':
                        img_arr = img_arr.reshape(img_arr.shape[0], img_arr.shape[1], 1) 
                    X.append(img_arr)

                with Image.open(mask_path) as mask:
                    mask_arr = np.array(mask)
                    if mask.mode != 'RGB':
                        mask_arr = mask_arr.reshape(mask_arr.shape[0], mask_arr.shape[1], 1)
                    Y.append(mask_arr)
            else:
                print(f"No matching file for {img_name} and {mask_name}")

        return np.array(X), np.array(Y)


    # --------------------------------------------------------------------------------------------#
    def split_train_test(self, data, label=None, train_size=0.8):
        if data is None:
            return None, None
        
        if isinstance(data, pd.DataFrame):
            return train_test_split(data, train_size=train_size)
        
        if isinstance(data, np.ndarray):
            if not label:
                raise Exception('label must be provided for numpy array.')
            return train_test_split(data, label, train_size=train_size)
        
    # --------------------------------------------------------------------------------------------#
    def clean_dir(dir_path):
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
        os.makedirs(dir_path)

    # --------------------------------------------------------------------------------------------#
    def extract_zip(self, zip_path, unzip_path):
        if not os.path.exists(unzip_path):
            os.makedirs(unzip_path)

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)
            file_names = zip_ref.namelist()
        
        return file_names
    

    # --------------------------------------------------------------------------------------------#
    def CAAFEFeatureEngineering(self, workspace_id, dataset_id, file_save_location, data_type, 
                                problem_type, target=None, dataset_description="", model="gpt-3.5-turbo",
                                iterations = 3):
        
        from caafe import CAAFEImageClassifier, CAAFEClassifier, CAAFEImageSegmentor
        from sklearn.metrics import accuracy_score, jaccard_score
        from sklearn.ensemble import RandomForestClassifier
        from tabpfn.scripts import tabular_metrics
        import openai

        if data_type == 'tabular' and problem_type == 'classification':
            df = self.query_data(workspace_id, dataset_id)
            df_train, df_test = self.split_train_test(df)


            base_clf = RandomForestClassifier()
            base_clf.fit(df_train.drop(columns=[target]), df_train[target])
            pred = base_clf.predict(df_test.drop(columns=[target]))
            acc = accuracy_score(pred, df_test[target])
            print(f'Accuracy before CAAFE {acc}')

            caafe_clf = CAAFEClassifier(
                base_classifier=base_clf,
                llm_model=model,
                iterations=iterations
                )

            if not target:
                raise Exception('target must be provided for tabular data.')
            
            caafe_clf.fit_pandas(
                df_train,
                target_column_name=target,
                dataset_description=dataset_description
                )

            pred = caafe_clf.predict(df_test)
            acc = accuracy_score(pred, df_test[target])
            print(f'Accuracy after CAAFE {acc}')

            return caafe_clf.apply_code(df)

        elif data_type == 'image' and problem_type == 'classification':
            loc = self.query_data(workspace_id, dataset_id, file_save_location=file_save_location)
            X, y, map = self.zip_to_class_np(loc[0], file_save_location+'_unzipped')

            caafe_clf = CAAFEImageClassifier(
                llm_model=model,
                iterations=iterations
                )

            pred = caafe_clf.performance_before_run(X, y)
            acc = accuracy_score(pred, y)
            print(f'Accuracy before CAAFE {acc}')    
            caafe_clf.fit_images(
                X,
                y,
                dataset_description
                )

            pred = caafe_clf.predict(X)
            acc = accuracy_score(pred, y)
            print(f'Accuracy after CAAFE {acc}')
            X_, y_ = caafe_clf.apply_code(X, y)

            return X_, y_, map
        
        elif data_type == 'image' and problem_type == 'segmentation':
            loc = self.query_data(workspace_id, dataset_id, file_save_location=file_save_location)
            X, y = self.zip_to_segmentation_np(loc[0], file_save_location+'_unzipped')

            caafe_clf = CAAFEImageSegmentor(
                llm_model=model,
                iterations=iterations
                )

            pred = caafe_clf.performance_before_run(X, y)
            iou = iou = jaccard_score(y.flatten(), pred.flatten(), average='macro')
            print(f'IoU before CAAFE {iou}') 


            caafe_clf.fit_images(
                X,
                y,
                dataset_description
                )

            pred = caafe_clf.predict(X)
            iou = iou = jaccard_score(y.flatten(), pred.flatten(), average='macro')
            print(f'IoU after CAAFE {iou}') 
            X_, y_ = caafe_clf.apply_code(X, y)
            return X_, y_,

        else:
            raise Exception('Not implemented yet.')
                                    


