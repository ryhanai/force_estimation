# Force Estimation

## Dockerを利用する場合（おすすめ）

### 準備
- Docker imageとsampleログファイルをここからダウンロードします．
  ```sh
  $ git clone -b airec https://github.com/ryhanai/force_estimation.git
  ```
  - [Docker image (5.9GB)](https://aist.box.com/s/son0751m5kkhl7utroyno36iz06lxjsz)（産総研内のみ（Box））
  - [sampleログファイル (208MB)](https://drive.google.com/file/d/1b1lcsoz_MxtpR1gUzYOYI5AWGufV3sU9/view?usp=sharing)

### 実行（以下のパスはforce_estimationからの相対パス）
- Docker imageの実行とviewerの起動
  ```sh
  $ docker/run_force_estimation.sh
  $ roslaunch force_estimation viewer_AIREC.launch
  ```
- 認識プログラムの実行
  ```sh
  $ docker/shell.sh
  $ rosrun force_estimation demo_force_estimation.py
  ```
- sampleログファイルの実行（ホストで実行します）
  ```sh
  $ rosbag play rosbag-airec-sr300-rgbd_pointcloud_tf_2024-10-25-20-17-55.bag -l
  ```

### 認識プログラム実行中の設定変更
- 設定プログラムを起動
  ```sh
  $ docker/shell.sh
  $ rqt
  ```
![rqt1](https://github.com/user-attachments/assets/4446416e-371a-4f99-9657-0a3c9ccb2071)
- Plugins -> Configuration -> Dynamic Reconfigure -> force_distribution_publisherを選択します．
- force_vis_threshold: 推定した力分布を可視化するときの閾値です．大きくすると大きな力のみを可視化します．多くの場合0.45くらいに設定します．
- calc_lifting_direction: チェックが入っているときには，lifting directionの計算を行います．
- object_position: lifting対象物の指定方法を選択します．
  - “Object_recognition(1)”: 対象物の位置・姿勢をtopicで送ります．
    - topicによるlifting対象物（位置）を指定する例：
    ```sh
    $ rostopic pub -1 /foundationpose/position geometry_msgs/Vector3 "{x: 0.01, y: 0.1, z: 0.743}"
    ```
  - “Interactive_marker(0)”: interactive markerで指定されている位置を対象物の位置として推定を行います．**このときはtopicによる指定を受け付けません．**

### AIRECシミュレータ（Gazebo）に対する実行
- 上記の「ログファイルの再生」の替わりにAIRECのシミュレーション環境を起動することで，シミュレータ（Gazebo）から取得した画像に対して力分布の予測及びlifting directionの計算を行うことができます．
- viewer（RVis）において，pointcloudのtopicがログとシミュレータで異なるので修正が必要な場合があります（シミュレータでは点群が/torobo/head/sr300/camera/depth/points）


## 環境設定（変更する場合）

### ROSの設定（docker/config）
- ROS_MASTER_URIとROS_HOSTNAMEを環境にあわせて設定します．
```sh
ROS_MASTER_URI=http://192.168.10.109:11311
ROS_HOSTNAME=192.168.10.109
```

### 学習済みモデルの指定（docker imageにはデフォルトで含まれ，設定されています）
- 学習済みモデルの[download](https://drive.google.com/file/d/1-e002NMWVaFazrGgvLzdr5vN4YDN_w5V/view?usp=sharing)
- 推論に使うモデルの指定（configs/hydra_config.yaml）
  ```yaml
  checkpoint_directory: "20250126_1251_33"
  weight_file: '01000.pth'
  ```

### 座標系の設定（launch/viewer_AIREC.launch）
- 下記のようにstatic_transformを設定します．
  ```xml
  <node pkg=“tf” type=“static_transform_publisher” name=“fmap_frame_broadcaster” args=“0.605 0.0 0.1 0 0 0 world fmap_frame 100” />
  ```
- 説明
  - 予測したforcemap及びlifting directionはworld座標系であるので，学習したsimulation環境と実行時の環境の位置合わせが必要です．
  - forcemapはfmap_frame相対で結果を出力するので，/world -> /fmap_frameのtransformとして設定します．
  - ただし，table上にはxyを合わせる目印がないので，camera姿勢を大まかに揃える方が容易です．
  - 参考データ
    - 学習時のcamera座標([-0.42, 0, 1.15], [0, 45, 0]) [m, degree]
    - AIREC([0.185, 0.005, 1.38], [0, 45, 0]) [m, degree] （Gazeboの場合）

### Topicの設定（configs/hydra_config.yaml）
- 入力画像，頭部realsense
  ```yaml
  image_topic: /torobo/head/sr300/camera/color/image_raw
  ```
- lifting directionを計算する対象物の位置・姿勢
  ```yaml
  object_position_topic: /foundationpose/position
  ```
- lifting directionの出力先
  ```yaml
  lifting_direction_topic: /force_estimation/lifting_direction
  ```

### viewer機能を別のRVizに統合する
- 本ソフトウェアのviewerはRVizにtopic等の設定をしたものです．既にRVizを使っていてそこにviewer機能を統合することができます．
  - 推定した力の分布やlifting directionはMarkerArray，pointcloudのoverlayにはPointCloud2，対象物位置指定にはInteractiveMarkersの各topicを利用します．
  - 下図を参考にtopicの設定を行ってください．
![viewer](https://github.com/user-attachments/assets/af9633f7-afb6-4852-923f-47c1fbd885f3)


## Dockerを利用しない場合（under construction）

### 準備
```sh
$ git clone
$ pip install -r requirements.txt
```

### 実行
1. viewerを起動
  ```sh
  $ roslaunch force_estimation viewer_AIREC.launch
  ```
2. 認識プログラムを実行
  ```sh
  $ rosrun force_estimation demo_force_estimation.py
  ```
3. ログファイルを再生
  ```sh
  $ rosbag play rosbag-airec-sr300-rgbd_pointcloud_tf_2024-10-25-20-17-55.bag -l
  ```
