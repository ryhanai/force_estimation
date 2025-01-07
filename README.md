# Force Estimation
<img width="220" alt="image" src="https://github.com/user-attachments/assets/e8decee6-7674-491f-9283-64297a077b41" />

## Dockerを使う場合
### Docker imageのダウンロード
- Docker imageの実行とviewerの起動
  ```sh
  $ docker/run_force_estimation.sh
  $ roslaunch force_estimation viewer_AIREC.launch
  ```
- Docker imageの実行
  ```sh
  $ ...
  ```

- 学習済みモデルの[download](https://drive.google.com/file/d/1b1lcsoz_MxtpR1gUzYOYI5AWGufV3sU9/view?usp=sharing)と指定（configs/hydra_config.yaml）
  ```yaml
  check_point_dir: "../runs/20241017_0052_35"
  weight_file: '08000.pth'
  ```
- ログに対する推論をする場合はbagファイルを[download](https://drive.google.com/file/d/1b1lcsoz_MxtpR1gUzYOYI5AWGufV3sU9/view?usp=sharing)


## 環境設定
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

## ログに対する実行
1. 上でdownloadしたbagファイルを再生
  ```sh
  $ rosbag play rosbag-airec-sr300-rgbd_pointcloud_tf_2024-10-25-20-17-55.bag -l
  ```
2. viewerを起動
  ```sh
  $ roslaunch force_estimation viewer_AIREC.launch
  ```
3. 認識プログラムを実行
  ```sh
  $ rosrun force_estimation demo_force_estimation.py
  ```

## 認識プログラム実行中の設定変更
```sh
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
  - “Interactive_marker(0)”: interactive markerで指定されている位置を対象物の位置として推定を行います．このときはtopicによる指定を受け付けません．

## viewer機能を別のRVizに統合する
- 本ツールのviewerはRVizにtopic等の設定をしたものです．既にRVizを使っていてそこにviewer機能を統合することができます．
  - 推定した力の分布やlifting directionはMarkerArray，pointcloudのoverlayにはPointCloud2，対象物位置指定にはInteractiveMarkersの各topicを利用します．
  - 下図を参考にtopicの設定を行ってください．
![viewer](https://github.com/user-attachments/assets/af9633f7-afb6-4852-923f-47c1fbd885f3)

## AIRECシミュレータ（Gazebo）に対する実行
- 上記の「ログに対する実行」の替わりにAIRECのシミュレーション環境を起動することで，シミュレータ（Gazebo）から取得した画像に対して力分布の予測及びlifting directionの計算を行うことができます．
  - そのためには，最初にbagを再生するのでなく，AIRECのシミュレーション環境を起動します．以降の手順は同じです．
- viewer（RVis）において，pointcloudのtopicがログとシミュレータで異なるので適宜修正が必要な場合があります（シミュレータでは点群が/torobo/head/sr300/camera/depth/points）
