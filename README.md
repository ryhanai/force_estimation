# Force Estimation
<img width="220" alt="image" src="https://github.com/user-attachments/assets/e8decee6-7674-491f-9283-64297a077b41" />

## 準備
- 学習済みモデルの[download](https://drive.google.com/file/d/1b1lcsoz_MxtpR1gUzYOYI5AWGufV3sU9/view?usp=sharing)と指定（configs/hydra_config.yaml）
```yaml
check_point_dir: "../runs/20241017_0052_35"
weight_file: '08000.pth'
```
- ログに対する推論をする場合はbagファイルを[download](https://drive.google.com/file/d/1b1lcsoz_MxtpR1gUzYOYI5AWGufV3sU9/view?usp=sharing)
- Docker Image (coming soon ...)

## 環境設定
### 座標系の設定（launch/viewer_AIREC.launch）
```xml
<node pkg=“tf” type=“static_transform_publisher” name=“fmap_frame_broadcaster” args=“0.605 0.0 0.1 0 0 0 world fmap_frame 100” />
```
- 説明
  - 予測したforcemap及びlifting directionはworld座標系であるので，学習したsimulation環境と実行時の環境の位置合わせが必要
  - forcemapはfmap_frame相対で結果を出力するので，/world -> /fmap_frameのtransformを設定する
  - ただし，table上にはxyを合わせる目印がないので，camera姿勢を大まかに揃える方が容易
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
- 上でdownloadしたbagファイルを再生
```sh
$ rosbag play rosbag-airec-sr300-rgbd_pointcloud_tf_2024-10-25-20-17-55.bag -l
```
- viewerを起動
```sh
$ roslaunch force_estimation viewer_AIREC.launch
```
- 認識プログラムを実行
```sh
$ rosrun force_estimation demo_force_estimation.py
```

## 認識プログラム実行中の設定変更
```sh
$ rqt
```

## viewer機能を別のRVizに統合する
- 本ツールのviewerはRVizにtopic等の設定をしたものです．既にRVisを使っていてそこにviewer機能を統合することができます．
- 画像

- 実行画面
  - 対象物の位置をtopicで送る場合には，rqtでobject_position=“Object_recognition(1)”に設定しておく
  - object_position=“Interactive_marker(0)”の場合は，interactive markerで指定されている位置を対象物の位置として推定する

## AIRECシミュレータ（Gazebo）に対する実行
- AIRECの環境を起動する（省略）
- 上記の「ログに対する実行」の「viewerを起動」以降の手順を実行する
- viewer（RVis）において，pointcloudのtopicがログとシミュレータで異なるので適宜修正する（　．．． ）
