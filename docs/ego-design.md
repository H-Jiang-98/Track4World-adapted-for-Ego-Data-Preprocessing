# Ego-centric 全序列缓存设计说明

本入口固定使用 DA3，对有效输入前缀只运行一次 DA3 和 get_fmaps，然后为每个 source
切片 cache 并调用未修改的 legacy Tracking Head。该新增入口不依赖 Pi3/Pi3X；本次
清理不修改上游 legacy 入口。最终三维输出由以下两个文件组成：

3d_ff_ego_output/flows.npz

camera_intrinsics.xml（位于输出根目录，与 `3d_ff_ego_output/` 同级）

实现文件：

- demo_3dff_ego.py：CLI、全序列调度、有效性、坐标变换、NPZ 和原图像素单位内参 XML；
- _demo_3dff_support.py：3D-FF 独立使用的输入、checkpoint 和基础输出辅助函数；
- track4world/nets/model_3dff_ego.py：ego-centric adapter、SequenceCache、
  encode_sequence 和 track_cached_window；
- track4world/nets/_model_3dff_core.py：Ego-centric 使用的 3D-FF 基线 core；
- _demo_3dff_common.py：legacy 输出逻辑保持不变，仅将辅助函数导入切换到独立的
  _demo_3dff_support.py。

## 时间轴与运行时 W

--H 和 --S 都是必填参数。W 不使用硬编码 fallback，而是从 checkpoint 的 time_emb 和
time_emb3d 时间维读取；两者必须相等，且 W 必须为不小于 2 的偶数。当前
track4world_da3.pth 的 W=16。

输入验证：

- T <= 128；
- H ≥ 3 且 T ≥ H；
- H 能被 W 整除；
- 1 ≤ S ≤ H。

source 为 0, S, 2S, …，只保留完整 horizon：

- N = ⌊(T − H) / S⌋ + 1
- T_eff = (N − 1) × S + H

RGB、动态 mask、DA3 输入和 cache 都截断为 [0,T_eff)。即使没有尾帧被截断，
truncated_frame_start 也固定等于 T_eff。final_rgb/ 和 final_dyn_mask/ 会通过临时
目录整体替换，因此旧运行的尾帧不会残留。

外层 S 只选择 source，绝不传入模型核心的 stride。每个 horizon 调用
infer(..., eval_dict=window_cache, window_len=W, stride=None)。因此 legacy 内部仍使用
W/2 overlap，并以 half-Hann 对 2D flow、3D flow 和两个 post-sigmoid confidence 通道
做融合。

## 全序列平均 K 与 cache

DA3 对全部 T_eff 帧只调用一次。adapter 从每帧原始像素内参提取 fx、fy、cx、cy，
在整个有效序列上计算算术平均，再构造：

K̄ = [[fx̄, 0, cx̄], [0, fȳ, cȳ], [0, 0, 1]]

同一个 K̄ 用于全部 source 和 target 的 depth 反投影、geometry、point-map feature 和
最终 tracking 投影。均值使用 float32 且关闭 autocast；不平均 skew 或齐次行。

SequenceCache 固定 B=1，保存并验证：

- T_eff、原图尺寸、64 对齐后尺寸、四侧 padding、输入 dtype/device；
- W、模型 mask threshold、metric-scale 原始值及 enabled 状态；
- 原始 DA3 K、全序列平均 K、未 padding 输出像素网格上的 ego_K；
- fmaps、ctxfeats、fmaps3d_detail、pms；
- camera/world geometry、mask 和 C2W pose。

encode_sequence(images) 只做 legacy 相同的归一化、空间 padding 和唯一一次 get_fmaps，
不会调用 Tracking Head。track_cached_window 将所有时间轴切成 window-local
[source:end)，使全局 source 成为局部 anchor 0，并在 tracking 前恢复和核对全序列
metric scale。

## 查询、有效性与坐标

每个 source 的查询 mask 是：

模型 source mask ∧ source XYZ 有限 ∧ z>0 ∧ 非 depth edge

随后沿用 legacy 的 image_mesh(..., tri=True) 生成 query_uv。动态 mask 仅作为预处理
输入保存，不参与模型查询筛选。

对局部时间 j：

- j=0 的 XYZ 直接取 DA3 source geometry；track_valid=True；
- j=0 的两个 confidence 值仍来自 Tracking Head 的 half-Hann 融合结果；
- j>0 先在完整 target-camera dense tracking map 上计算 XYZ 有限、z>0、非 depth-edge
  的有效性，再按 source UV 采样；
- target mask 的同位置像素不参与有效性，depth edge 也不会在稀疏查询点上计算；
- 有效 target-camera 点通过 Pₛ⁻¹PₜXₜ 转到该 source camera 坐标；
- 无效 XYZ 保存 NaN，但两个 confidence 值始终保留且不影响 track_valid。

这是有意的 j=0 输出例外：NPZ 的 j=0 以 DA3 source geometry 为准，不要求等于
旧版 flow_000。

ego_K 已换算到未 padding 的输出图像整数 UV 网格，和 query_uv 使用相同像素约定。

`camera_intrinsics.xml` 保存同一个全序列时间平均 K，但将 fx、fy、cx、cy 从未 padding
的 resize 后网格换算到原始 RGB 图像尺寸的像素单位。由于高度和宽度可能分别向下取整到
64 的倍数，横纵方向使用独立比例 `original_width / resized_width` 和
`original_height / resized_height`。XML 根节点为 `opencv_storage`，矩阵位于
`camera_matrix`，并附带 `fx`、`fy`、`cx`、`cy`、原始/缩放尺寸和比例元数据。

## flows.npz schema

所有字符串使用固定宽度 Unicode；文件可用 allow_pickle=False 读取。M 是 source 数量，
H 是 horizon，Q 是所有 source 查询数之和。时间是 track 数组的第一轴，查询是第二轴。

| 字段 | dtype / shape | 含义 |
| --- | --- | --- |
| schema_version | Unicode scalar | track4world.ego_flows.v1 |
| coordinate_system | Unicode scalar | source_camera |
| pixel_convention | Unicode scalar | unpadded_uv_integer |
| source_frame_index | int32 [M] | 每个 horizon 的全局 source |
| target_frame_index | int32 [M,H] | target[i,j]=source[i]+j |
| query_offsets | int64 [M+1] | source 的 ragged 查询列边界 |
| query_uv | int32 [Q,2] | source 图像上的 (u,v) |
| query_rgb | uint8 [Q,3] | 与 query_uv 对齐的 source 帧 RGB 颜色 |
| track_xyz | float32 [H,Q,3] | 各 source-camera 坐标中的 XYZ |
| track_valid | bool [H,Q] | 几何有效性 |
| confidence | float32 [H,Q,2] | 两个融合后的 post-sigmoid 通道 |
| ego_K | float32 [3,3] | 实际推理使用的输出像素内参 |
| metric_scale | float32 scalar | 全序列 DA3 原始 scale，始终保存 |
| metric_scale_enabled | bool scalar | scale 是否已应用到 XYZ 和 C2W translation |
| c2w | float32 [T_eff,4,4] | 全部有效帧的 C2W，与 XYZ 同尺度 |
| input_frame_count | int32 scalar | 截断前 T |
| effective_frame_count | int32 scalar | T_eff |
| truncated_frame_start | int32 scalar | 始终等于 T_eff |
| horizon_length | int32 scalar | H |
| source_stride | int32 scalar | 外层 S |
| model_window_length | int32 scalar | checkpoint W |
| image_size_hw | int32 [2] | 未 padding 输出尺寸 |
| last_complete_source_frame_index | int32 scalar | 最后完成的 source |

query_offsets[0]=0，其值单调不减，末项等于 Q。source i 使用查询列区间
[query_offsets[i], query_offsets[i+1])；允许区间为空。不使用 Qmax padding，也不使用
对象数组。

query_rgb 与 query_uv 使用相同的 Q 行顺序；颜色来自模型实际输入的 resize 后 source
RGB 帧，采用 uint8 `[R,G,B]`，在所有 horizon timestep 中复用。

metric scale 关闭时，XYZ 和 C2W translation 保持 Track4World 归一化尺度，但
metric_scale 仍保存原始值。开启时，同一个全序列 scale 同时应用到 XYZ、world
geometry 和 pose translation。

## 运行

视频输入：

~~~bash
conda run -n track4world python demo_3dff_ego.py \
  --mp4_path demo_data/cat.mp4 \
  --save_base_dir results/cat_ego \
  --mask_dir results/cat_ego/mask \
  --coordinate world_depthanythingv3 \
  --H 32 \
  --S 8 \
  --Ts 63 \
  --metric_scale
~~~

有序 RGB 目录输入：

~~~bash
conda run -n track4world python demo_3dff_ego.py \
  --rgb_dir demo_data/test_screen/Color \
  --rgb_fps 15 \
  --save_base_dir results/test_screen_ego \
  --coordinate world_depthanythingv3 \
  --H 16 \
  --S 4 \
  --Ts 63
~~~

读取：

~~~python
import numpy as np

with np.load(
    "results/cat_ego/3d_ff_ego_output/flows.npz",
    allow_pickle=False,
) as flows:
    i = 0
    q0 = int(flows["query_offsets"][i])
    q1 = int(flows["query_offsets"][i + 1])
    xyz_for_source_i = flows["track_xyz"][:, q0:q1]
~~~

### 静态背景评估

`evaluate_3dff_static_background.py` 面向本入口生成的 ragged `flows.npz`，按每个
source/horizon 独立统计三项时序一致性指标：MaxDrift、RobustMaxDrift 和 Coverage。
评估器先根据 `source_frame_index` 在 `--mask-dir` 中查找对应的 `mask_XXXX.png`，再用
`query_offsets` 取得该 source 的查询列；漂移以全局 frame 0 静态查询的中位深度归一化
为百分比，Coverage 为目标帧有效静态查询比例。前两项越低越好，Coverage 越高越好。
默认在 `flows.npz` 同目录生成 `static_background_metrics.json` 和
`static_background_metrics.png`，也可通过 `--output` 与 `--plot-output` 指定路径。

运行示例：

~~~bash
python evaluate_3dff_static_background.py \
  --flows-path results/cat_ego/3d_ff_ego_output/flows.npz \
  --mask-dir results/cat_ego/final_dyn_mask
~~~

### 棋盘格 metric-scale 评估

`evaluate_chessboard_scale.py` 读取本入口的 `flows.npz` 和同一模型尺寸下的
`final_rgb/`。每个 source 只使用 `track_xyz[0, q0:q1]`，因为该切片是
`build_source_tracks` 保存的 source DA3 geometry；目标时间 `j>0` 的 point-flow 不参与
绝对尺度评估。棋盘角点使用 `cv2.findChessboardCornersSB` 检测，并在 source 点云的四个
整数像素邻域上做双线性采样。

参数中的 `--pattern-cols/--pattern-rows` 是 OpenCV 内角点数量。`chessboard.mp4` 中的
棋盘为 `8×11` 个内角点（即 `9×12` 个方格），真实方格边长通过必填的
`--square-size-m` 以米传入。评估要求 `metric_scale_enabled=True`，因此不会把归一化
坐标误判为米制坐标。

每个 source 用全部角点拟合带旋转、平移和统一尺度的相似变换，得到尺度比 α、预测方格
边长、相对误差和拟合 RMSE；同时报告水平/竖直相邻边长误差。默认输出
`chessboard_scale_metrics.json`、`chessboard_scale_metrics.png` 和
`chessboard_corner_overlays/`。单个 source 检测或 3D 采样失败会跳过并保留原因，全部
失败时命令报错。

~~~bash
python evaluate_chessboard_scale.py \
  --flows-path results/chessboard_448/3d_ff_ego_output/flows.npz \
  --pattern-cols 8 \
  --pattern-rows 11 \
  --square-size-m <实际方格边长米数>
~~~

`visualization/vis_3d_ff.py` 仍读取旧版 PLY/NPY 布局。

`visualization/vis_3d_ff_ego.py` 读取本入口的 `flows.npz`，将所选 source 的
source-camera XYZ 通过对应 `c2w` 变换到统一世界坐标。点云沿用 `query_rgb`，轨迹线使用
稳定的高饱和 HSV 颜色，并在同一条轨迹的相邻时间段保持连续同色，避免颜色跳变
破坏路径的可读性。
可视化器按 source 懒加载派生 world tracks，缓存最近两个 source，并在连续播放时只追加
最新的轨迹线段；默认显示旧版相同的 5 帧局部尾迹，`--max_render_points` 用于控制
点云 query 的自适应降采样预算。首次连接或 “Frame selected source” 按钮会按所选
source 自动取景。勾选 “Full trajectory (all horizon)” 时会显示完整路径，并由
“Max trajectory segments” 限制总线段数；关闭后则按 “Trail length” 只显示局部尾迹。
点云与轨迹线分别使用 “Query downsample” 和 “Trajectory downsample”，后者默认采用
旧版的 50，并允许使用更大的步长来减少密集 query 的遮挡；“Show track heads”
会叠加当前时间端点，帮助在 RGB 点云遮挡时辨认轨迹。
点云预算由 `Max render points` 控制，完整轨迹预算由 `Max trajectory segments` 控制。
