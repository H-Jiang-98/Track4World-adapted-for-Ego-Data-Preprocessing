# Track4World Ego-centric v1

这是一个基于 [TencentARC/Track4World](https://github.com/TencentARC/Track4World)
的实验性优化分支，当前基线为上游提交 `fbd59ff`。本仓库保留上游 Track4World
代码与接口，并新增面向 action trunk 内 first-frame 3D tracking 的 ego-centric 预处理入口。

> 原项目的安装、模型介绍和完整评测说明请参考
> [上游 README](https://github.com/TencentARC/Track4World/blob/fbd59ffadf2de9fccba5ea017e13af239075fbe9/README.md)。
> 下文新增的 Ego-centric 入口固定使用 DA3，不依赖 Pi3/Pi3X；上游原有的
> `demo.py`、评测脚本和其它 legacy 入口不在本次清理范围内，运行这些入口时仍应
> 遵循上游的依赖说明。

## 新增功能

- **全序列缓存**：对有效帧前缀只运行一次 DA3 与 `get_fmaps`，随后按 source
  切片缓存并复用原 tracking head，避免为每个 horizon 重复提取特征。
- **Ego-centric**：对全序列 DA3 预测的 `fx`、`fy`、`cx`、`cy` 取均值，构造唯一的
  相机内参 `K̄`；深度反投影、几何、point-map feature 和 tracking 投影使用同一个
  `K̄`。
- **可控时间调度**：`H` 表示包含 source 的 tracking horizon，`S` 表示相邻 source
  的步长；模型窗口 `W` 直接从 checkpoint 的 `time_emb` 与 `time_emb3d` 读取。
- **稳定输出协议**：所有 source 的 ragged tracks 写入唯一的
  `3d_ff_ego_output/flows.npz`，无需 pickle，包含有效性、置信度、相机位姿、
  统一相机内参 K、source query RGB、metric scale 和完整时间轴元数据；另在输出根目录
  写入 `camera_intrinsics.xml`，保存原图像素单位下的时间平均相机内参；写入和目录替换均为原子操作。
- **视频与图像序列输入**：支持 MP4 或按文件名排序的 RGB 目录；动态 mask 可选，
  缺失帧安全地按静态区域处理。
- **按 Horizon 交互可视化**：直接读取 `flows.npz`，可在不同 source horizon 之间切换，
  单独播放或拖动当前 horizon 内的 timestep，并提供 source-camera 正视角与世界场景视角。
- **配套验证**：包含缓存切片、坐标变换、边界条件、序列化和真实 checkpoint
  兼容性测试；另提供面向 Ego-centric `flows.npz` 的静态背景一致性评估工具。

## 修复的问题

- 修正 DA3 像素坐标与 `utils3d` 像素中心约定之间的半像素偏差，并在 resize、padding
  和 crop 后正确变换完整的逐帧内参，而不是只保留单个平均焦距。
- 先 resize depth 再按匹配内参重建 XYZ，避免直接插值 XYZ 在深度边缘破坏针孔关系；
  矩阵求逆固定在 float32 中进行。
- 在新增 Ego-centric core 中修复 `force_projection` 原地修改 2D flow、遗漏半像素偏移，
  以及 pairwise tracking 错用 source K 而非 target K 的问题；legacy 路径中的对应
  坐标修复见下方“上游文件原位修改”。
- 修复几何张量在 `B > 1` 时错误 `squeeze(0)` 的形状问题，并补齐 cache 中的
  world points、camera poses 与 intrinsics 时间轴。
- 推理窗口重叠区改为归一化 half-Hann 融合，使 2D flow、3D flow 和两个 confidence
  通道采用一致权重，避免 first-visit 拼接造成窗口边界跳变。
- `bilinear_sampler` 的归一化常量现在继承 `coords` 的 dtype/device，避免混合精度下的
  CPU/GPU 或 fp16/fp32 不一致。
- Grounded-SAM-2 预处理支持视频、单图和 RGB 目录，使用稳定的顺序文件名与对象 ID，
  生成 `frame_manifest.json`，并避免 CPU 模式下访问 CUDA device properties。
- 预处理目录整体替换，旧运行遗留的尾帧不会污染新结果；输入、checkpoint、cache、
  pose 和 NPZ schema 均增加了显式校验。

## 上游文件原位保留的功能性修改

以下修改有意保留在原来的上游文件位置。它们会改变运行时结果或输入输出协议，

| 文件 | 保留的修改 | 作用与范围 |
| --- | --- | --- |
| `track4world/nets/model.py` | `infer()` 和 `infer_pair()` 的 `force_projection` 使用临时 `flow_uv`，加入半像素中心偏移，并不再原地缩放返回的 `flow2d_c` | 修正像素中心约定和中间张量污染；这是 legacy 模型路径的几何修复。新增 Ego-centric core 还额外修正了 pairwise 的 target-K 选择，二者不要混为同一处修改。 |
| `track4world/nets/blocks.py` | `bilinear_sampler` 的归一化常量改用 `coords.new_tensor(...)` | 使常量继承坐标的 dtype/device，避免 CUDA 混合精度下的类型或设备不匹配；影响经过该公共采样器的推理路径。 |
| `scripts/run_dino_sam2.py` | 支持视频、单图和有序 RGB 目录；新增 `--max-frames`、`--preserve-source-names`；CPU 下跳过 CUDA 属性查询；输出采用确定性的帧内对象 ID、顺序文件名和 `frame_manifest.json` | 为 Ego-centric 动态 mask 预处理提供稳定的输入与输出协议；旧的 `--video-path` 仍兼容。 |

`demo.py` 中此前仅用于说明的注释和空白改动已回退，因此该上游文件目前没有本次
保留的功能性修改。上游 legacy 入口的其余 Pi3/Pi3X 依赖和行为也保持不变。

## 快速开始

Ego-centric 路径目前只支持 DA3 (`world_depthanythingv3`) 和 CUDA。先按上游方式创建
Python 3.11 / PyTorch 环境并安装依赖：

```bash
git lfs install
git lfs pull

conda create -n track4world python=3.11
conda activate track4world
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# Track4World 的运行时依赖；目录已由 .gitignore 排除
git clone https://github.com/jiah-cloud/utils3d.git
```

下载 DA3 checkpoint；若省略 `--ckpt_init`，入口也会尝试自动下载对应权重：

```bash
mkdir -p checkpoints
wget https://huggingface.co/TencentARC/Track4World/resolve/main/track4world_da3.pth \
  -O checkpoints/track4world_da3.pth
```

运行视频输入：

```bash
python demo_3dff_ego.py \
  --mp4_path demo_data/cat.mp4 \
  --ckpt_init checkpoints/track4world_da3.pth \
  --save_base_dir results/cat_ego \
  --coordinate world_depthanythingv3 \
  --H 32 \
  --S 16 \
  --Ts 63 \
  --metric_scale
```

运行有序 RGB 目录：

```bash
python demo_3dff_ego.py \
  --rgb_dir /path/to/rgb_frames \
  --rgb_fps 15 \
  --ckpt_init checkpoints/track4world_da3.pth \
  --save_base_dir results/sequence_ego \
  --coordinate world_depthanythingv3 \
  --H 32 \
  --S 8
```

如需动态 mask，可先按上游说明安装 Grounded-SAM-2，然后运行：

```bash
python scripts/run_dino_sam2.py \
  --input-path demo_data/cat.mp4 \
  --output-dir results/cat_ego \
  --text-prompt "cat."
```

该命令生成的 `results/cat_ego/mask/` 会被 ego-centric 入口自动读取。当前动态 mask
只用于保存预处理输入，不参与模型 query 筛选。

### 时间参数约束

- 输入帧数 `T ≥ H` 且 `T <= 128`；
- `H ≥ 3`，`1 ≤ S ≤ H`；
- `H` 必须能被 checkpoint 窗口 `W` 整除；当前 DA3 checkpoint 的 `W = 16`；
- 只保留完整 horizon：`N = ⌊(T − H) / S⌋ + 1`，有效前缀为
  `T_eff = (N − 1) × S + H`。

全序列几何与 feature 会同时驻留显存；遇到 OOM 时优先降低 `--Ts` 或
`--image_size`。

### GPU 显存不足：降低图片输入尺寸

`--image_size` 控制送入 Ego-centric 3D-FF 模型的 RGB 帧尺寸上限。它会按原始宽高比
缩放图像，并将得到的高、宽向下取整到 64 的倍数；因此实际尺寸可能略小于参数值，且
`--image_size` 不能小于 64。例如，可以将默认尺寸从 640 降到 448：

```bash
python demo_3dff_ego.py \
  --mp4_path demo_data/cat.mp4 \
  --ckpt_init checkpoints/track4world_da3.pth \
  --save_base_dir results/cat_ego_448 \
  --coordinate world_depthanythingv3 \
  --image_size 448 \
  --H 32 \
  --S 8 \
  --Ts 63
```

降低 `--image_size` 会同时减少 DA3、几何张量和 tracking feature 的空间尺寸，通常能
明显降低显存占用；若仍然 OOM，再降低 `--Ts` 或减少有效帧数。代价是空间细节和小目标
轨迹精度会下降。建议逐步尝试 512、448、384 等仍能为输入图像保留足够细节的值。

#### 是否需要重新生成动态物体 mask？

只改变 `demo_3dff_ego.py` 的 `--image_size` 时，不需要重新运行动态物体 mask 预处理：

- `scripts/run_dino_sam2.py` 生成的 mask 可以保留原始分辨率；
- 3D-FF 输入加载器会把每一帧 mask 使用最近邻插值调整到缩放后的 RGB 尺寸，保证 mask
  与模型输入逐像素对齐；
- mask 文件的帧顺序、帧编号和 RGB 输入必须保持一致，`--mask_dir` 仍指向同一目录即可；
- 当前动态 mask 只会保存到 `final_dyn_mask/`，不参与 3D-FF 的 query 筛选，因此不会
  因为降低 `--image_size` 而需要改变筛选逻辑。

如果 OOM 发生在 `scripts/run_dino_sam2.py` 本身，则降低 3D-FF 的 `--image_size` 不会
降低 SAM2 的显存占用。此时需要先将输入视频或 RGB 序列缩小后再运行 mask 预处理，并用
同一批（同样的帧数、顺序和编号）的缩小后 RGB 输入运行 3D-FF。不要使用双线性插值手动
缩放离散 mask；当前加载器会使用最近邻插值，避免产生新的半透明类别值。

## 输出

核心文件为 `results/<run>/3d_ff_ego_output/flows.npz`，以及同级的
`results/<run>/camera_intrinsics.xml`：

| 字段 | 形状 | 含义 |
| --- | --- | --- |
| `query_uv` / `query_offsets` | `[Q,2]` / `[M+1]` | 各 source 的 ragged 查询与列边界 |
| `query_rgb` | `[Q,3]` uint8 | 与 query_uv 对齐的 source 帧 RGB 颜色 |
| `track_xyz` / `track_valid` | `[H,Q,3]` / `[H,Q]` | source-camera 坐标中的轨迹与几何有效性 |
| `confidence` | `[H,Q,2]` | tracking head 的两个置信度通道 |
| `ego_K` / `c2w` | `[3,3]` / `[T_eff,4,4]` | 实际使用的内参与有效序列位姿 |
| `source_frame_index` / `target_frame_index` | `[M]` / `[M,H]` | 完整的全局时间索引 |

`ego_K` 保持模型实际使用的缩放后、未 padding 输出网格单位；`camera_intrinsics.xml` 将
同一个时间平均 K 按原图与缩放图的宽高比例分别换算为原图像素单位。XML 使用 OpenCV
`FileStorage` 兼容的 `camera_matrix` 节点，并同时提供 `fx`、`fy`、`cx`、`cy` 和原/缩放
尺寸元数据；可通过 `--intrinsics_xml` 指定输出路径。

```python
import numpy as np

with np.load(
    "results/cat_ego/3d_ff_ego_output/flows.npz",
    allow_pickle=False,
) as flows:
    source = 0
    q0 = int(flows["query_offsets"][source])
    q1 = int(flows["query_offsets"][source + 1])
    xyz = flows["track_xyz"][:, q0:q1]
    rgb = flows["query_rgb"][q0:q1]
```

### 静态背景评估

`evaluate_3dff_static_background.py` 读取 `flows.npz` 和 source 对应的动态 mask，按
每个 source/horizon 独立统计 MaxDrift、RobustMaxDrift 与 Coverage。评估器根据
`source_frame_index` 在 `--mask-dir` 中查找 `mask_XXXX.png`，再用 `query_offsets`
选取该 source 的查询列；前两项以全局 frame 0 的静态查询中位深度归一化为百分比，
Coverage 表示目标帧仍有效的静态查询比例。MaxDrift/RobustMaxDrift 越低越好，Coverage
越高越好。默认在 `flows.npz` 同目录写入 `static_background_metrics.json` 和
`static_background_metrics.png`，可用 `--output`、`--plot-output` 覆盖路径。

```bash
python evaluate_3dff_static_background.py \
  --flows-path results/cat_ego/3d_ff_ego_output/flows.npz \
  --mask-dir results/cat_ego/final_dyn_mask
```

### 棋盘格 metric-scale 评估

`evaluate_chessboard_scale.py` 使用每个 source 的 `track_xyz[0]`（即
`demo_3dff_ego.py` 保存的 source DA3 geometry）评估绝对尺度。它不会使用
`j>0` 的 point-flow 结果，也不会把不同 source 的 source-camera 坐标直接混合。
棋盘角点从与 `query_uv` 同尺寸的 `final_rgb/` 中检测，随后在 source 点云上做四邻域
双线性采样。默认棋盘规格为 OpenCV 内角点 `8×11`；当前视频对应 `9×12` 个方格。

方格边长必须以米传入，并且 `flows.npz` 必须由 `--metric_scale` 生成：

```bash
python evaluate_chessboard_scale.py \
  --flows-path results/chessboard_448/3d_ff_ego_output/flows.npz \
  --pattern-cols 8 \
  --pattern-rows 11 \
  --square-size-m <实际方格边长米数>
```

评估器对全部 source 分别进行整板相似变换拟合，输出预测/真实尺度比 α（理想值为 1）、
有符号和绝对尺度误差、拟合 RMSE，以及水平/竖直相邻边长误差。默认在
`flows.npz` 同目录生成 `chessboard_scale_metrics.json`、
`chessboard_scale_metrics.png` 和 `chessboard_corner_overlays/`。检测失败或 source 角点
3D 无效时会跳过并在 JSON 中记录原因；所有 source 都失败时命令以错误退出。

## 代码结构

| 路径 | 职责 |
| --- | --- |
| `demo_3dff_ego.py` | CLI、时间调度、坐标转换、NPZ 与原图像素单位内参 XML 输出 |
| `_demo_3dff_support.py` | 输入、mask、checkpoint 与预处理辅助函数 |
| `_demo_3dff_common.py` | Legacy 3D-FF 通用辅助逻辑与 PLY/NPY 输出兼容层 |
| `track4world/nets/model_3dff_ego.py` | Ego-centric adapter 与 `SequenceCache` |
| `track4world/nets/_model_3dff_core.py` | Ego-centric 使用的 3D-FF 基线核心 |
| `visualization/vis_3d_ff_ego.py` | 读取 `flows.npz`，按 source/horizon 交互显示点云、轨迹和相机位姿 |
| `evaluate_3dff_static_background.py` | 按 source/horizon 统计静态背景 MaxDrift、RobustMaxDrift、Coverage，并绘制汇总图 |
| `evaluate_chessboard_scale.py` | 根据 source 棋盘格角点 3D geometry 评估 metric-scale 准确性 |
| `tests/` | CPU 合约测试与可选 CUDA 数值兼容测试 |

### Ego-centric 轨迹可视化

可视化器直接读取 `demo_3dff_ego.py` 生成的 `flows.npz`，不需要再次读取原视频：

```bash
python visualization/vis_3d_ff_ego.py \
  --npz_path results/cat_ego/3d_ff_ego_output/flows.npz
```

例如查看 `test_screen.mp4` 对应的输出：

```bash
python visualization/vis_3d_ff_ego.py \
  --npz_path results/screen_ego/3d_ff_ego_output/flows.npz
```

`Source frame` 下拉框中的值是每个完整 horizon 的全局起始帧；切换它会只加载并显示
该 source 拥有的 ragged query 列。`Horizon timestep` 表示当前 horizon 内的相对位置，
范围为 0 到 H − 1，其全局目标帧由 `target_frame_index` 给出。可以使用 `Playing`
连续播放当前 horizon，也可以暂停后拖动滑块逐帧检查。所有 horizon 具有相同长度，因此
切换 source 时会保留当前 timestep。命令行参数 `--initial_source` 使用从 0 开始的
source 序号，而不是全局帧号。

选中 source 的轨迹最初存储在 source-camera 坐标中；可视化器通过该 source 对应的
`c2w` 将其转换到统一世界坐标。点云使用 `query_rgb`，轨迹使用稳定的高饱和 HSV 颜色，
同一条轨迹在整个 horizon 内保持同色。默认显示 5 帧局部尾迹；勾选
`Full trajectory (all horizon)` 可显示完整 horizon 路径，`Show track heads` 使用相同
轨迹颜色标出当前端点。

相机视角不使用 Viser 的固定默认方向，而是根据当前 source 的 `c2w` 初始化：相机朝向
source-camera 的正 Z 光轴，图像上方向与原视频一致。这样对于 `test_screen.mp4` 这类
原相机正对屏幕的输入，首次打开以及每次切换 `Source frame` 后都会自动得到正视角。

| 控件 | 作用 |
| --- | --- |
| `View source camera` | 随时恢复当前 source 的原视频观察方向；鼠标旋转也会从该方向和合理的观察目标开始 |
| `Frame scene` | 将 timestep 重置为 0，并从斜上方显示当前 source 的整体三维范围 |
| `Query downsample` / `Max render points` | 控制当前点云的采样步长和最大显示点数 |
| `Trajectory downsample` / `Max trajectory segments` | 控制轨迹采样步长和完整轨迹的最大线段数 |
| `Trail length` | 控制非完整轨迹模式下保留的历史 timestep 数量 |
| `Show points` / `Show tracks` / `Show track heads` | 分别控制 RGB 点云、轨迹线和当前轨迹端点 |
| `Show cameras` | 显示有效序列的相机路径和当前 source camera 坐标轴 |

较大的 `flows.npz` 会按需缓存最近使用的 source，并自动根据显示预算提高采样步长；仍然
卡顿时，可先提高 `--point_downsample` 或 `--trajectory_downsample`，或降低
`--max_render_points` 和 `--max_trajectory_segments`。

### 数值兼容测试

`tests/test_ego_cuda_compatibility.py` 使用真实的
`checkpoints/track4world_da3.pth`，以固定随机种子生成两个模型窗口长度的
半精度输入序列，然后在相同 checkpoint 和随机种子下分别执行直接窗口推理和
`encode_sequence()` + `track_cached_window()`。测试会比较：

- DA3 的逐帧原始内参、Ego-centric 内参和 metric scale；
- dense geometry、3D flow、confidence map 以及 camera-to-world pose；
- 将 dense 结果转换到 source-camera 坐标后得到的稀疏 source tracks。

内参和 scale 使用严格容差，模型输出使用适合 fp16 推理的相对/绝对容差（`2e-3`）。
因此它验证的是“加入缓存和 Ego-centric 后，tracking head 的数值行为保持一致”，而不只是
检查文件是否发生变化。没有 CUDA 或对应 checkpoint 时，该测试会自动跳过；其余缓存切片、
坐标、有效性、边界条件和无 pickle 序列化测试可在 CPU 上运行。

```bash
conda run -n track4world python -m unittest discover -s tests -v
```

更完整的实现与 schema 说明见 [Ego-centric design](docs/ego-design.md)。


## TODO for v1
- [ ] 面向 `T > 128` 的长视频输入。
- [ ] 外源相机内外参真值的输入。

## 上游与许可证

本仓库是 Track4World 的衍生工作，原项目版权、引用方式和模型许可保持不变。使用前请阅读
[LICENSE.txt](LICENSE.txt) 及上游项目说明。
