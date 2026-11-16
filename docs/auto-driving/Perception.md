# 自动驾驶感知

??? info "版本信息"
	Version 1.0.0 Created on 2026-11-16

	* 首次添加：感知系统分层、传感器与标定、2D/3D 感知、BEV、多传感器融合、端到端、评测与部署
	* 待补充：路侧雷视一体标定实验记录、自采数据集的占用栅格标注结果

感知（Perception）解决三个问题：**有什么**、**在哪里**、**往哪去**。它把多源传感器的原始数据转换成带有语义和运动状态的目标级/栅格级描述，是下游预测与规划的唯一信息来源，也是整个自动驾驶链路里最吃数据、最吃算力的一环。

## 1. 感知在系统中的位置

```text
传感器 ──▶ 感知 ──▶ 预测 ──▶ 规划 ──▶ 控制
 (相机/LiDAR/雷达)  (检测/跟踪/占用)  (轨迹预测)  (行为+轨迹)  (横纵向)
```

典型的感知输出接口如下，工程上这层接口一旦定型就很难改，设计时要把字段冗余（置信度、协方差、跟踪 ID、遮挡标记）留足：

| 输出 | 内容 | 典型频率 | 使用方 |
| --- | --- | --- | --- |
| 目标列表 | 类别、位置 $(x,y)$、尺寸 $(l,w,h)$、朝向 $\theta$、速度、协方差、跟踪 ID | 10–20 Hz | 预测、规划 |
| 车道线/路沿 | 三次多项式或点序列、线型、置信度 | 10 Hz | 规划、地图匹配 |
| 可行驶区域 | BEV 语义栅格 / 轮廓多边形 | 10 Hz | 规划（安全兜底） |
| 占用栅格 | 体素或 BEV 栅格占用概率、流场 | 10 Hz | 规划（不规则障碍物） |
| 交通要素 | 红绿灯状态、限速牌、施工锥桶 | 事件触发 | 决策 |
| 自车位姿 | 位置、姿态、速度、IMU 偏置 | 50–100 Hz | 定位、融合 |

!!! note "时延比精度更容易被忽视"
    10 Hz 的控制周期只有 100 ms，感知端到端（曝光 → 结果发出）通常要吃掉 60–80 ms。一个 mAP 高 3 个点但慢 30 ms 的模型，在实车上往往更差。

## 2. 传感器配置与特性

| 传感器 | 优势 | 短板 | 典型参数 | 感知角色 |
| --- | --- | --- | --- | --- |
| 前视/环视相机 | 语义丰富（颜色、文字、纹理），成本低 | 测距弱、受光照/逆光/雨雾影响大 | 8 MP、30 fps、FOV 30°–120° | 分类、车道线、标志、红绿灯 |
| 机械/半固态 LiDAR | 直接三维测距、不受光照影响 | 雨雾衰减、点云稀疏、成本高 | 64/128 线、10 Hz、120 m | 3D 检测、占用、定位 |
| 毫米波雷达 | 测速准、全天候、成本低 | 角分辨率低、金属反射虚警、静止目标差 | 77 GHz、4D 成像 | 测速、远距跟踪、冗余 |
| 超声波 | 近距、便宜 | 距离极短 | 0.2–5 m | 泊车 |
| GNSS/RTK + IMU | 绝对位姿、高频 | 隧道/城市峡谷失效 | 100 Hz IMU | 定位、运动补偿 |
| 轮速/方向盘 | 可靠、便宜 | 打滑误差 | 50 Hz | 航位推算 |

车路协同场景下，路侧还会布设**固定视角相机 + 雷视一体机**，其特点是标定后位姿不变、视角俯视、覆盖范围大，但会引入通信时延与跨设备坐标系对齐问题（详见第 10 节）。

## 3. 坐标系与标定

### 3.1 坐标链

```text
world (UTM/ECEF) ─▶ ego (车辆后轴中心) ─▶ sensor (各传感器本体) ─▶ image (像素)
```

标定的目标就是求出一串外参 $T_{s}^{e}$（sensor→ego）与内参 $K$，并且保证它们随时间是稳定的（温度、振动、装配公差都会让外参漂移，所以量产车要有**在线标定/外参自监控**）。

### 3.2 相机模型

设世界点为 $P_w=(X,Y,Z,1)^T$，投影到像素：

\[
\tilde{p} = K \,[R\,|\,t]\, P_w, \qquad
K = \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix}
\]

带畸变（径向 + 切向，OpenCV 模型 $k_1,k_2,p_1,p_2,k_3$）时，归一化坐标先做畸变再乘内参：

\[
x_d = x\frac{1+k_1r^2+k_2r^4+k_3r^6}{1} + 2p_1xy + p_2(r^2+2x^2), \quad r^2=x^2+y^2
\]

!!! tip "标定误差的放大效应"
    外参角度误差 $\delta\theta$ 在距离 $d$ 处造成的横向误差约为 $d\cdot\delta\theta$。0.1° 的误差在 100 m 处就是 17 cm，这是 BEV 融合里目标"重影"的常见原因。

### 3.3 点云投影到图像（实用片段）

```python
import numpy as np

def lidar_to_image(pts_lidar, K, T_velo_to_cam, img_shape, min_depth=1.0):
    """pts_lidar: (N,3) 激光雷达坐标; T_velo_to_cam: 4x4"""
    N = pts_lidar.shape[0]
    pts_h = np.hstack([pts_lidar, np.ones((N, 1))]).T          # (4,N)
    pts_cam = (T_velo_to_cam @ pts_h)[:3]                       # (3,N)
    depth = pts_cam[2]
    valid = depth > min_depth
    uv = (K @ pts_cam)[:, valid]
    uv = uv[:2] / uv[2]                                         # 归一化
    u, v = uv[0], uv[1]
    h, w = img_shape[:2]
    mask = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    return np.stack([u[mask], v[mask]], axis=1), depth[valid][mask]
```

### 3.4 时间同步

- **硬同步**：PTP / gPTP（IEEE 802.1AS）给全车传感器打统一时间戳，抖动可到微秒级；低成本方案用 GPS 秒脉冲 + 触发线。
- **软同步**：按时间戳插值。相机 30 Hz、LiDAR 10 Hz、IMU 100 Hz 之间必然存在时间差，运动中的车辆 20 ms 的错位就是 0.3 m 的位置误差（30 km/h 下）。
- **运动补偿**：把点云按 IMU/轮速积分到同一时刻，再做融合，否则高速场景点云会"拖尾"。

## 4. 2D 视觉感知

### 4.1 目标检测

- **YOLO 系列**：单阶段 anchor-based（v3–v5）→ anchor-free + 解耦头（v6–v8）→ 端到端免 NMS（YOLOv10）。工程上仍是性价比最高的选择，TensorRT 部署链路最成熟。
- **DETR 系列**：set prediction + 匈牙利匹配，天然免 NMS；RT-DETR 用混合编码器把实时性做到了 30 FPS 以上。
- **小目标与远距离**：靠高分辨率输入、切片推理（SAHI）或专门的远距检测头解决，代价是算力。

IoU 与 NMS 是所有检测后处理的基础：

\[
\mathrm{IoU} = \frac{|A \cap B|}{|A \cup B|}
\]

```python
import numpy as np

def nms(boxes, scores, iou_thr=0.5):
    """boxes: (N,4) xyxy"""
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_o = (boxes[order[1:], 2] - boxes[order[1:], 0]) * \
                 (boxes[order[1:], 3] - boxes[order[1:], 1])
        iou = inter / (area_i + area_o - inter + 1e-9)
        order = order[1:][iou <= iou_thr]
    return np.array(keep, dtype=np.int64)
```

### 4.2 分割与车道线

- **语义 / 实例 / 全景分割**：Mask2Former 用 masked attention 统一三类任务；实车多用量化后的轻量分割头做可行驶区域兜底。
- **车道线**：早期是逐行分类（UFLD/Ultra-Fast），现在是 query-based（LaneATT、PETR）+ 3D 车道线（PersFormer、MapTR）。输出通常是 3D 点序列而不是 2D 像素，否则投影回 BEV 会二次放大误差。
- **BEV 地图**：MapTR / MapTRv2 把矢量化地图元素当作多边形点集回归，是"重感知轻地图"方案的核心组件。

### 4.3 单目测距

针孔几何：已知目标在图像中的底边接地点 $v$、相机高度 $H$、俯仰角，则

\[
d \approx \frac{f_y H}{v - c_y}
\]

平路、目标贴地的假设下误差可接受；坡道、载重变化、俯仰抖动会直接破坏这个假设，所以量产方案会用多帧跟踪 + 地面平面估计来平滑。

## 5. 3D 感知：点云与 BEV

### 5.1 点云表示

| 表示 | 代表工作 | 特点 |
| --- | --- | --- |
| Point-based | PointNet++、PointRCNN | 保留原始精度，稀疏卷积/采样开销大 |
| Voxel-based | VoxelNet、SECOND、CenterPoint | 规则卷积友好，量化损失 |
| Pillar-based | PointPillars | 极快，适合实时，Z 方向信息损失 |
| Range image | RangeDet | 计算高效，遮挡/尺度畸变明显 |

体素化就是把连续点云量化到网格：

\[
i = \left\lfloor \frac{x - x_{\min}}{\Delta_x} \right\rfloor,\quad
j = \left\lfloor \frac{y - y_{\min}}{\Delta_y} \right\rfloor,\quad
k = \left\lfloor \frac{z - z_{\min}}{\Delta_z} \right\rfloor
\]

```python
import numpy as np

def voxelize(points, voxel_size=(0.1, 0.1, 0.2), pc_range=(-51.2, -51.2, -5.0, 51.2, 51.2, 3.0)):
    """points: (N, >=3)，返回每个点所属体素索引与唯一体素列表"""
    x_min, y_min, z_min, x_max, y_max, z_max = pc_range
    grid = np.array([
        (x_max - x_min) / voxel_size[0],
        (y_max - y_min) / voxel_size[1],
        (z_max - z_min) / voxel_size[2],
    ], dtype=np.int64)
    idx = ((points[:, :3] - np.array([x_min, y_min, z_min])) / np.array(voxel_size)).astype(np.int64)
    inside = np.all((idx >= 0) & (idx < grid), axis=1)
    idx = idx[inside]
    flat = idx[:, 0] * grid[1] * grid[2] + idx[:, 1] * grid[2] + idx[:, 2]
    uniq, inv = np.unique(flat, return_inverse=True)
    return idx, flat, uniq, inv
```

### 5.2 BEV 范式

BEV（Bird's Eye View）把多相机图像统一到俯视栅格，好处是**尺度一致、便于融合、便于下游规划**。

- **LSS（Lift-Splat-Shoot）**：每个像素预测离散深度分布，把特征"抬"到 3D 视锥，再 splat 到 BEV 栅格。
- **BEVFormer**：用 BEV query 通过可变形注意力从多视角、多时刻采样特征，时序融合靠 ego 位姿对齐，是 nuScenes 上的经典 baseline。
- **BEVDet / BEVPoolv2**：工程优化派，把 BEV pooling 的算力从瓶颈里挪开。
- **BEVFusion**：BEV 空间里融合相机与 LiDAR，比图像域融合更自然。

!!! warning "BEV 的隐性成本"
    BEV 分辨率翻倍，pooling 的访存开销约按 $O(HW \cdot D)$ 增长。200×200 的 BEV 在 Orin 上往往就是实时性的天花板，这也是大量论文在做稀疏 BEV / 查询式 BEV 的原因。

### 5.3 3D 检测头与损失

CenterPoint 类方法把目标表示为 BEV 热力图峰值 + 尺寸/朝向/速度回归。热力图用 Gaussian Focal Loss：

\[
L_{\text{heat}} = -\frac{1}{N}\sum_{c}\sum_{i,j}
\begin{cases}
(1-\hat{p}_{ij})^{\alpha}\log \hat{p}_{ij}, & p_{ij}=1\\
(1-p_{ij})^{\beta}\,\hat{p}_{ij}^{\alpha}\log(1-\hat{p}_{ij}), & \text{otherwise}
\end{cases}
\]

朝向回归通常用 bin-based（12 或 24 个 bin 分类 + 残差回归）来绕开 $\pi$ 周期性，再用 $\sin/\cos$ 双通道回归做平滑。

### 5.4 占用栅格（Occupancy）

Occupancy 把空间离散成体素并预测每格占用概率，天然处理不规则障碍（锥桶、掉落物、施工围挡），是"重感知"路线的事实标准：

- 监督来源：LiDAR 点云的 ray-based 免费标注（射线穿过的体素记为 free，端点记为 occupied）；
- 代表工作：OccNet、SurroundOcc、TPVFormer（三视角平面表示，省显存）、OccFormer；
- 评测：只在前景/背景的可见区域算 mIoU，否则空体素会把指标刷到 90% 以上而无意义。

## 6. 多传感器融合

### 6.1 融合层级

| 层级 | 做法 | 优点 | 缺点 |
| --- | --- | --- | --- |
| 前融合（数据级） | 原始点云上色、点级特征拼接 | 信息损失最小 | 对时空对齐极敏感，标定误差直接变噪点 |
| 特征融合（中期） | BEV/视锥空间拼接或 Cross-Attention | 精度与时序友好的折中，主流方案 | 结构复杂，部署成本高 |
| 后融合（目标级） | 各自出目标再做关联与滤波 | 模块解耦、易量产、可解释 | 信息损失大，遮挡目标难互补 |

### 6.2 数据关联

先做**门限**（马氏距离）再解**匈牙利指派**：

\[
d^2_{ij} = (z_i - H\hat{x}_j)^T S_j^{-1} (z_i - H\hat{x}_j) \le \chi^2_{n,\alpha}
\]

```python
import numpy as np
from scipy.optimize import linear_sum_assignment

def associate(det_boxes, trk_boxes, gate=4.0):
    """按中心距离做门限，再匈牙利指派；返回匹配对与未匹配项"""
    d = np.linalg.norm(det_boxes[:, None, :2] - trk_boxes[None, :, :2], axis=-1)
    cost = d.copy()
    cost[d > gate] = 1e6                       # 超出门限禁止匹配
    rows, cols = linear_sum_assignment(cost)
    ok = d[rows, cols] <= gate
    matches = list(zip(rows[ok], cols[ok]))
    un_det = [i for i in range(len(det_boxes)) if i not in set(rows[ok])]
    un_trk = [j for j in range(len(trk_boxes)) if j not in set(cols[ok])]
    return matches, un_det, un_trk
```

### 6.3 跟踪

匀速模型 Kalman 滤波是标配，预测步：

\[
\hat{x}_{k|k-1} = F\hat{x}_{k-1|k-1},\qquad
P_{k|k-1} = FP_{k-1|k-1}F^T + Q
\]

- **SORT / ByteTrack**：仅用运动信息，ByteTrack 的两阶段关联（低分框也参与二次关联）在遮挡场景收益明显；
- **DeepSORT / StrongSORT**：加外观 ReID 特征，跨相机、长时遮挡更稳；
- **IMM**：交互多模型，机动/非机动双模型并行，弯道与变道场景的朝向估计更平滑。

## 7. 端到端与一体化

UniAD、VAD 把感知、预测、规划放进一个可微网络，用规划损失反向约束感知表示。工程上要清醒地看待：

- 收益：减少模块间信息损失与手工接口，长尾场景下全局最优；
- 代价：可解释性差、出问题难定位、训练数据要求高、安全认证路径不清晰；
- 现实做法：**感知仍出结构化结果**（供规划与安全监控），同时并行一条端到端支路做对比与兜底。

## 8. 评测指标

| 任务 | 指标 | 说明 |
| --- | --- | --- |
| 2D 检测 | mAP@0.5、mAP@[.5:.95]、AR | COCO 口径 |
| 3D 检测 | nuScenes mAP + NDS、Waymo LEVEL 1/2、KITTI 3D AP | NDS 是 mAP/ATE/ASE/AOE/AVE 加权；KITTI 分 Easy/Mod/Hard |
| 跟踪 | AMOTA、AMOTP、IDS | nuScenes 用 AMOTA 为主 |
| 分割 | mIoU、F1（车道线） | 前景类单独看，别被背景类稀释 |
| 占用 | 前景/背景 mIoU | 只在可见区域统计 |
| 工程 | P99 时延、帧率、峰值显存/功耗、漏检率 | 安全相关场景的**漏检率**比 mAP 更关键 |

!!! tip "评测要按场景切片"
    只看总 mAP 会掩盖问题。按距离（0–30/30–60/60 m 以上）、光照（白天/黄昏/夜间）、天气（晴/雨/雾）、目标尺寸（行人/轿车/卡车）切片统计，才能定位真实短板。

## 9. 工程落地

### 9.1 数据闭环

1. **影子模式**：量产车静默跑新模型，把与线上模型不一致的片段回传；
2. **主动学习**：按不确定度（熵、时序不一致、占用冲突）挑样本；
3. **自动标注**：离线大模型 + 多帧融合生成伪标签，人工只做校验；
4. **长尾挖掘**：按场景标签检索（施工区、隧道口、夜间逆光、异形车辆）；
5. **badcase 管理**：每个 case 有归属人、复现路径、回归用例，防止修一个坏一个。

### 9.2 部署优化

```text
PyTorch ──▶ ONNX ──▶ TensorRT(FP16/INT8) ──▶ 车端推理
              │
              ├─ 算子替换：DeformableAttn / GridSample / ScatterND 常需自定义插件
              └─ 量化校准：KL 散度或 min-max，注意 BEV pooling 的数值范围
```

- 显存与带宽常是瓶颈：BEV pool、稀疏卷积、Transformer 的 attention 是三大优化点；
- 用固定 shape + CUDA Graph 减少 launch 开销，把 10 路相机的推理批处理成一次前向；
- 时序模型要做**状态缓存复用**，否则每帧重算历史 BEV，帧率直接崩。

### 9.3 时延预算（10 Hz 周期示例）

| 环节 | 预算 | 备注 |
| --- | --- | --- |
| 曝光 + 传输 + 去畸变 | 8 ms | ISP 与快门是硬开销 |
| 骨干网络 + neck | 25 ms | INT8，800×1280 输入 |
| BEV / 检测头 | 20 ms | 含 pooling |
| 后处理 + 跟踪 | 10 ms | NMS、匈牙利、卡尔曼 |
| 结果发出 + 通信 | 5 ms | CAN/以太网 |
| **合计** | **68 ms** | 余量用于抖动与降级 |

## 10. 车路协同感知要点

- **路侧固定标定**：一次标定长期使用，但需周期性校验（振动、风吹、温度会让云台漂移），可用地面标线或已知标定物自动校核；
- **跨设备坐标统一**：路侧结果统一转 ECEF/UTM，再结合车辆自身定位转到 ego 坐标系；
- **时延与补偿**：V2X 端到端时延常在 100–300 ms，必须做位置外推（用目标速度与时间戳补偿），否则融合后目标会"跳"；
- **消息体系**：BSM（车辆状态）、RSI（路侧信息）、RSM（路侧感知目标）、SPAT（信号灯相位），注意不同厂商字段含义差异；
- **融合策略**：路侧更擅长"看得远、看得全"（俯视无遮挡），车端更擅长"看得准"（近距离高精度），权重应按距离与置信度动态分配。

## 11. 常用数据集

| 数据集 | 传感器 | 规模/特点 |
| --- | --- | --- |
| nuScenes | 6 相机 + 32 线 LiDAR + 5 雷达 | 1000 场景，BEV/3D 检测事实标准 |
| Waymo Open | 5 相机 + 3 LiDAR | 精度高、标注质量好，评测严 |
| KITTI | 2 相机 + 64 线 LiDAR | 老基准，仍用于算法对比 |
| Argoverse 2 | 7 相机 + 2 LiDAR | 侧重预测与地图 |
| ONCE | 1 LiDAR + 7 相机 | 国内道路，量大 |
| DAIR-V2X | 车端 + 路侧 | 车路协同研究首选 |
| Cityscapes / BDD100K | 单目 | 2D 分割与检测预训练 |

## 12. 工具链与参考

- **框架**：MMDetection3D、OpenPCDet、BEVFormer/BEVDet（官方实现）、Detectron2
- **部署**：TensorRT、ONNX Runtime、CUDA Graph、NVIDIA DeepStream
- **中间件**：ROS 2 / Autoware、CyberRT
- **论文**：PointPillars (2019)、LSS (2020)、BEVFormer (2022)、CenterPoint (2021)、BEVFusion (2022)、UniAD (2023)、MapTR (2023)、OccFormer (2023)

!!! quote "一句话总结"
    感知的上限由数据与标定决定，下限由工程与算力决定；模型结构只是中间那一层，别把它当成全部。
