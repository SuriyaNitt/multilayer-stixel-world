import numpy as np
import pypcd

def from_array(npc):
    dt = np.dtype([("x", np.float32, (1, )), ("y", np.float32, (1, )), ("z", np.float32, (1, ))])
    pc_data = np.ndarray((npc.shape[0],), dtype=dt)

    for i in range(npc.shape[0]):
        pc_data['x'][i] = npc[i, 0]
        pc_data['y'][i] = npc[i, 1]
        pc_data['z'][i] = npc[i, 2]

    md = {'version': .7,
              'fields': [],
              'size': [],
              'count': [],
              'width': 0,
              'height': 1,
              'viewpoint': [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
              'points': 0,
              'type': [],
              'data': 'binary_compressed'}

    for i in range(npc.shape[1]):
        md['type'].append("F")
        md['size'].append(4)
        md['count'].append(1)


    md['fields'] = ["x", "y", "z"]

    # pc_data = np.array({"x": npc[:, 0],"y": npc[:, 1], "z": npc[:, 2]})

    print("ckpt 0")

    md['width'] = len(npc)
    md['points'] = len(npc)

    print(len(npc))
    print(len(pc_data))
    print(pc_data.shape)
    print(npc.shape)
    print(pc_data['x'].shape)


    pc = pypcd.PointCloud(md, pc_data)
    return pc

if __name__ == "__main__":
    pcl_file = open("pointcloud.txt")

    lines = pcl_file.readlines()

    npc = np.zeros((len(lines), 3))

    for i, line in enumerate(lines):
        xyz = line.split(",")
        npc[i, 0] = float(xyz[2])
        npc[i, 1] = -float(xyz[0])
        npc[i, 2] = -float(xyz[1])


    pc = from_array(npc)

    pc.save_pcd("pointcloud.pcd", compression="binary_compressed")
