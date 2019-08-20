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

def create_cube(distance):

    z_len = 20
    x_len = 20
    y_len = 20

    cube = np.zeros((z_len * x_len * y_len, 3))

    for z in range(z_len):
        for y in range(y_len):
            for x in range(x_len):
                cube[x + y * x_len + z * (x_len * y_len), 0] = (z+1) * 0.1
                cube[x + y * x_len + z * (x_len * y_len), 1] = x * 0.1
                cube[x + y * x_len + z * (x_len * y_len), 2] = y * 0.1

    return cube

if __name__ == '__main__':

    cube = create_cube(10)
    cube_pc = from_array(cube)

    cube_pc.save_pcd("cube.pcd", compression="binary_compressed")
