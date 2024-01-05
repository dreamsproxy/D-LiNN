import pyflowchart as pfc

st = pfc.StartNode('Start')
init_op = pfc.OperationNode('Init Network')

gen_sensor_coords = pfc.OperationNode("Create Input sensor coordinates")
create_sensors = pfc.OperationNode("Create sensors")

gen_normal_coords = pfc.OperationNode("Create normal neuron coordinates")
create_normals = pfc.OperationNode("Create neurons")

populate = pfc.OperationNode("populate self.LIFneurons")
init_weight_matrix = pfc.OperationNode("Init Weight Matrix")

init_returns = pfc.OperationNode("Coordinates in dict and json dumps ")

st.connect(init_op)
init_op.connect(gen_sensor_coords)
gen_sensor_coords.connect(create_sensors)
create_sensors.connect(gen_normal_coords)
gen_normal_coords.connect(create_normals)
create_normals.connect(populate)
populate.connect(init_weight_matrix)
init_weight_matrix.connect(init_returns)

fc = pfc.Flowchart(st)
print(fc.flowchart())