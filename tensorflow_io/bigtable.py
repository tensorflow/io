from tensorflow_io.python.ops.bigtable_dataset_ops import BigtableDataset

print("started")
x = BigtableDataset("test-project", "test-instance", "t1", ["cf1:c1", "cf1:c2"])
print("dataset created")

for r in x:
  print(r.numpy())

print("finished")
