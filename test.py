from dsets import LunaDataset

ds = LunaDataset()

pos = sum(
    1 for x in ds.candidateInfo_list
    if x.isNodule_bool
)

neg = len(ds) - pos

print("total =", len(ds))
print("pos =", pos)
print("neg =", neg)
print("pos ratio =", pos / len(ds))