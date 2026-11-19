import struct, sys, re, collections
T = {0:"F32",1:"F16",2:"Q4_0",3:"Q4_1",6:"Q5_0",7:"Q5_1",8:"Q8_0",9:"Q8_1",10:"Q2_K",11:"Q3_K",12:"Q4_K",13:"Q5_K",14:"Q6_K",15:"Q8_K",16:"IQ2_XXS",17:"IQ2_XS",18:"IQ3_XXS",19:"IQ1_S",20:"IQ4_NL",21:"IQ3_S",22:"IQ2_S",23:"IQ4_XS",24:"I8",25:"I16",26:"I32",27:"I64",28:"F64",29:"IQ1_M",30:"BF16"}
f = open(sys.argv[1], "rb")
def rd(fmt): return struct.unpack("<"+fmt, f.read(struct.calcsize("<"+fmt)))[0]
def rs(): n = rd("Q"); return f.read(n).decode("utf-8", "replace")
def rv(t):
    if t in (0,1): return rd("Bb"[t] if False else ("B" if t==0 else "b"))
    sc = {2:"H",3:"h",4:"I",5:"i",6:"f",7:"?",10:"Q",11:"q",12:"d"}
    if t in sc: return rd(sc[t])
    if t == 8: return rs()
    if t == 9:
        at = rd("I"); n = rd("Q"); return [rv(at) for _ in range(n)]
    raise ValueError(t)
assert f.read(4) == b"GGUF"; ver = rd("I"); nt = rd("Q"); nkv = rd("Q")
meta = {}
for _ in range(nkv):
    k = rs(); t = rd("I"); meta[k] = rv(t)
for k, v in meta.items():
    if k.startswith("tokenizer.ggml.") and isinstance(v, list): v = f"[{len(v)} items]"
    elif isinstance(v, list) and len(v) > 24: v = f"{v[:24]}... ({len(v)} items)"
    print("META", k, "=", v)
tensors = []
for _ in range(nt):
    name = rs(); nd = rd("I"); dims = [rd("Q") for _ in range(nd)]; typ = rd("I"); off = rd("Q")
    tensors.append((name, dims, T.get(typ, str(typ))))
groups = collections.defaultdict(list)
for name, dims, typ in tensors:
    m = re.match(r"blk\.(\d+)\.(.*)", name)
    key = m.group(2) if m else name
    groups[(key, typ, tuple(dims))].append(int(m.group(1)) if m else -1)
print("TENSORS", nt)
for (key, typ, dims), layers in sorted(groups.items()):
    ls = sorted(layers)
    print(f"  {key:40s} {typ:6s} {list(dims)}  layers={ls if len(ls) <= 12 else str(len(ls)) + ' layers'}")
