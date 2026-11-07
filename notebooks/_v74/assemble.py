import base64, pathlib
root = pathlib.Path(__file__).parent
parts = sorted(root.glob("part*.b64"), key=lambda p: int(p.stem[4:]))
data = "".join(p.read_text() for p in parts)
out = root.parent / "llm-gmc.ipynb"
out.write_bytes(base64.b64decode(data))
print("wrote", out, "bytes", out.stat().st_size)
