from pathlib import Path

TFLITE_PATH = Path(
    r"C:\Users\M S I\PycharmProjects\KWS\Models\CNN1D_keyword_spotting_int8.tflite"
)
HEADER_PATH = Path(
    r"C:\Users\M S I\PycharmProjects\KWS\esp32\model_data_int8_onedcnn.h"
)
HEADER_PATH.parent.mkdir(parents=True, exist_ok=True)

ARRAY_NAME = "g_model"

data = TFLITE_PATH.read_bytes()
length = len(data)

lines = []
lines.append("// Auto-generated from " + TFLITE_PATH.name)
lines.append("// Do not edit manually.")
lines.append("")
lines.append("#ifndef MODEL_DATA_INT8_H_")
lines.append("#define MODEL_DATA_INT8_H_")
lines.append("")
lines.append(f"const unsigned char {ARRAY_NAME}[] = {{")

# 12 bytes per line
for i in range(0, length, 12):
    chunk = data[i:i+12]
    hex_bytes = ", ".join(f"0x{b:02x}" for b in chunk)
    lines.append(f"  {hex_bytes},")

lines.append("};")
lines.append("")
lines.append(f"const unsigned int {ARRAY_NAME}_len = {length};")
lines.append("")
lines.append("#endif  // MODEL_DATA_INT8_H_")

HEADER_PATH.write_text("\n".join(lines), encoding="utf-8")

print(f"TFLite file  : {TFLITE_PATH}")
print(f"Header file  : {HEADER_PATH}")
print(f"Array name   : {ARRAY_NAME}")
print(f"Array length : {length} bytes ({length/1024:.2f} KB)")