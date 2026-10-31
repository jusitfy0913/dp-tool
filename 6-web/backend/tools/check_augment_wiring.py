"""检查前端增强字段 ↔ 后端字段 ↔ JS 引用 三者是否一致。"""

import re
import sys
from pathlib import Path

WEB = Path(__file__).resolve().parent.parent.parent
html = (WEB / "frontend" / "index.html").read_text(encoding="utf-8")
js = (WEB / "frontend" / "js" / "train.js").read_text(encoding="utf-8")
app = (WEB / "backend" / "app.py").read_text(encoding="utf-8")

ids = re.findall(r'id="(aug[A-Za-z]+)"', html)
fields = [i for i in ids if i not in ("augReset", "augSmallDefect")]
print(f"前端增强输入框：{len(fields)} 个")
print(" ", ids)

# JS 里必须能取到每个输入框
missing_js = [i for i in ids if i not in js]
print(f"JS 未引用：{missing_js if missing_js else '无'}")

# 后端模型必须声明全部 15 个字段
backend_fields = ["hsv_h", "hsv_s", "hsv_v", "degrees", "translate", "scale",
                  "shear", "perspective", "flipud", "fliplr", "bgr", "mosaic",
                  "mixup", "copy_paste", "close_mosaic"]
missing_backend = [f for f in backend_fields if f"{f}:" not in app]
print(f"后端缺失字段：{missing_backend if missing_backend else '无'}")

# 命令行 flag 必须在 launcher 里注册
launcher = (WEB.parent / "1-python-yolo" / "launcher" / "tf_yolo_launcher.py").read_text(encoding="utf-8")
flags = ["--hsv-h", "--hsv-s", "--hsv-v", "--degrees", "--translate", "--scale",
         "--shear", "--perspective", "--flipud", "--fliplr", "--bgr", "--mosaic",
         "--mixup", "--copy-paste", "--close-mosaic"]
missing_flags = [f for f in flags if f'"{f}"' not in launcher]
print(f"launcher 未注册：{missing_flags if missing_flags else '无'}")

ok = not (missing_js or missing_backend or missing_flags) and len(fields) == 15
print("\n结果：", "一致" if ok else "不一致")
sys.exit(0 if ok else 1)
