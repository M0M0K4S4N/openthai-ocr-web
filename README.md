# OpenThai OCR — in-browser (onnxruntime-web WASM)

> 🤖 **AI-generated project** — โค้ดทั้งหมดเขียนโดย AI agent (Hermes) โดยไม่มี human review
> This project was entirely written by an AI coding agent — no human review.

OCR ภาษาไทย ทำงานในเบราว์เซอร์ 100% client-side — ไม่มี server, ไม่อัปโหลดรูป

## โมเดล

เลือกได้จาก dropdown ในหน้าเว็บ (โหลดครั้งแรกครั้งเดียวต่อโมเดล, cache ไว้ใน tab):

| โมเดล | ไฟล์ | ขนาด | หมายเหตุ |
|---|---|---|---|
| [OpenThai2.0-OCR-Text](https://huggingface.co/iapp/OpenThai2.0-OCR-Text) (v2.0.0) | `models/` | ~4.2MB | พื้นฐาน |
| [OpenThai2.1-OCR-Text-20261118](https://huggingface.co/iapp/OpenThai2.1-OCR-Text-20261118) (v2.1.4) | `models21/` | ~6.5MB | เพิ่ม rec2.onnx + runtime ใหม่ |

จุดต่างของ v2.1 (port จาก `runtime/line_engine.py` v2.1.4 ของ repo ต้นทาง):

- **rec2 re-read** — recognizer ตัวที่สองสำหรับฟอนต์ loopless (ฟอนต์หัวอย่าง Kanit): ถ้าผลอ่านมีตัว Latin ปนในคำไทย (เช่น `lละพลNSะทU`) จะอ่านซ้ำด้วย rec2 แล้วเก็บผลที่ดีกว่า
- **เก็บ cell เลขโดด/ตัวอักษรเดี่ยว** — กล่องที่มีเลขโดด 1 ตัว (หรือ 1 ตัวอักษร) จะไม่ถูกกรองทิ้ง เมื่อความสูงกล่อง ≥ 0.62× ความสูงบรรทัดมัธยฐาน (ตารางเวลา/ใบเสร็จ)
- **แก้จุดทศนิยม** — `88,226,21` → `88,226.21` (รูปแบบ comma ซ้ำจาก OCR)
- ไฟล์ det.onnx / rec.onnx / rec_dict.txt เป็นไบต์เดียวกับ v2.0 (ตรวจด้วย SHA256)

## การทำงาน

- Runtime: onnxruntime-web 1.22.0 (WASM) จาก jsDelivr CDN
- Static site ไม่มี build step — เปิด `index.html` ผ่าน http(s) ได้เลย
- ใช้งาน: เลือกรูป / ลากวาง / วาง Ctrl+V → ได้ข้อความใน textarea (มี auto-orient: deskew, หมุน 90°, กลับหัว 180°)

ข้อจำกัด (port ไม่ครบทุกฟีเจอร์ของ runtime ต้นทาง): การผูกสระ/วรรณยุกต์ลอยเข้ากล่อง, การเรียงลำดับแบบ list และการรวมย่อหน้าข้ามบรรทัดยังไม่รองรับ
