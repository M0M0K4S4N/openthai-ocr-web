# OpenThai2.0-OCR — in-browser (onnxruntime-web WASM)

> 🤖 **AI-generated project** — โค้ดทั้งหมดเขียนโดย AI agent (Hermes) ภายใต้การดูแลของ @M0M0K4S4N
> This project was entirely written by an AI coding agent, with human direction and review.

OCR ภาษาไทย ทำงานในเบราว์เซอร์ 100% client-side — ไม่มี server, ไม่อัปโหลดรูป

- Model: [iapp/OpenThai2.0-OCR-Text](https://huggingface.co/iapp/OpenThai2.0-OCR-Text) (Apache-2.0) — det.onnx + rec.onnx (~4.2MB) โหลดจาก `models/`
- Runtime: onnxruntime-web 1.22.0 (WASM) จาก jsDelivr CDN
- Static site ไม่มี build step — เปิด `index.html` ผ่าน http(s) ได้เลย

ใช้งาน: เลือกรูป / ลากวาง / วาง Ctrl+V → ได้ข้อความใน textarea (มี auto-orient: deskew, หมุน 90°, กลับหัว 180°)
