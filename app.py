import torch
import gradio as gr
from transformers import DonutProcessor, VisionEncoderDecoderModel as DonutModel
from PIL import Image
import argparse
import numpy as np

# VietOCR
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- Donut ----------------
def load_model(path):
    processor = DonutProcessor.from_pretrained(path)
    model = DonutModel.from_pretrained(path).to(device)
    if torch.cuda.is_available():
        model.half()
    model.eval()
    return processor, model

def run_donut(image, question=None, processor=None, model=None, task_prompt=None):
    pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)
    prompt = task_prompt.format(user_input=question) if question else task_prompt

    decoder_input_ids = processor.tokenizer(
        prompt, add_special_tokens=False, return_tensors="pt"
    ).input_ids.to(device)

    outputs = model.generate(
        pixel_values,
        decoder_input_ids=decoder_input_ids,
        max_length=model.decoder.config.max_position_embeddings,
        early_stopping=True,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        use_cache=True,
        num_beams=1,
    )
    result = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    return result

# ---------------- VietOCR ----------------
def run_vietocr(image):
    # Nếu ảnh là numpy (Gradio upload) thì chuyển sang PIL
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image).convert("RGB")

    config = Cfg.load_config_from_name('vgg_transformer')
    config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    detector = Predictor(config)
    text = detector.predict(image)
    return text

# ---------------- Main ----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="docvqa", help="docvqa | receipt | invoice | cord | general")
    parser.add_argument("--pretrained_path", type=str, default="naver-clova-ix/donut-base-finetuned-docvqa")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--url", type=str, default=None)
    parser.add_argument("--sample_img_path", type=str, default=None)
    parser.add_argument("--lang", type=str, default="en", help="en | vie")
    args = parser.parse_args()

    task_name = args.task.lower()

    # Load model tùy theo ngôn ngữ
    if args.lang == "vie":
        processor, model = None, None
    else:
        processor, model = load_model(args.pretrained_path)

    # Gradio function
    if args.lang == "vie":
        def gr_fn(image):
            output = run_vietocr(image)
            return {"text": output}

        inputs = "image"
        examples = None  # bỏ ví dụ mẫu để upload ảnh tùy ý

    elif task_name == "docvqa":
        task_prompt = "<s_docvqa><s_question>{user_input}</s_question><s_answer>"

        def gr_fn(image, question):
            output = run_donut(
                image=image,
                question=question,
                processor=processor,
                model=model,
                task_prompt=task_prompt
            )
            return {"question": question, "answer": output}

        inputs = ["image", "text"]
        examples = None

    elif task_name in ["invoice", "receipt", "cord"]:
        task_prompt = "<s_cord>"

        def gr_fn(image):
            output = run_donut(
                image=image,
                question=None,
                processor=processor,
                model=model,
                task_prompt=task_prompt
            )
            return {"data": output}

        inputs = "image"
        examples = None

    # Gradio demo
    demo = gr.Interface(
        fn=gr_fn,
        inputs=inputs,
        outputs="json",
        title=f"Demo Document Extraction - Task: {task_name}, Lang: {args.lang}",
        examples=examples
    )

    demo.launch(share=True)
