import gradio as gr
from typing import Optional, Tuple, List
from omni_fusion_model import OmniFusionModel  # <-- your standalone class
from omni_fusion_processor import TranslationInput, LANG_MAP


def process_single_input(
    audio: Optional[str],
    image: Optional[str],
    text: str,
    lang: str,
    cot: bool,
    system: OmniFusionModel
) -> str:
    """Process a single translation input."""
    translations = system.translate(
        audio_paths=[audio] if audio else [],
        image_paths=[image] if image else [],
        source_texts=[text],
        target_lang=lang,
        use_cot=cot
    )
    return translations[0] if translations else ""


def process_batch_inputs(
    audio1: Optional[str],
    image1: Optional[str],
    src1: str,
    audio2: Optional[str],
    image2: Optional[str],
    src2: str,
    target_lang: str,
    use_cot: bool,
    system: OmniFusionModel
) -> Tuple[str, str]:
    """Process up to two inputs in a batch."""
    audio_paths: List[Optional[str]] = []
    image_paths: List[Optional[str]] = []
    source_texts: List[str] = []

    if audio1 or image1 or src1:
        audio_paths.append(audio1)
        image_paths.append(image1)
        source_texts.append(src1 or "")

    if audio2 or image2 or src2:
        audio_paths.append(audio2)
        image_paths.append(image2)
        source_texts.append(src2 or "")

    if not source_texts:
        return "No input provided", ""

    translations = system.translate(
        audio_paths=audio_paths,
        image_paths=image_paths,
        source_texts=source_texts,
        target_lang=target_lang,
        use_cot=use_cot
    )

    out1 = translations[0] if len(translations) > 0 else ""
    out2 = translations[1] if len(translations) > 1 else ""
    return out1, out2


def create_gradio_interface(system: OmniFusionModel) -> gr.Blocks:
    """Build the Gradio interface."""
    with gr.Blocks(theme="compact") as demo:
        gr.Markdown("# Multimodal Translation System")
        gr.Markdown("Supports audio transcription and image+text contextual translation.")

        # ------------------ Single Input ------------------
        with gr.Tab("Single Input"):
            with gr.Row():
                with gr.Column():
                    single_audio = gr.Audio(type="filepath", label="Audio (optional)")
                    single_image = gr.Image(type="filepath", label="Image (optional)")
                    single_text = gr.Textbox(label="Source Text")
                    single_lang = gr.Dropdown(
                        choices=list(LANG_MAP.keys()), value="English", label="Target Language"
                    )
                    single_cot = gr.Checkbox(label="Chain of Thought", value=False)
                    single_btn = gr.Button("Translate")

                with gr.Column():
                    single_output = gr.Textbox(label="Translation")

            single_btn.click(
                fn=lambda *args: process_single_input(*args, system=system),
                inputs=[single_audio, single_image, single_text, single_lang, single_cot],
                outputs=single_output,
            )

        # ------------------ Batch Input ------------------
        with gr.Tab("Batch (2 Inputs)"):
            gr.Markdown("Process two inputs at once for faster batching.")

            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Input 1")
                    batch_audio1 = gr.Audio(type="filepath", label="Audio 1")
                    batch_image1 = gr.Image(type="filepath", label="Image 1")
                    batch_text1 = gr.Textbox(label="Source Text 1")

                with gr.Column():
                    gr.Markdown("### Input 2")
                    batch_audio2 = gr.Audio(type="filepath", label="Audio 2")
                    batch_image2 = gr.Image(type="filepath", label="Image 2")
                    batch_text2 = gr.Textbox(label="Source Text 2")

            with gr.Row():
                batch_lang = gr.Dropdown(
                    choices=list(LANG_MAP.keys()), value="English", label="Target Language"
                )
                batch_cot = gr.Checkbox(label="Chain of Thought", value=False)
                batch_btn = gr.Button("Process Batch")

            with gr.Row():
                batch_output1 = gr.Textbox(label="Output 1")
                batch_output2 = gr.Textbox(label="Output 2")

            batch_btn.click(
                fn=lambda *args: process_batch_inputs(*args, system=system),
                inputs=[
                    batch_audio1, batch_image1, batch_text1,
                    batch_audio2, batch_image2, batch_text2,
                    batch_lang, batch_cot
                ],
                outputs=[batch_output1, batch_output2],
            )

    return demo


if __name__ == "__main__":
    system = OmniFusionModel(cache_dir="")
    demo = create_gradio_interface(system)
    demo.launch(server_name="0.0.0.0", server_port=7979)
