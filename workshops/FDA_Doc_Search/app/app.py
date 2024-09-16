import gradio as gr
import logging
import server

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

theme = gr.themes.Base()

with gr.Blocks(theme=theme, css="www/main.css") as demo:
    with gr.Row(elem_classes="header-bar"):
        gr.Markdown(
            "# Drugs@FDA Search"
        )

    with gr.Row():
        with gr.Column(scale=1, visible=False) as nav:
            with gr.Row():
                category_filter = gr.CheckboxGroup(show_label=False, choices=None)
                brand_name_filter = gr.CheckboxGroup(show_label=False, choices=None)
                generic_name_filter = gr.CheckboxGroup(show_label=False, choices=None)
                manufacturer_name_filter = gr.CheckboxGroup(
                    show_label=False, choices=None
                )
        with gr.Column(scale=12):
            with gr.Row():
                query_input = gr.Textbox(
                    placeholder="🔎", show_label=False, interactive=True
                )
            with gr.Row():
                query_examples = gr.Examples(
                    [
                        ["What are the approved indications for Mounjaro?"],
                        ["What are some FDA-approved treatments for non-small cell lung cancer?"],
                        ["What were the Phase III trial sites for Trulicity?"]
                    ],
                    [query_input],
                    fn=server.submit_query,
                )
            with gr.Row():
                search_button = gr.Button("Search", scale=0)
                clear_button = gr.Button("Clear", scale=0)
            with gr.Row():
                generate_results = gr.HTML(
                    visible=True,
                    value=None,
                    show_label=False,
                )
            with gr.Row():
                query_results = gr.HTML(visible=True, value=None, show_label=False)

    query_input.submit(
        fn=server.submit_query,
        inputs=query_input,
        outputs=[
            query_results,
            generate_results,
            category_filter,
            brand_name_filter,
            generic_name_filter,
            manufacturer_name_filter,
            nav
        ],
    )

    search_button.click(
        fn=server.submit_query,
        inputs=query_input,
        outputs=[
            query_results,
            generate_results,
            category_filter,
            brand_name_filter,
            generic_name_filter,
            manufacturer_name_filter,
            nav
        ],
    )
    clear_button.click(
        fn=lambda: [
            gr.Textbox(
                placeholder="🔎", value=None, show_label=False, interactive=True
            ),
            gr.HTML(visible=True, value=None, show_label=False),
            gr.HTML(visible=True, value=None, show_label=False),
            gr.CheckboxGroup(show_label=False, choices=None),
            gr.CheckboxGroup(show_label=False, choices=None),
            gr.CheckboxGroup(show_label=False, choices=None),
            gr.CheckboxGroup(show_label=False, choices=None),
            gr.Column(visible=False)
        ],
        inputs=None,
        outputs=[
            query_input,
            query_results,
            generate_results,
            category_filter,
            brand_name_filter,
            generic_name_filter,
            manufacturer_name_filter,
            nav
        ],
    )

    category_filter.input(
        fn=server.filter_query,
        inputs=[
            query_input,
            category_filter,
            brand_name_filter,
            generic_name_filter,
            manufacturer_name_filter,
        ],
        outputs=[query_input],
    )

    brand_name_filter.input(
        fn=server.filter_query,
        inputs=[
            query_input,
            category_filter,
            brand_name_filter,
            generic_name_filter,
            manufacturer_name_filter,
        ],
        outputs=[query_input],
    )

    generic_name_filter.input(
        fn=server.filter_query,
        inputs=[
            query_input,
            category_filter,
            brand_name_filter,
            generic_name_filter,
            manufacturer_name_filter,
        ],
        outputs=[query_input],
    )

    manufacturer_name_filter.input(
        fn=server.filter_query,
        inputs=[
            query_input,
            category_filter,
            brand_name_filter,
            generic_name_filter,
            manufacturer_name_filter,
        ],
        outputs=[query_input],
    )

demo.queue()
demo.launch(allowed_paths=["www"])
