""" Frontend for Semantic Search over Earnings Call Transcripts """
import json
from pathlib import Path
from typing import Dict, List

import gradio as gr
import weaviate
import weaviate.classes as wvc


with Path('assets/config.json').open(encoding='utf-8') as f:
    PATH_BG_MD = Path(json.load(f)['text-background'])


client = weaviate.connect_to_local(
    port=8080,
    grpc_port=50051,
    timeout=(5, 15)
)
db = client.collections.get('Sentences')


def search(search_query: str, mode: str = 'Vector', n: int = 5) -> List[Dict]:
    """ perform database search """
    if mode == 'Vector':
        response = db.query.near_text(
            query=search_query,
            distance=None,
            filters=wvc.Filter('role').equal('Firm') & wvc.Filter('section').equal('Q&A'),
            return_properties=['title', 'coname', 'fy', 'q', 'speaker', 'text'],
            return_metadata=wvc.MetadataQuery(distance=True),
            rerank=wvc.query.Rerank(
                prop='text',
                query=search_query,
            ),
            include_vector=False,
            limit=n,
        )
    elif mode == 'Keyword':
        response = db.query.bm25(
            query=search_query,
            query_properties=['text'],
            return_properties=['title', 'coname', 'fy', 'q', 'speaker', 'text'],
            return_metadata=wvc.MetadataQuery(score=True),
            include_vector=False,
            limit=n,
        )
    results = []
    for _, obj in enumerate(response.objects, start=1):
        if mode == 'Vector':
            rank = round(1 - obj.metadata.distance, 4)
            #rerank = obj.metadata.rerank_score
        else:
            rank = round(obj.metadata.score, 4)
        results.append({
            'relevance': rank,
            'coname': obj.properties['coname'].strip(),
            'fyq': f"FY{obj.properties['fy']} Q{obj.properties['q']}",
            'speaker': obj.properties['speaker'],
            'text': obj.properties['text'],
        })
    return results


app = gr.Blocks(
    theme=gr.themes.Default(),
    css='#component-0 {max-width: 730px; margin: auto; padding-top: 1.5rem}'
)

with app:
    gr.Markdown(
        """
        # Earnings Call Search Tool
        ### Perform Semantic Search over Sentences in Earnings Conference Calls
        """
    )
    with gr.Tabs() as tabs:
        with gr.TabItem("🔍 Search", id=0):
            search_mode = gr.Radio(
                ["Vector", "Keyword"],
                value='Vector', type="value",
                label="Search Mode",
                info="Choose between vector search (semantic) and keyword search (literal)"
            )
            n_results = gr.Slider(
                1, 100, value=5, step=1,
                label="Number of Results",
                info="Choose between 1 and 100 to limit the number of retrieved sentences"
            )
            with gr.Row():
                text_in = gr.Textbox(lines=2, placeholder="Insert text here", label="Search Query", scale=5)
                search_bt = gr.Button("Search", scale=1)
            res_out = gr.JSON(label="Search Results", value=None, scale=1)
            gr.Examples(
                examples=[
                    ["If we look at the plans for 2018, it is to introduce 650 new products, which is an absolute all- time high."],
                    ["We have been doing kind of an integrated campaign, so it's TV, online, we do the Google Ad Words - all those different elements together."],
                    ["So that turned out to be beneficial for us, and I think, we'll just see how the market and interest rates move over the course of the year,"]
                ],
                label="Vector search examples (click to start search)",
                inputs=[text_in, 'Vector', n_results],
                outputs=[res_out],
                fn=search,
                run_on_click=False,
                cache_examples=False,
            )
            gr.Examples(
                examples=[
                    ["artificial intelligence"],
                    ["sustainability"],
                    ["supply chain risk"]
                ],
                label="Keyword search examples (click to start search)",
                inputs=[text_in, 'Keyword', n_results],
                outputs=[res_out],
                fn=search,
                run_on_click=False,
                cache_examples=False,
            )
        with gr.TabItem("📝 Instructions", id=1):
            gr.Markdown(
                """
                This tool is optimized for semantic search (i.e., retrieving documents that are *semantically* related to a given query) instead of retrieval (i.e., retrieving documents that are relevant to a given query).
                In the latter case, the query and the retrieved documents usually look less similar than in the former case (i.e., queries are rather short while retrieved documents can be lengthy)
                
                1. Select the number of results you want to retrieve using the slider.
                2. Enter your search query in the text box (or select one of the examples).
                3. Click the search button to retrieve the results.        
                """
            )
        with gr.TabItem("📝 Background", id=2):
            gr.Markdown(PATH_BG_MD.read_text(encoding='utf-8'))
    with gr.Accordion("📙 Citation", open=False):
        citation_button = gr.Textbox(
            value='Placeholder',
            label='Copy to cite these results.',
            show_copy_button=True
        )

    search_bt.click(search, inputs=[text_in, search_mode, n_results], outputs=[res_out])


if __name__ == "__main__":
    app.launch()
