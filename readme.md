## Tag-to-Caption Augmentation using Large Language Model

This project aims to generate captions for music using existing tags. We leverage the power of OpenAI's GPT-3.5 Turbo API to generate high-quality and contextually relevant captions based on music tags.

This is a PyTorch implementation of [LP-MusicCaps: LLM-Based Pseudo Music Captioning](#) for multi-modal music representation learning.

> [**LP-MusicCaps: LLM-Based Pseudo Music Captioning**](#)   
> SeungHeon Doh, Keunwoo Choi, Jongpil Lee, Juhan Nam
> To appear ISMIR 2023   


### Current Instructions
```
{
    "writing": {
        "singular":"write a song description sentence including the following single attribute.",
        "plural":"write a song description sentence including the following attributes.",
        },
    "summary": {
        "singular":"write a single sentence that summarize a song with the following single attribute. Don't write artist name or album name.",
        "plural":"write a single sentence that summarize a song with the following attributes. Don't write artist name or album name.",
        },
    "paraphrase": {
        "singular":"write a song description sentence including the following single attribute. paraphraze paraphrasing is acceptable.",
        "plural":"write a song description sentence including the following attributes. paraphraze paraphrasing is acceptable.",
        },
    "prediction_attribute": {
        "singular":"write a song description sentence including the following single attribute.",
        "plural":"write the answer as a python dictionary with new_attribute and description as keys. for new_attribute, write new attributes with high co-occurrence with the following attributes. for description, write a song description sentence including the following attributes and new attributes.",
        }
}
```

### Installation
To run this project locally, follow the steps below:

```
pip install -r requirements.txt
```
Set up your OpenAI API credentials by creating a `.env` file in the root directory. Add the following lines and replace YOUR_API_KEY with your actual API key:

```bash
OPENAI_API_KEY={your_key_in_here}
api_host={0.0.0.0}
api_port={8088}
```


### How to Use
To generate captions using music tags, simply run the following command:

```
python run.py --tags <music_tags>
```
Replace <music_tags> with the tags you want to generate captions for. Separate multiple tags with commas, such as `happy, piano, pop, dynamics`.

### Reproduce ISMIR2023 Paper results 
```
python run_lp_music.py --dataset_type mtat --prompt {writing, summary, paraphrase, prediction_attribute}
```

### License
This project is licensed under the MIT License.

### Acknowledgements
We would like to thank OpenAI for providing the GPT-3.5 Turbo API, which powers this project.

### Contact
For any questions or inquiries, please contact seungheondoh@kaist.ac.kr.