# Tag-to-Caption Augmentation using Large Language Model

This project aims to generate captions for music using existing tags. We leverage the power of OpenAI's GPT-3.5 Turbo API to generate high-quality and contextually relevant captions based on music tags.

This is a implementation of [LP-MusicCaps: LLM-Based Pseudo Music Captioning](#) for music tag-to-caption. If you wanna contents-based-music captioning, please check this [update-soon](#) 

> [**LP-MusicCaps: LLM-Based Pseudo Music Captioning**](#)   
> SeungHeon Doh, Keunwoo Choi, Jongpil Lee, Juhan Nam   
> To appear ISMIR 2023   

## Pesudo Caption Dataset

We released a 2.2M (total) pesudo caption dataset using the MSD-ECALS subset, Magnatagatune, and Music Caps dataset. Download from,

- **LP-MusicCaps-MSD**, [HuggingFace](#), [Zenodo](#).
- **LP-MusicCaps-MTT**, [HuggingFace](#), [Zenodo](#).
- **LP-MusicCaps-MC**, [HuggingFace](#), [Zenodo](#).

```python
from datasets import load_dataset
dataset = load_dataset("seungheondoh/LP-MusicCaps-MSD")
dataset['test'][0]
```

```yaml
{
    'track_id': 'TREADPD128F933E680',
    'title': 'Written in the stars',
    'artist_name': "Blackmore's Night",
    'release': 'Fires at midnight',
    'year': 2001,
    'tag': [
        'dramatic',
        'guitar virtuoso',
        'heavy metal',
        'indulgent',
        'pop rock',
        'atmospheric',
        'hard rock',
        'refined',
        'folk'
        ],
    'writing': 'This song showcases the undeniable talent of a guitar virtuoso, seamlessly blending the refined elements of pop rock and folk with the atmospheric and dramatic sounds of heavy metal and hard rock, resulting in an indulgent and unforgettable musical experience.',
    'summary': "This song showcases a guitar virtuoso's refined and atmospheric pop rock sound, with elements of dramatic heavy metal, folk, and indulgent hard rock.",
    'paraphrase': 'This song showcases the refined playing of a guitar virtuoso at the forefront of intricate pop rock arrangements, with atmospheric and dramatic elements that draw from heavy metal, folk, and indulgent hard rock influences.',
    'attribute_prediction': 'This pop rock ballad is a showcase for the guitar virtuoso\\s refined playing style, blending atmospheric and heavy metal sounds into a unique folk rock sound. With indulgent solos and dramatic duets, the track creates a hard rock energy that is both mellow and upbeat, introspective and soulful.'
}
```


## Installation
To run this project locally, follow the steps below:

```
pip install -r requirements.txt
# our exp version (date): openai-0.27.8, python-dotenv-1.0.0  (2023.04 ~ 2023.05)
```
Set up your OpenAI API credentials by creating a `.env` file in the root directory. Add the following lines and replace YOUR_API_KEY with your actual API key:

```bash
OPENAI_API_KEY=your_key_in_here
api_host=0.0.0.0
api_port=8088
```

## Quick Start
To generate captions using music tags, simply run the following command:

```bash
python run.py --prompt {writing, summary, paraphrase, attribute_prediction} --tags <music_tags>
```
Replace <music_tags> with the tags you want to generate captions for. Separate multiple tags with commas, such as `beatbox, finger snipping, male voice, amateur recording, medium tempo`.

tag_to_caption generation `writing` results:
```
query: 
write a song description sentence including the following attributes
beatbox, finger snipping, male voice, amateur recording, medium tempo
----------
results: 
"Experience the raw and authentic energy of an amateur recording as mesmerizing beatbox rhythms intertwine with catchy finger snipping, while a soulful male voice delivers heartfelt lyrics on a medium tempo track."
```

tag_to_caption generation `summary` results:
```
query: 
write a single sentence that summarize a song with the following attributes. Don't write artist name or album name.
beatbox, finger snipping, male voice, amateur recording, medium tempo
----------
results: 
"Experience the raw and authentic energy of an amateur recording as mesmerizing beatbox rhythms intertwine with catchy finger snipping, while a soulful male voice delivers heartfelt lyrics on a medium tempo track."
```

tag_to_caption generation `paraphrase` results:
```
query: 
write a song description sentence including the following attributes. paraphraze paraphrasing is acceptable.
beatbox, finger snipping, male voice, amateur recording, medium tempo
----------
results: 
"Experience the raw and authentic energy of an amateur recording as mesmerizing beatbox rhythms intertwine with catchy finger snipping, while a soulful male voice delivers heartfelt lyrics on a medium tempo track."
```

tag_to_caption generation `attribute_prediction` results:
```
query: 
write the answer as a python dictionary with new_attribute and description as keys. for new_attribute, write new attributes with high co-occurrence with the following attributes. for description, write a song description sentence including the following attributes and new attributes."
beatbox, finger snipping, male voice, amateur recording, medium tempo
----------
results: 
"Experience the raw and authentic energy of an amateur recording as mesmerizing beatbox rhythms intertwine with catchy finger snipping, while a soulful male voice delivers heartfelt lyrics on a medium tempo track."
```

## Caption Generation from Existing Tag Dataset
We have prepared a simple example of 26 `tag-caption` pairs from [Music Caps](https://huggingface.co/datasets/google/MusicCaps). if you want to reproduce the 2.2 Miliion examples from the paper, download annotation data & track split from [MSD-ECALS](https://zenodo.org/record/7107130), [Magnatagatune-MTT](https://github.com/seungheondoh/msu-benchmark). 

```bash
python run_lp_music.py --dataset_type {musiccaps, msd, mtat} --prompt {writing, summary, paraphrase, attribute_prediction}
```

## Reproduce ISMIR 2023 paper results

```bash
python eval_lp_music.py
```

### License
This project is licensed under the MIT License.

### Acknowledgements
We would like to thank OpenAI for providing the GPT-3.5 Turbo API, which powers this project.

### Contact
For any questions or inquiries, please contact seungheondoh@kaist.ac.kr.