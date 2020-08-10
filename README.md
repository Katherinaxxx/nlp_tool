# NLP tool

## Key Features
- BERT API built on [Hugging Face's transformer library](https://huggingface.co/transformers/)
- Data augmentation: add noise to generate more data
- Text Style Transformer
- others：word frequency(histogram&cloud)、split by word

### BERT API
| Methods                            | BERT |
|------------------------------------|--------------|
| Masked Word Prediction             | ✔            |
| Sequence Classification            |              |
| Next Sentence Prediction           |              |
| Question Answering                 |              |
| Masked Word Prediction Finetuning  |              |
 
### Data augmentation
| Methods                            | BERT |
|------------------------------------|--------------|
| replace token with synonyms           |             |
| replace token with prediction of bert |              |

### Text Style Transformer
(to do)

### others
| Methods                            | Example|
|------------------------------------|--------------|
| word frequency             |            |
| split by word            |              |

## usage
### BERT API
#### predict masked word using bert
```sh
from bert_api import bert_tool
bert_tool = bert_tool("bert-base-uncased")
text = '[CLS]You will have more [MASK] than you want.[SEP]'
print(bert_tool.predict_masked(text)) # friends
```