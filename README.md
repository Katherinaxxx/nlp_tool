# NLP toolbox

## Key Features
- BERT API built on [Hugging Face's transformer library](https://huggingface.co/transformers/)
- Data augmentation: add noise to generate more data
- Text Style Transformer (TO DO)
- others：word frequency(histogram&cloud)多线程、split by word (TO DO)

### BERT API
| Methods                            | BERT |
|------------------------------------|--------------|
| Masked Word Prediction             | ✔            |
| Sequence Classification            | ✔            |
| Next Sentence Prediction           |              |
| Question Answering                 |              |
| Masked Word Prediction Finetuning  |              |
 
### Data augmentation
| Methods                            |  |
|------------------------------------|--------------|
| replace token with synonyms           |             |
| replace token with prediction of bert |              |
| delete random token |              |
| random token permutation |              |

### Text Style Transformer
(to do)

### others
| Methods                            | |
|------------------------------------|--------------|
| word frequency             |     ✔       |
| split by word            |       ✔       |
| synonym detection        |              |


## usage
### BERT API
#### predict masked word using bert
```sh
from bert_api import bert_tool
bert_tool = bert_tool("bert-base-uncased")
text = '[CLS]You will have more [MASK] than you want.[SEP]'
print(bert_tool.predict_masked(text)) # friends
```
### others
#### word frequency
```sh
```

#### split by word
```sh
```

#### synonym detection
```sh
```
