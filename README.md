---
language:
- pt
- en
license: mit
library_name: peft
tags:
- LLM
- Portuguese
- Bode
- Alpaca
- Llama 2
- Q&A
metrics:
- accuracy
- f1
- precision
- recall
pipeline_tag: text-generation
inference: false
model-index:
- name: bode-7b-alpaca-pt-br
  results:
  - task:
      type: text-generation
      name: Text Generation
    dataset:
      name: ENEM Challenge (No Images)
      type: eduagarcia/enem_challenge
      split: train
      args:
        num_few_shot: 3
    metrics:
    - type: acc
      value: 34.36
      name: accuracy
    source:
      url: https://huggingface.co/spaces/eduagarcia/open_pt_llm_leaderboard?query=recogna-nlp/bode-7b-alpaca-pt-br
      name: Open Portuguese LLM Leaderboard
  - task:
      type: text-generation
      name: Text Generation
    dataset:
      name: BLUEX (No Images)
      type: eduagarcia-temp/BLUEX_without_images
      split: train
      args:
        num_few_shot: 3
    metrics:
    - type: acc
      value: 28.93
      name: accuracy
    source:
      url: https://huggingface.co/spaces/eduagarcia/open_pt_llm_leaderboard?query=recogna-nlp/bode-7b-alpaca-pt-br
      name: Open Portuguese LLM Leaderboard
  - task:
      type: text-generation
      name: Text Generation
    dataset:
      name: OAB Exams
      type: eduagarcia/oab_exams
      split: train
      args:
        num_few_shot: 3
    metrics:
    - type: acc
      value: 30.84
      name: accuracy
    source:
      url: https://huggingface.co/spaces/eduagarcia/open_pt_llm_leaderboard?query=recogna-nlp/bode-7b-alpaca-pt-br
      name: Open Portuguese LLM Leaderboard
  - task:
      type: text-generation
      name: Text Generation
    dataset:
      name: Assin2 RTE
      type: assin2
      split: test
      args:
        num_few_shot: 15
    metrics:
    - type: f1_macro
      value: 79.83
      name: f1-macro
    - type: pearson
      value: 43.47
      name: pearson
    source:
      url: https://huggingface.co/spaces/eduagarcia/open_pt_llm_leaderboard?query=recogna-nlp/bode-7b-alpaca-pt-br
      name: Open Portuguese LLM Leaderboard
  - task:
      type: text-generation
      name: Text Generation
    dataset:
      name: FaQuAD NLI
      type: ruanchaves/faquad-nli
      split: test
      args:
        num_few_shot: 15
    metrics:
    - type: f1_macro
      value: 67.45
      name: f1-macro
    source:
      url: https://huggingface.co/spaces/eduagarcia/open_pt_llm_leaderboard?query=recogna-nlp/bode-7b-alpaca-pt-br
      name: Open Portuguese LLM Leaderboard
  - task:
      type: text-generation
      name: Text Generation
    dataset:
      name: HateBR Binary
      type: eduagarcia/portuguese_benchmark
      split: test
      args:
        num_few_shot: 25
    metrics:
    - type: f1_macro
      value: 85.06
      name: f1-macro
    - type: f1_macro
      value: 65.73
      name: f1-macro
    source:
      url: https://huggingface.co/spaces/eduagarcia/open_pt_llm_leaderboard?query=recogna-nlp/bode-7b-alpaca-pt-br
      name: Open Portuguese LLM Leaderboard
  - task:
      type: text-generation
      name: Text Generation
    dataset:
      name: tweetSentBR
      type: eduagarcia-temp/tweetsentbr
      split: test
      args:
        num_few_shot: 25
    metrics:
    - type: f1_macro
      value: 43.25
      name: f1-macro
    source:
      url: https://huggingface.co/spaces/eduagarcia/open_pt_llm_leaderboard?query=recogna-nlp/bode-7b-alpaca-pt-br
      name: Open Portuguese LLM Leaderboard
---

# BODE

<!--- PROJECT LOGO -->
<p align="center">
  <img src="https://huggingface.co/recogna-nlp/bode-7b-alpaca-pt-br/resolve/main/Logo_Bode_LLM_Circle.png" alt="Bode Logo" width="400" style="margin-left:'auto' margin-right:'auto' display:'block'"/>
</p>

Bode é um modelo de linguagem (LLM) para o português desenvolvido a partir do modelo Llama 2 por meio de fine-tuning no dataset Alpaca, traduzido para o português pelos autores do Cabrita. Este modelo é projetado para tarefas de processamento de linguagem natural em português, como geração de texto, tradução automática, resumo de texto e muito mais. 
O objetivo do desenvolvimento do BODE é suprir a escassez de LLMs para a língua portuguesa. Modelos clássicos, como o próprio LLaMa, são capazes de responder prompts em português, mas estão sujeitos a muitos erros de gramática e, por vezes, geram respostas na língua inglesa. Ainda há poucos modelos em português disponíveis para uso gratuito e, segundo nosso conhecimento, não modelos disponíveis com 13b de parâmetros ou mais treinados especificamente com dados em português. 

Acesse o [artigo](https://arxiv.org/abs/2401.02909) para mais informações sobre o Bode. 


## Detalhes do Modelo

- **Modelo Base:** Llama 2
- **Dataset de Treinamento:** Alpaca
- **Idioma:** Português

## Versões disponíveis

| Quantidade de parâmetros       | PEFT | Modelo                                                                                      | 
| :-:                            | :-:  |  :-:                                                                                         | 
| 7b                             | &check; | [recogna-nlp/bode-7b-alpaca-pt-br](https://huggingface.co/recogna-nlp/bode-7b-alpaca-pt-br)  |
| 13b                            | &check; | [recogna-nlp/bode-13b-alpaca-pt-br](https://huggingface.co/recogna-nlp/bode-13b-alpaca-pt-br)|
| 7b                             |    | [recogna-nlp/bode-7b-alpaca-pt-br-no-peft](https://huggingface.co/recogna-nlp/bode-7b-alpaca-pt-br-no-peft)  |
| 13b                             |    | [recogna-nlp/bode-13b-alpaca-pt-br-no-peft](https://huggingface.co/recogna-nlp/bode-13b-alpaca-pt-br-no-peft)  |
| 7b-gguf                             |    | [recogna-nlp/bode-7b-alpaca-pt-br-gguf](https://huggingface.co/recogna-nlp/bode-7b-alpaca-pt-br-gguf)  |
| 13b-gguf                             |    | [recogna-nlp/bode-13b-alpaca-pt-br-gguf](https://huggingface.co/recogna-nlp/bode-13b-alpaca-pt-br-gguf)  |

## Uso

Recomendamos fortemente que utilizem o Kaggle com GPU. Você pode usar o Bode facilmente com a biblioteca Transformers do HuggingFace. Entretanto, é necessário ter a autorização de acesso ao LLaMa 2. Também disponibilizamos um jupyter notebook no Google Colab, [clique aqui](https://colab.research.google.com/drive/1uqVCED2wNPXIa7On0OAnghJNr13PUB5o?usp=sharing) para acessar.

Abaixo, colocamos um exemplo simples de como carregar o modelo e gerar texto:

```python

# Downloads necessários
!pip install transformers
!pip install einops accelerate bitsandbytes
!pip install sentence_transformers
!pip install git+https://github.com/huggingface/peft.git

from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from peft import PeftModel, PeftConfig

llm_model = 'recogna-nlp/bode-7b-alpaca-pt-br'
hf_auth = 'HF_ACCESS_KEY'
config = PeftConfig.from_pretrained(llm_model)
model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, trust_remote_code=True, return_dict=True, load_in_8bit=True, device_map='auto', token=hf_auth)
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path, token=hf_auth)
model = PeftModel.from_pretrained(model, llm_model) # Caso ocorra o seguinte erro: "ValueError: We need an `offload_dir`... Você deve acrescentar o parâmetro: offload_folder="./offload_dir".
model.eval()

#Testando geração de texto
def generate_prompt(instruction, input=None):
    if input:
        return f"""Abaixo está uma instrução que descreve uma tarefa, juntamente com uma entrada que fornece mais contexto. Escreva uma resposta que complete adequadamente o pedido.

### Instrução:
{instruction}

### Entrada:
{input}

### Resposta:"""
    else:
        return f"""Abaixo está uma instrução que descreve uma tarefa. Escreva uma resposta que complete adequadamente o pedido.

### Instrução:
{instruction}

### Resposta:"""
     
generation_config = GenerationConfig(
    temperature=0.2,
    top_p=0.75,
    num_beams=2,
    do_sample=True
)

def evaluate(instruction, input=None):
    prompt = generate_prompt(instruction, input)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].cuda()
    generation_output = model.generate(
        input_ids=input_ids,
        generation_config=generation_config,
        return_dict_in_generate=True,
        output_scores=True,
        max_length=300
    )
    for s in generation_output.sequences:
        output = tokenizer.decode(s)
        print("Resposta:", output.split("### Resposta:")[1].strip())

evaluate("Responda com detalhes: O que é um bode?")
#Exemplo de resposta obtida (pode variar devido a temperatura): Um bode é um animal do gênero Bubalus, da família Bovidae, que é um membro da ordem Artiodactyla. Os bodes são mamíferos herbívoros que são nativos da Ásia, África e Europa. Eles são conhecidos por seus cornos, que podem ser usados para defesa e como uma ferramenta.
```

## Treinamento e Dados

O modelo Bode foi treinado por fine-tuning a partir do modelo Llama 2 usando o dataset Alpaca em português, que consiste em um Instruction-based dataset. O treinamento foi realizado no Supercomputador Santos Dumont do LNCC, através do projeto da Fundunesp 2019/00697-8.

## Citação

Se você deseja utilizar o Bode em sua pesquisa, pode citar este [artigo](https://arxiv.org/abs/2401.02909) que discute o modelo com mais detalhes. Cite-o da seguinte maneira:


```
    @misc{bode2024,
      title={Introducing Bode: A Fine-Tuned Large Language Model for Portuguese Prompt-Based Task}, 
      author={Gabriel Lino Garcia and Pedro Henrique Paiola and Luis Henrique Morelli and Giovani Candido and Arnaldo Cândido Júnior and Danilo Samuel Jodas and Luis C. S. Afonso and Ivan Rizzo Guilherme and Bruno Elias Penteado and João Paulo Papa},
      year={2024},
      eprint={2401.02909},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## Contribuições

Contribuições para a melhoria deste modelo são bem-vindas. Sinta-se à vontade para abrir problemas e solicitações pull.

## Agradecimentos

Agradecemos ao Laboratório Nacional de Computação Científica (LNCC/MCTI, Brasil) por prover os recursos de CAD do supercomputador SDumont.
# [Open Portuguese LLM Leaderboard Evaluation Results](https://huggingface.co/spaces/eduagarcia/open_pt_llm_leaderboard)
Detailed results can be found [here](https://huggingface.co/datasets/eduagarcia-temp/llm_pt_leaderboard_raw_results/tree/main/recogna-nlp/bode-7b-alpaca-pt-br)

|          Metric          |  Value  |
|--------------------------|---------|
|Average                   |**53.21**|
|ENEM Challenge (No Images)|    34.36|
|BLUEX (No Images)         |    28.93|
|OAB Exams                 |    30.84|
|Assin2 RTE                |    79.83|
|Assin2 STS                |    43.47|
|FaQuAD NLI                |    67.45|
|HateBR Binary             |    85.06|
|PT Hate Speech Binary     |    65.73|
|tweetSentBR               |    43.25|

