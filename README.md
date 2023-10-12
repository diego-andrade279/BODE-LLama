---
license: mit
language:
- pt
- en
metrics:
- accuracy
- f1
- precision
- recall
pipeline_tag: text-generation
tags:
- LLM
- Portuguese
- Bode
- Alpaca
- Llama 2
- Q&A
---

<img src="https://huggingface.co/recogna-nlp/bode-7b-alpaca-pt-br/blob/main/Logo_Bode_LLM.jpg" width="200" >


# BODE

Bode é um modelo de linguagem (LLM) para o português desenvolvido a partir do modelo Llama 2 por meio de fine-tuning no dataset Alpaca. Este modelo é projetado para tarefas de processamento de linguagem natural em português, como geração de texto, tradução automática, resumo de texto e muito mais.

## Detalhes do Modelo

- **Modelo Base:** Llama 2
- **Dataset de Treinamento:** Alpaca
- **Idioma:** Português

## Uso

Você pode usar o Bode facilmente com a biblioteca Transformers do HuggingFace. Aqui está um exemplo simples de como carregar o modelo e gerar texto:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "recogna-nlp/bode-7b-alpaca-pt-br"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

input_text = "Bode é um modelo de linguagem muito eficiente para o português."
input_ids = tokenizer.encode(input_text, return_tensors="pt")
output = model.generate(input_ids, max_length=50, num_return_sequences=1)

generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)
```

## Treinamento e Dados

O modelo Bode foi treinado por fine-tuning a partir do modelo Llama 2 usando o dataset Alpaca em português. O dataset Alpaca contém X milhões de amostras de texto em português, coletadas de [fontes] e pré-processadas para o treinamento do modelo. O treinamento foi realizado com os seguintes hiperparâmetros: [inserir hiperparâmetros].

## Contribuições

Contribuições para a melhoria deste modelo são bem-vindas. Sinta-se à vontade para abrir problemas e solicitações pull.

## Agradecimentos e Considerações

Agracimentos aqui...

## Contato

Para perguntas, sugestões ou colaborações, entre em contato com [recogna-nlp@gmail.com].

## Citação

Se você usar o modelo de linguagem Bode em sua pesquisa ou projeto, por favor, cite-o da seguinte maneira: