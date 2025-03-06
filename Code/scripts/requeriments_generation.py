from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import pandas as pd

# Configuración
GENERATION_MODEL = "../models/Qwant 1.5B Destil"  # Modelo local alternativo
FILTERED_CSV = "../data/swiftkey_informative (sample).csv"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PROMPT_TEMPLATE = """
Analiza los siguientes comentarios de usuarios y genera:
1. Requisitos funcionales claros
2. Requisitos de calidad (no funcionales)
3. Principales problemas identificados

Comentarios:
{comentarios}

Respuesta (estructura en markdown):
### Requisitos Funcionales
- 
### Requisitos de Calidad
- 
### Principales Problemas
- 
"""


def generar_requisitos(generation_model=GENERATION_MODEL, filtered_csv=FILTERED_CSV):
    # Cargar datos filtrados
    comentarios = pd.read_csv(filtered_csv)["Review"].str.cat(sep="\n- ")

    # Cargar modelo de generación
    tokenizer = AutoTokenizer.from_pretrained(generation_model,
                                              padding_side="right",
                                              pad_token="<pad>",
                                              truncation=True)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    model = AutoModelForCausalLM.from_pretrained(generation_model).to(DEVICE)

    # Crear prompt
    prompt = PROMPT_TEMPLATE.format(comentarios=comentarios)

    # Generar texto
    inputs = tokenizer(prompt,
                       return_tensors="pt",
                       padding=True,
                       truncation=True,
                       max_length=1000
                       ).to(DEVICE)

    outputs = model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_length=15000,
        temperature=0.7,
        num_return_sequences=1,
        do_sample=True
    )

    # Decodificar y mostrar resultados
    resultado = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(resultado)


if __name__ == "__main__":
    generar_requisitos(GENERATION_MODEL, FILTERED_CSV)
