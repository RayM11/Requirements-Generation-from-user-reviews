import pandas as pd
import argparse
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

PROMPT_TEMPLATE = """
# Application Context
{app_description}

# User Comments to Analyze
{comments}

# Task
Analyze the comments and extract clear software requirements. Follow these guidelines:

1. **Requirement Types**:
   - **Functional**: Describe what the system should do.
   - **Non-Functional**: Describe how the system should perform (e.g., performance, security, usability).

2. **Requirement Format**:
   - Use the structure: "The system shall [action] [condition/criteria]."
   - Be specific and include measurable criteria where possible.

3. **Examples**:
   - Functional: "The system shall allow users to export reports in PDF format."
   - Non-Functional: "The system shall load all dashboard data in under 2 seconds."
   
4. **Quality Criteria**:
- Atomicity: 1 requirement per feature
- Specificity: Include verifiable metrics/criteria
- Traceability: Link each requirement to specific comments
- Completeness: Cover functional and quality aspects

# Output Format:
## Functional Requirements
- [Requirement description]

## Non-Functional Requirements
- [Requirement description]
"""


class RequirementsGenerator:
    def __init__(self, model_path):
        # Cargar el modelo de lenguaje local
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32 #,
            #device_map="auto"
        )

        # Configurar el pipeline
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_length=4096,
            temperature=0.8,
            top_p=0.95,
            repetition_penalty=1.15
        )

        # Crear el modelo de LangChain
        self.llm = HuggingFacePipeline(pipeline=self.pipe)

        # Definir el prompt para extraer requisitos
        self.prompt_template = PromptTemplate(
            input_variables=["app_description", "comments"],
            template=PROMPT_TEMPLATE
        )

        # Crear la cadena de LangChain
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt_template)

    def generate_from_comments(self, comments, app_description):
        # Unir todos los comentarios en un solo texto
        comments_text = "".join([f"{i+1}- {comment.rstrip()}\n" for i, comment in enumerate(comments)])

        # Generar los requisitos
        result = self.chain.run(app_description=app_description, comments=comments_text)
        return result

    def generate_from_csv(self, csv_path, output_path=None, app_description=""):
        # Cargar los comentarios desde el CSV
        df = pd.read_csv(csv_path)

        # Asegurarse de que exista la columna de comentarios
        if 'Review' not in df.columns:
            raise ValueError("El CSV debe contener una columna llamada 'Review'")

        # Obtener la lista de comentarios
        comments = df['Review'].tolist()

        # Generar los requisitos
        print("Generando requisitos a partir de los comentarios...")
        requirements = self.generate_from_comments(comments, app_description)

        # Guardar los requisitos en un archivo si se especifica output_path
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(requirements)
            print(f"Requisitos guardados en: {output_path}")

        return requirements


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generar requisitos de software a partir de comentarios de usuarios')
    parser.add_argument('--input', type=str, default='../data/swiftkey_informative_cluster9 themes.csv',
                        help='Ruta al archivo CSV con comentarios filtrados')
    parser.add_argument('--output', type=str, default='../data/swiftKeyrequirements.txt',
                        help='Ruta para guardar los requisitos generados')
    parser.add_argument('--model', type=str, default='../models/Qwant 1.5B Destil',
                        help='Ruta al modelo de lenguaje local')
    parser.add_argument('--app-description', type=str, default='This is an app named SwiftKey that offers an alternative keyboard for your smartphone',
                        help='Descripción de la aplicación para contextualizar los comentarios')

    args = parser.parse_args()

    generator = RequirementsGenerator(args.model)
    requirements = generator.generate_from_csv(args.input, args.output, args.app_description)

    print("\nRequisitos generados:")
    print(requirements)


# #Quality Criteria
# - **Atomicity**: 1 requirement per feature
# - **Specificity**: Include verifiable metrics/criteria
# - **Traceability**: Link each requirement to specific comments
# - **Completeness**: Cover functional and quality aspects
