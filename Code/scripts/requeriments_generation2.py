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

# Task
Analyze the following user comments about the application described above and generate a well-defined list of software requirements.
Identify both functional and non-functional requirements (quality, performance, security, etc.).
For each requirement, assign a unique ID, specify its type (functional/non-functional),
and write a clear and precise description.

# User Comments:
{comments}

# Output Format:
## Functional Requirements
- FR001: [Description of the functional requirement]
- FR002: [Description of the functional requirement]
...

## Non-Functional Requirements
- NFR001: [Description of the non-functional requirement]
- NFR002: [Description of the non-functional requirement]
...

Please ensure that each requirement is:
1. Clear and unambiguous
2. Verifiable
3. Feasible
4. Relevant to the problem
5. Traceable to the source comments
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
            max_length=2048,
            temperature=0.7,
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
        comments_text = "".join([f"- {comment}" for comment in comments])

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
    parser.add_argument('--input', type=str, default='../data/swiftkey_informative_cluster16.csv',
                        help='Ruta al archivo CSV con comentarios filtrados')
    parser.add_argument('--output', type=str, default='../data/swiftKeyrequirements.txt',
                        help='Ruta para guardar los requisitos generados')
    parser.add_argument('--model', type=str, default='../models/Qwant 1.5B Destil',
                        help='Ruta al modelo de lenguaje local')
    parser.add_argument('--app-description', type=str, default='This is an app named SwiftKey that offers an alternative keyboard',
                        help='Descripción de la aplicación para contextualizar los comentarios')

    args = parser.parse_args()

    generator = RequirementsGenerator(args.model)
    requirements = generator.generate_from_csv(args.input, args.output, args.app_description)

    print("\nRequisitos generados:")
    print(requirements)