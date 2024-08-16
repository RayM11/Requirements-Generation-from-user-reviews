from customtkinter import *
from classes.System import System


class Interfaz(CTk):

    textbox_comment: CTkTextbox
    result_label: CTkEntry

    def __init__(self):
        super().__init__()

        self.geometry("500x400")
        self.title("Clasificador de opiniones")
        set_default_color_theme("dark-blue")

        label = CTkLabel(master=self, text="Introduzca un comentario:", font=("Arial", 20))

        self.textbox_comment = CTkTextbox(master=self, width=400, height=100, corner_radius=16, border_width=2)

        self.result_label = CTkEntry(master=self, placeholder_text=" ", width=400, state="normal")

        button_predict = CTkButton(master=self, text="Clasificar",
                                   corner_radius=32, border_width=2,
                                   command=self.predict_action)

        button_path = CTkButton(master=self, text="Categorizar un archivo CSV", corner_radius=32, border_width=2, command=self.predict_csv_action)

        label.pack(anchor="n", pady=(20, 0), ipady=5)
        self.textbox_comment.pack(anchor="n", pady=(20, 0), ipady=5)
        button_predict.pack(anchor="n", pady=(20, 0), padx=1)
        self.result_label.pack(anchor="n", pady=(20, 0), padx=1)
        button_path.pack(anchor="n", pady=(20, 0), padx=1)

    def predict_action(self):

        self.result_label.insert(0, "                                                                          ")

        print(self.textbox_comment.get("0.0", "end"))
        prediction = System().predict_relevance_comment(self.textbox_comment.get("0.0", "end"))
        print(prediction)

        self.result_label.insert(0, f"{("Relevante " if prediction > 0.5 else "No relevante ")} ({prediction})")

    def predict_csv_action(self):

        archivo = filedialog.askopenfilename(initialdir="/", title="Seleccione archivo", filetypes=[("Archivos CSV", "*.csv"), ("Todos los archivos", "*.*")])
        print(archivo)







